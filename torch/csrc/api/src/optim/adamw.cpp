#include <torch/optim/adamw.h>

#include <torch/utils.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <cmath>
#include <functional>
#include <map>

namespace torch::optim {

namespace {

// --- lbtorch extension helpers (foreach / fused / capturable) ---
//
// The multi-tensor ("foreach") and fused kernels require every tensor in a
// given call to share one device and one dtype. Python achieves this via
// `Optimizer._group_tensors_by_device_and_dtype`
// (torch/utils/_foreach_utils.py, implemented in `torch._C`); no equivalent
// utility is exposed at the C++ frontend layer, so we implement the same
// grouping ourselves. This is intentionally a plain, simple grouping (a
// `std::map` keyed by device+dtype) rather than a port of the exact Python
// internals -- correct and sufficient for every real target workload (GRPO
// FSDP training keeps every param of a given optimizer param group on one
// device and, typically, one dtype), but not claimed to be as allocation-
// lean as the upstream implementation. See docs/adamw_checkpoint_interop.md.
struct DeviceDtypeKey {
  c10::DeviceType device_type;
  c10::DeviceIndex device_index;
  at::ScalarType dtype;
  bool operator<(const DeviceDtypeKey& other) const {
    if (device_type != other.device_type) {
      return device_type < other.device_type;
    }
    if (device_index != other.device_index) {
      return device_index < other.device_index;
    }
    return dtype < other.dtype;
  }
};

DeviceDtypeKey key_for(const Tensor& t) {
  return DeviceDtypeKey{t.device().type(), t.device().index(), t.scalar_type()};
}

struct ForeachBucket {
  std::vector<Tensor> params;
  std::vector<Tensor> grads;
  std::vector<Tensor> exp_avgs;
  std::vector<Tensor> exp_avg_sqs;
  std::vector<Tensor> max_exp_avg_sqs; // only populated when amsgrad
  std::vector<Tensor> step_tensors; // only populated when capturable/fused
  std::vector<int64_t> host_steps; // only populated when !capturable (post-increment value)
};

// Upstream Python hard-errors `capturable=True` on a device outside
// {cuda, xpu, hpu, <privateuse1>, xla} (see `_get_capturable_supported_devices`
// in torch/optim/optimizer.py) because CPU tensors can't be captured into a
// CUDA graph and gain nothing from the device-resident step trick. lbtorch
// relaxes this to a warning so the capturable code path (identical math,
// just device- instead of host-resident bookkeeping) can be exercised and
// unit tested in environments without a CUDA device. This is a deliberate,
// documented departure from upstream's device gating, not a numerics
// difference -- see docs/adamw_checkpoint_interop.md, "capturable on CPU".
void maybe_warn_capturable_cpu(const Tensor& p) {
  if (p.device().is_cpu()) {
    TORCH_WARN_ONCE(
        "torch::optim::AdamWOptions::capturable(true) used with a CPU "
        "tensor. Upstream PyTorch disallows this (CPU can't be CUDA-graph "
        "captured). lbtorch allows it so the capturable path can be unit "
        "tested without a GPU; it is numerically identical to the "
        "non-capturable path on CPU, merely slower. Do not rely on this for "
        "production CPU training -- use capturable=false on CPU.");
  }
}

} // namespace

AdamWOptions::AdamWOptions(double lr) : lr_(lr) {}

bool operator==(const AdamWOptions& lhs, const AdamWOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&
      (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas())) &&
      (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas())) &&
      (lhs.eps() == rhs.eps()) && (lhs.weight_decay() == rhs.weight_decay()) &&
      (lhs.amsgrad() == rhs.amsgrad()) && (lhs.foreach() == rhs.foreach()) &&
      (lhs.fused() == rhs.fused()) && (lhs.capturable() == rhs.capturable());
}

void AdamWOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(amsgrad);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(foreach);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(fused);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(capturable);
}

void AdamWOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, amsgrad);
  // foreach/fused/capturable are new fields: archives written before this
  // change won't have them. Use try_read with a false default instead of
  // `_TORCH_OPTIM_DESERIALIZE_TORCH_ARG`, whose missing-key path
  // TORCH_INTERNAL_ASSERTs for non-Tensor types (it was only ever written
  // for required fields) -- that would make this a breaking change for
  // every archive saved by the pre-existing AdamW. All three default to
  // `false`, matching `AdamWOptions`'s defaults, if absent.
  {
    c10::IValue ivalue;
    if (archive.try_read("foreach", ivalue)) {
      foreach(ivalue.to<bool>());
    }
  }
  {
    c10::IValue ivalue;
    if (archive.try_read("fused", ivalue)) {
      fused(ivalue.to<bool>());
    }
  }
  {
    c10::IValue ivalue;
    if (archive.try_read("capturable", ivalue)) {
      capturable(ivalue.to<bool>());
    }
  }
}

double AdamWOptions::get_lr() const {
  return lr();
}

void AdamWOptions::set_lr(const double lr) {
  this->lr(lr);
}

bool operator==(const AdamWParamState& lhs, const AdamWParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
      torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
      torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq()) &&
      torch::equal_if_defined(lhs.max_exp_avg_sq(), rhs.max_exp_avg_sq()) &&
      torch::equal_if_defined(lhs.step_tensor(), rhs.step_tensor());
}

void AdamWParamState::serialize(
    torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_exp_avg_sq);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step_tensor);
}

void AdamWParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg_sq);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, max_exp_avg_sq);
  // Tensor type -> safe to use the macro: its missing-key path only
  // TORCH_INTERNAL_ASSERTs when the field is NOT a Tensor, so an archive
  // saved before this change (no `step_tensor` key) deserializes fine and
  // simply leaves `step_tensor` undefined, exactly as a freshly constructed
  // state would have it.
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, step_tensor);
}

Tensor AdamW::step(LossClosure closure) {
  NoGradGuard no_grad;
  Tensor loss = {};
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  for (auto& group : param_groups_) {
    auto& options = static_cast<AdamWOptions&>(group.options());
    TORCH_CHECK(
        !(options.fused() && options.foreach()),
        "`fused` and `foreach` cannot both be true");
    if (options.fused()) {
      step_fused(group, options);
    } else if (options.foreach()) {
      step_foreach(group, options);
    } else {
      step_single_tensor(group, options);
    }
  }
  return loss;
}

// Original algorithm, extended with a `capturable` branch. Mirrors
// `_single_tensor_adam` in torch/optim/adam.py (decoupled-weight-decay /
// AdamW branch).
void AdamW::step_single_tensor(
    OptimizerParamGroup& group,
    AdamWOptions& options) {
  const bool capturable = options.capturable();
  for (auto& p : group.params()) {
    if (!p.grad().defined()) {
      continue;
    }
    const auto& grad = p.grad();
    TORCH_CHECK(
        !grad.is_sparse(),
        "AdamW does not support sparse gradients" /*, please consider SparseAdamW instead*/);
    auto param_state = state_.find(p.unsafeGetTensorImpl());

    // Perform step weight decay (decoupled -- AdamW, not Adam)
    if (options.weight_decay() != 0) {
      p.mul_(1 - options.lr() * options.weight_decay());
    }

    // State initialization
    if (param_state == state_.end()) {
      auto state = std::make_unique<AdamWParamState>();
      state->step(0);
      state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
      state->exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
      if (options.amsgrad()) {
        state->max_exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
      }
      if (capturable) {
        maybe_warn_capturable_cpu(p);
        // dtype matches Python's `_get_scalar_dtype(is_fused=False)` under
        // the (near-universal in practice) assumption that the default
        // torch dtype is float32 -- see docs/adamw_checkpoint_interop.md.
        state->step_tensor(
            torch::zeros({}, p.options().dtype(torch::kFloat32)));
      }
      state_[p.unsafeGetTensorImpl()] = std::move(state);
    }

    auto& state =
        static_cast<AdamWParamState&>(*state_[p.unsafeGetTensorImpl()]);
    auto& exp_avg = state.exp_avg();
    auto& exp_avg_sq = state.exp_avg_sq();
    auto& max_exp_avg_sq = state.max_exp_avg_sq();

    auto beta1 = std::get<0>(options.betas());
    auto beta2 = std::get<1>(options.betas());

    // Decay the first and second moment running average coefficient
    exp_avg.lerp_(grad, 1 - beta1);
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

    if (capturable) {
      if (!state.step_tensor().defined()) {
        // Lazily materialize a tensor step for state created before
        // `capturable` was turned on (e.g. loaded from an older archive, or
        // from a group where capturable was flipped after the fact).
        maybe_warn_capturable_cpu(p);
        state.step_tensor(torch::full(
            {}, static_cast<double>(state.step()), p.options().dtype(torch::kFloat32)));
      }
      // Best-effort host mirror: kept so `state.step()` stays readable for
      // logging/debugging in eager (non-captured) use. NOT authoritative
      // once this step has run as part of a replayed CUDA graph, since host
      // instructions between replays don't execute -- see adamw.h doc on
      // `step_tensor`. The device tensor is the only value the math below
      // actually reads.
      state.step(state.step() + 1);
      state.step_tensor().add_(1);

      // bias_correction{1,2} as device tensors, mirroring the capturable
      // branch of `_single_tensor_adam` exactly (no `.item()` anywhere).
      auto bias_correction1 = 1 - torch::pow(beta1, state.step_tensor());
      auto bias_correction2 = 1 - torch::pow(beta2, state.step_tensor());
      auto step_size = options.lr() / bias_correction1;
      auto step_size_neg = -step_size;
      auto bias_correction2_sqrt = bias_correction2.sqrt();

      Tensor denom;
      if (options.amsgrad()) {
        torch::max_out(max_exp_avg_sq, exp_avg_sq, max_exp_avg_sq);
        denom = (max_exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg))
                    .add_(options.eps() / step_size_neg);
      } else {
        denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg))
                    .add_(options.eps() / step_size_neg);
      }
      // step_size is folded into `denom` above (it's a Tensor, and
      // `addcdiv_`'s `value` arg must be a Scalar), exactly as upstream
      // does for the same reason.
      p.addcdiv_(exp_avg, denom);
    } else {
      state.step(state.step() + 1);

      auto bias_correction1 = 1 - std::pow(beta1, state.step());
      auto bias_correction2 = 1 - std::pow(beta2, state.step());

      Tensor denom;
      if (options.amsgrad()) {
        torch::max_out(max_exp_avg_sq, exp_avg_sq, max_exp_avg_sq);
        denom = (max_exp_avg_sq.sqrt() / sqrt(bias_correction2))
                    .add_(options.eps());
      } else {
        denom =
            (exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(options.eps());
      }

      auto step_size = options.lr() / bias_correction1;
      p.addcdiv_(exp_avg, denom, -step_size);
    }
  }
}

// Multi-tensor ("foreach") path. Mirrors `_multi_tensor_adam` in
// torch/optim/adam.py (decoupled-weight-decay branch), including its
// `capturable` sub-branch (`foreach=true, capturable=true` is a supported,
// tested combination, same as upstream).
void AdamW::step_foreach(OptimizerParamGroup& group, AdamWOptions& options) {
  const bool capturable = options.capturable();
  const bool amsgrad = options.amsgrad();
  auto beta1 = std::get<0>(options.betas());
  auto beta2 = std::get<1>(options.betas());

  std::map<DeviceDtypeKey, ForeachBucket> buckets;

  for (auto& p : group.params()) {
    if (!p.grad().defined()) {
      continue;
    }
    const auto& grad = p.grad();
    TORCH_CHECK(!grad.is_sparse(), "AdamW does not support sparse gradients");

    if (options.weight_decay() != 0) {
      p.mul_(1 - options.lr() * options.weight_decay());
    }

    auto param_state = state_.find(p.unsafeGetTensorImpl());
    if (param_state == state_.end()) {
      auto state = std::make_unique<AdamWParamState>();
      state->step(0);
      state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
      state->exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
      if (amsgrad) {
        state->max_exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
      }
      if (capturable) {
        maybe_warn_capturable_cpu(p);
        state->step_tensor(
            torch::zeros({}, p.options().dtype(torch::kFloat32)));
      }
      state_[p.unsafeGetTensorImpl()] = std::move(state);
    }

    auto& state =
        static_cast<AdamWParamState&>(*state_[p.unsafeGetTensorImpl()]);
    if (capturable && !state.step_tensor().defined()) {
      maybe_warn_capturable_cpu(p);
      state.step_tensor(torch::full(
          {}, static_cast<double>(state.step()), p.options().dtype(torch::kFloat32)));
    }

    auto& bucket = buckets[key_for(p)];
    bucket.params.push_back(p);
    bucket.grads.push_back(grad);
    bucket.exp_avgs.push_back(state.exp_avg());
    bucket.exp_avg_sqs.push_back(state.exp_avg_sq());
    if (amsgrad) {
      bucket.max_exp_avg_sqs.push_back(state.max_exp_avg_sq());
    }
    if (capturable) {
      bucket.step_tensors.push_back(state.step_tensor());
    } else {
      // Host-side step bookkeeping happens here (not in a foreach call) --
      // it's bookkeeping, not a tensor op, so there's no kernel-launch cost
      // to batch away.
      state.step(state.step() + 1);
      bucket.host_steps.push_back(state.step());
    }
  }

  for (auto& entry : buckets) {
    auto& b = entry.second;
    if (b.params.empty()) {
      continue;
    }

    if (capturable) {
      at::_foreach_add_(b.step_tensors, 1.0);
    }

    at::_foreach_lerp_(b.exp_avgs, b.grads, 1 - beta1);
    at::_foreach_mul_(b.exp_avg_sqs, beta2);
    at::_foreach_addcmul_(b.exp_avg_sqs, b.grads, b.grads, 1 - beta2);

    if (capturable) {
      // Exactly mirrors the capturable branch of `_multi_tensor_adam`:
      // bias_correction1 ends up holding -lr/(1-beta1^t) (already negative)
      // after sub(1)/div(lr)/reciprocal; bias_correction2 ends up holding
      // sqrt(1-beta2^t) after sub(1)/neg/sqrt. No further negation of
      // either is performed -- the later `foreach_div_` by the (negative)
      // step_size is what flips the final sign, same as upstream.
      auto bias_correction1 = at::_foreach_pow(beta1, b.step_tensors);
      auto bias_correction2 = at::_foreach_pow(beta2, b.step_tensors);
      at::_foreach_sub_(bias_correction1, 1.0);
      at::_foreach_sub_(bias_correction2, 1.0);
      at::_foreach_neg_(bias_correction2);
      at::_foreach_div_(bias_correction1, options.lr());
      at::_foreach_reciprocal_(bias_correction1);
      at::_foreach_sqrt_(bias_correction2);

      auto& step_size = bias_correction1; // already = -lr/(1-beta1^t)
      auto& bias_correction2_sqrt = bias_correction2;

      std::vector<Tensor> exp_avg_sq_sqrt;
      if (amsgrad) {
        at::_foreach_maximum_(b.max_exp_avg_sqs, b.exp_avg_sqs);
        exp_avg_sq_sqrt = at::_foreach_sqrt(b.max_exp_avg_sqs);
      } else {
        exp_avg_sq_sqrt = at::_foreach_sqrt(b.exp_avg_sqs);
      }
      at::_foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt);
      at::_foreach_add_(exp_avg_sq_sqrt, options.eps());
      at::_foreach_div_(exp_avg_sq_sqrt, step_size);
      at::_foreach_addcdiv_(b.params, b.exp_avgs, exp_avg_sq_sqrt, 1.0);
    } else {
      // `_foreach_div_`/`_foreach_addcdiv_`'s per-tensor-scalar ("ScalarList")
      // overloads take `ArrayRef<Scalar>`, not `ArrayRef<double>` -- Scalar
      // doesn't have an implicit conversion from double, so these must be
      // built as `at::Scalar` directly (mirrors `torch._foreach_div_(t,
      // bias_correction2_sqrt)` where Python's list-of-float is converted to
      // a ScalarList by pybind11 automatically; we do that conversion by
      // hand here).
      std::vector<at::Scalar> step_size, bias_correction2_sqrt;
      step_size.reserve(b.host_steps.size());
      bias_correction2_sqrt.reserve(b.host_steps.size());
      for (auto s : b.host_steps) {
        double bc1 = 1 - std::pow(beta1, s);
        double bc2 = 1 - std::pow(beta2, s);
        step_size.emplace_back(-(options.lr() / bc1));
        bias_correction2_sqrt.emplace_back(std::sqrt(bc2));
      }

      std::vector<Tensor> exp_avg_sq_sqrt;
      if (amsgrad) {
        at::_foreach_maximum_(b.max_exp_avg_sqs, b.exp_avg_sqs);
        exp_avg_sq_sqrt = at::_foreach_sqrt(b.max_exp_avg_sqs);
      } else {
        exp_avg_sq_sqrt = at::_foreach_sqrt(b.exp_avg_sqs);
      }
      at::_foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt);
      at::_foreach_add_(exp_avg_sq_sqrt, options.eps());
      at::_foreach_addcdiv_(b.params, b.exp_avgs, exp_avg_sq_sqrt, step_size);
    }
  }
}

// True fused path: dispatches straight to `at::_fused_adamw_`, the same
// native op Python's `_fused_adam` (decoupled_weight_decay branch) calls.
// This is a single multi-tensor-apply CUDA kernel launch per device/dtype
// bucket on CUDA (one kernel does the whole bucket, vs. one kernel per op
// per tensor in the foreach path) -- we do not reimplement the math; we
// call the real registered kernel (CPU/CUDA/MPS/XPU dispatch is handled by
// the op's own dispatch table, see
// aten/src/ATen/native/native_functions.yaml:_fused_adamw_).
void AdamW::step_fused(OptimizerParamGroup& group, AdamWOptions& options) {
  const bool amsgrad = options.amsgrad();
  auto beta1 = std::get<0>(options.betas());
  auto beta2 = std::get<1>(options.betas());

  std::map<DeviceDtypeKey, ForeachBucket> buckets;

  for (auto& p : group.params()) {
    if (!p.grad().defined()) {
      continue;
    }
    const auto& grad = p.grad();
    TORCH_CHECK(!grad.is_sparse(), "AdamW does not support sparse gradients");
    TORCH_CHECK(
        p.is_floating_point(),
        "`fused=True` requires all params to be floating point, got ",
        p.scalar_type());

    auto param_state = state_.find(p.unsafeGetTensorImpl());
    if (param_state == state_.end()) {
      auto state = std::make_unique<AdamWParamState>();
      state->step(0);
      state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
      state->exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
      if (amsgrad) {
        state->max_exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
      }
      // Fused always uses a device-resident tensor step, capturable or not
      // -- matches `_init_group`'s `group["capturable"] or group["fused"]`
      // condition in torch/optim/adam.py.
      state->step_tensor(
          torch::zeros({}, p.options().dtype(torch::kFloat32)));
      state_[p.unsafeGetTensorImpl()] = std::move(state);
    }

    auto& state =
        static_cast<AdamWParamState&>(*state_[p.unsafeGetTensorImpl()]);
    if (!state.step_tensor().defined()) {
      state.step_tensor(torch::full(
          {}, static_cast<double>(state.step()), p.options().dtype(torch::kFloat32)));
    }

    auto& bucket = buckets[key_for(p)];
    bucket.params.push_back(p);
    bucket.grads.push_back(grad);
    bucket.exp_avgs.push_back(state.exp_avg());
    bucket.exp_avg_sqs.push_back(state.exp_avg_sq());
    if (amsgrad) {
      bucket.max_exp_avg_sqs.push_back(state.max_exp_avg_sq());
    }
    bucket.step_tensors.push_back(state.step_tensor());
  }

  for (auto& entry : buckets) {
    auto& b = entry.second;
    if (b.params.empty()) {
      continue;
    }
    // Python's `_fused_adam` increments the external step tensors itself
    // (via `torch._foreach_add_`) before calling the kernel; the kernel's
    // own step handling is purely internal bookkeeping for the fused math.
    at::_foreach_add_(b.step_tensors, 1.0);
    at::_fused_adamw_(
        b.params,
        b.grads,
        b.exp_avgs,
        b.exp_avg_sqs,
        amsgrad ? b.max_exp_avg_sqs : std::vector<Tensor>{},
        b.step_tensors,
        options.lr(),
        beta1,
        beta2,
        options.weight_decay(),
        options.eps(),
        amsgrad,
        /*maximize=*/false);
  }
}

void AdamW::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void AdamW::load(serialize::InputArchive& archive) {
  IValue pytorch_version;
  if (archive.try_read("pytorch_version", pytorch_version)) {
    serialize(*this, archive);
  } else { // deserializing archives saved in old format (prior to
           // version 1.5.0)
    TORCH_WARN(
        "Your serialized AdamW optimizer is still using the old serialization format. "
        "You should re-save your AdamW optimizer to use the new serialization format.");
    std::vector<int64_t> step_buffers;
    std::vector<at::Tensor> exp_average_buffers;
    std::vector<at::Tensor> exp_average_sq_buffers;
    std::vector<at::Tensor> max_exp_average_sq_buffers;
    torch::optim::serialize(archive, "step_buffers", step_buffers);
    torch::optim::serialize(
        archive, "exp_average_buffers", exp_average_buffers);
    torch::optim::serialize(
        archive, "exp_average_sq_buffers", exp_average_sq_buffers);
    torch::optim::serialize(
        archive, "max_exp_average_sq_buffers", max_exp_average_sq_buffers);
    // since there were no param_groups prior to version 1.5.0, assuming all
    // tensors are now in one param_group
    std::vector<Tensor> params = param_groups_.at(0).params();
    for (const auto idx : c10::irange(step_buffers.size())) {
      auto state = std::make_unique<AdamWParamState>();
      state->step(step_buffers.at(idx));
      state->exp_avg(exp_average_buffers.at(idx));
      state->exp_avg_sq(exp_average_sq_buffers.at(idx));
      if (idx < max_exp_average_sq_buffers.size()) {
        state->max_exp_avg_sq(max_exp_average_sq_buffers.at(idx));
      }
      state_[params.at(idx).unsafeGetTensorImpl()] = std::move(state);
    }
  }
}
} // namespace torch::optim
