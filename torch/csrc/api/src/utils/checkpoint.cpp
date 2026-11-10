#include <torch/utils/checkpoint.h>

#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/Context.h>
#include <ATen/core/ivalue.h>
#include <c10/core/GradMode.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <mutex>
#include <utility>

namespace torch::utils {

namespace {

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

// --- RNG state helpers -----------------------------------------------------
//
// Mirrors torch/utils/checkpoint.py's get_device_states()/set_device_states()
// and the bare torch.get_rng_state()/torch.set_rng_state() calls for the CPU
// generator. `at::Generator` is the same abstraction Python's
// `device_module.get_rng_state()` bottoms out on (CPUGeneratorImpl /
// CUDAGeneratorImpl / etc. all implement get_state()/set_state() the same
// way), so none of this is CPU- or CUDA-specific: it compiles and works
// identically whichever accelerator backend (if any) is actually linked in.

at::Tensor get_generator_state(const at::Generator& gen_const) {
  // Generator::get_state()/set_state() are non-const (they go through the
  // shared GeneratorImpl*), even though copying the Generator wrapper itself
  // is just an intrusive_ptr bump -- see at::manual_seed() in
  // aten/src/ATen/Context.h for the exact same "copy to a local, lock, use"
  // pattern this replicates.
  at::Generator gen = gen_const;
  std::lock_guard<std::mutex> lock(gen.mutex());
  return gen.get_state();
}

void set_generator_state(const at::Generator& gen_const, const at::Tensor& state) {
  at::Generator gen = gen_const;
  std::lock_guard<std::mutex> lock(gen.mutex());
  gen.set_state(state);
}

at::Tensor get_cpu_rng_state() {
  return get_generator_state(at::globalContext().defaultGenerator(at::kCPU));
}

void set_cpu_rng_state(const at::Tensor& state) {
  set_generator_state(at::globalContext().defaultGenerator(at::kCPU), state);
}

// Every distinct non-CPU/meta device seen among `tensors`, first-seen order,
// plus that device's current generator state -- mirrors
// torch/utils/checkpoint.py's get_device_states().
std::pair<std::vector<at::Device>, std::vector<at::Tensor>> get_device_states(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> devices;
  std::vector<at::Tensor> states;
  for (const auto& t : tensors) {
    if (!t.defined()) {
      continue;
    }
    const auto device = t.device();
    if (device.type() == at::kCPU || device.type() == at::kMeta) {
      continue;
    }
    if (std::any_of(
            devices.begin(), devices.end(), [&](const at::Device& d) {
              return d == device;
            })) {
      continue;
    }
    devices.push_back(device);
    states.push_back(
        get_generator_state(at::globalContext().defaultGenerator(device)));
  }
  return {std::move(devices), std::move(states)};
}

void set_device_states(
    const std::vector<at::Device>& devices,
    const std::vector<at::Tensor>& states) {
  TORCH_INTERNAL_ASSERT(devices.size() == states.size());
  for (const auto i : c10::irange(devices.size())) {
    set_generator_state(
        at::globalContext().defaultGenerator(devices[i]), states[i]);
  }
}

// RAII guard that saves the RNG state ambient *right now* (CPU + every
// device in `devices`) on construction, and restores exactly that state on
// destruction -- the save/restore pair `torch.random.fork_rng()` does around
// the body of its `with` block. A no-op if `enabled` is false, matching
// `fork_rng(enabled=...)`.
struct AmbientRngStateGuard {
  AmbientRngStateGuard(bool enabled, std::vector<at::Device> devices)
      : enabled_(enabled), devices_(std::move(devices)) {
    if (!enabled_) {
      return;
    }
    cpu_state_ = get_cpu_rng_state();
    device_states_.reserve(devices_.size());
    for (const auto& d : devices_) {
      device_states_.push_back(
          get_generator_state(at::globalContext().defaultGenerator(d)));
    }
  }

  AmbientRngStateGuard(const AmbientRngStateGuard&) = delete;
  AmbientRngStateGuard& operator=(const AmbientRngStateGuard&) = delete;

  ~AmbientRngStateGuard() {
    if (!enabled_) {
      return;
    }
    set_cpu_rng_state(cpu_state_);
    set_device_states(devices_, device_states_);
  }

 private:
  bool enabled_;
  std::vector<at::Device> devices_;
  at::Tensor cpu_state_;
  std::vector<at::Tensor> device_states_;
};

// --- the checkpointed autograd Function -------------------------------------

at::Tensor detach_preserving_requires_grad(const at::Tensor& t) {
  if (!t.defined()) {
    return t;
  }
  at::Tensor out = t.detach();
  out.requires_grad_(t.requires_grad());
  return out;
}

// Type-erased holder for the user's checkpointed function plus the
// RNG/config bookkeeping that has to survive from the forward pass to the
// backward/recompute pass. `torch::autograd::Function<T>::forward`/
// `::backward` are *static* methods -- the only per-invocation storage they
// get is `AutogradContext::saved_data`, a `map<string, at::IValue>` -- and a
// runtime `std::function` closure doesn't fit naturally into an IValue
// except via this "Capsule" mechanism. We stash the whole state object (not
// just the closure) in one capsule rather than trying to split the rest
// across multiple saved_data entries, since e.g. `std::vector<at::Device>`
// has no natural IValue representation either, and this way every field
// here is a plain, compiler-checked C++ type instead of something we have to
// IValue::to<T>() back out.
//
// `IValue::make_capsule`/`toCapsule` carry an explicit "no new callsites"
// comment in ivalue.h -- that's about not hand-rolling ad hoc script-visible
// custom classes in place of the proper `torch::class_<T>` registration
// system, for types that need to be introspectable by TorchScript/
// torch.export/serialization/compiled autograd. Nothing here is ever any of
// those: this capsule is created and consumed within a single, eager,
// in-process backward pass and never escapes it, which is exactly the
// "internal implementation detail" use case the Capsule tag itself (per its
// own doc comment one line up) exists for.
struct CheckpointState : public torch::CustomClassHolder {
  explicit CheckpointState(CheckpointFunction f, bool preserve_rng)
      : fn(std::move(f)), preserve_rng_state(preserve_rng) {}

  CheckpointFunction fn;
  bool preserve_rng_state;

  // Populated in forward() iff preserve_rng_state; consumed in backward().
  at::Tensor fwd_cpu_rng_state;
  std::vector<at::Device> fwd_devices;
  std::vector<at::Tensor> fwd_device_rng_states;
};

struct CheckpointFunctionNode
    : public torch::autograd::Function<CheckpointFunctionNode> {
  static variable_list forward(
      AutogradContext* ctx,
      const at::IValue& state_capsule,
      at::TensorList inputs);

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_outputs);
};

variable_list CheckpointFunctionNode::forward(
    AutogradContext* ctx,
    const at::IValue& state_capsule,
    at::TensorList inputs) {
  auto state = c10::static_intrusive_pointer_cast<CheckpointState>(
      state_capsule.toCapsule());

  if (!std::any_of(
          inputs.begin(), inputs.end(), [](const at::Tensor& t) {
            return t.defined() && t.requires_grad();
          })) {
    TORCH_WARN(
        "torch::utils::checkpoint: none of the inputs have requires_grad "
        "= true. Gradients will be undefined for all of them.");
  }

  std::vector<at::Tensor> input_vec(inputs.begin(), inputs.end());

  if (state->preserve_rng_state) {
    state->fwd_cpu_rng_state = get_cpu_rng_state();
    std::tie(state->fwd_devices, state->fwd_device_rng_states) =
        get_device_states(input_vec);
  }

  ctx->saved_data["state"] = state_capsule;
  ctx->save_for_backward(input_vec);

  std::vector<at::Tensor> outputs;
  {
    // Function<T>::apply() already runs forward() with grad mode disabled
    // (see CppNode<T>::apply in custom_function.h); this NoGradGuard is
    // defensive and self-documenting, mirroring Python's explicit
    // `with torch.no_grad():` around this same call.
    c10::NoGradGuard no_grad;
    outputs = state->fn(input_vec);
  }
  TORCH_CHECK(
      !outputs.empty(),
      "torch::utils::checkpoint: fn must return at least one tensor");
  return variable_list(outputs.begin(), outputs.end());
}

variable_list CheckpointFunctionNode::backward(
    AutogradContext* ctx,
    variable_list grad_outputs) {
  TORCH_CHECK(
      torch::autograd::Engine::get_default_engine().is_checkpoint_valid(),
      "torch::utils::checkpoint is incompatible with torch::autograd::grad() "
      "or with passing a non-empty `inputs` list to torch::autograd::backward() "
      "in an outer backward call that reaches this checkpointed region. Call "
      "torch::autograd::backward() without restricting `inputs` instead.");

  auto state = c10::static_intrusive_pointer_cast<CheckpointState>(
      ctx->saved_data["state"].toCapsule());

  variable_list saved_inputs = ctx->get_saved_variables();

  // Saves/restores whatever RNG state is ambient when backward() is entered,
  // around the block below that temporarily overwrites it with the
  // forward-time state for the recompute. Matches
  // torch.random.fork_rng(devices=fwd_devices, enabled=preserve_rng_state).
  AmbientRngStateGuard ambient_rng_guard(
      state->preserve_rng_state, state->fwd_devices);

  if (state->preserve_rng_state) {
    set_cpu_rng_state(state->fwd_cpu_rng_state);
    set_device_states(state->fwd_devices, state->fwd_device_rng_states);
  }

  std::vector<at::Tensor> detached_inputs;
  detached_inputs.reserve(saved_inputs.size());
  for (const auto& t : saved_inputs) {
    detached_inputs.push_back(detach_preserving_requires_grad(t));
  }

  std::vector<at::Tensor> outputs;
  {
    // Recompute WITH grad tracking this time, to rebuild a graph rooted at
    // `detached_inputs` that we can immediately run backward through below.
    c10::AutoGradMode enable_grad(true);
    outputs = state->fn(detached_inputs);
  }
  // ambient_rng_guard restores the pre-backward() RNG state when it goes out
  // of scope at the end of this function -- including if state->fn() above
  // throws, same as Python's try/finally around the recompute.

  TORCH_CHECK(
      outputs.size() == grad_outputs.size(),
      "torch::utils::checkpoint: the recomputed forward pass returned ",
      outputs.size(),
      " tensor(s) but the original forward pass returned ",
      grad_outputs.size(),
      "; fn must be deterministic in the number of outputs it returns.");

  variable_list outputs_with_grad;
  variable_list grads_for_outputs;
  for (const auto i : c10::irange(outputs.size())) {
    if (outputs[i].defined() && outputs[i].requires_grad()) {
      outputs_with_grad.push_back(outputs[i]);
      grads_for_outputs.push_back(grad_outputs[i]);
    }
  }
  TORCH_CHECK(
      !outputs_with_grad.empty(),
      "torch::utils::checkpoint: none of the recomputed outputs require "
      "grad; this checkpoint() call is unnecessary.");

  // Reentrant call back into the autograd engine -- this is what makes it
  // "reentrant" checkpointing. Deliberately torch::autograd::backward(), not
  // torch::autograd::grad() with an explicit `inputs` list: the latter is
  // exactly the combination the is_checkpoint_valid() check above guards
  // against (see Python's identical choice in CheckpointFunction.backward).
  torch::autograd::backward(outputs_with_grad, grads_for_outputs);

  variable_list grad_inputs;
  grad_inputs.reserve(1 + detached_inputs.size());
  grad_inputs.emplace_back(); // placeholder: the capsule arg isn't a Variable
  for (const auto& t : detached_inputs) {
    grad_inputs.push_back(t.grad());
  }
  return grad_inputs;
}

} // namespace

std::vector<Tensor> checkpoint(
    CheckpointFunction fn,
    std::vector<Tensor> inputs,
    bool preserve_rng_state) {
  auto state = c10::make_intrusive<CheckpointState>(
      std::move(fn), preserve_rng_state);
  at::IValue state_capsule = at::IValue::make_capsule(std::move(state));
  at::TensorList input_list = inputs;
  variable_list outputs =
      CheckpointFunctionNode::apply(state_capsule, input_list);
  return std::vector<Tensor>(outputs.begin(), outputs.end());
}

} // namespace torch::utils
