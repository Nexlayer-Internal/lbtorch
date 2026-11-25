#pragma once

// C++ FSDP orchestration layer: FSDP-DESIGN.md Sec 4, "Phase 1-3".
//
// This is the sharding/scheduling layer PLAN.md identified as the largest
// remaining piece of Python logic on the GRPO training hot path (row 11/12,
// ~13,000 lines of Python, 55-70% genuine logic). It is built, deliberately,
// as a new, focused C++ component -- not a line-for-line port of FSDP1's
// _flat_param.py/_init_utils.py or FSDP2's _fsdp_param_group.py -- that
// reproduces the real behavior those files implement for the validated GRPO
// use case (Qwen3-4B-scale actor models, AdamW, gradient checkpointing,
// 2-GPU sharding), per this task's explicit scope: "correct and fast for the
// real GRPO training use case is the bar, not perfect fidelity to every
// Python FSDP edge case."
//
// What's ported (the genuine logic PLAN.md flagged):
//   - FlatParameter construction: flatten + pad + shard a wrapped unit's
//     parameters into one contiguous per-rank buffer (FlatParameter concept,
//     PLAN.md row 11).
//   - Shard-offset math: every original parameter's [offset, offset+numel)
//     slice within the flat buffer, used both to reconstruct per-parameter
//     views after all-gather and to re-pack per-parameter gradients into one
//     flat buffer before reduce-scatter.
//   - The "unshard" / "reshard" state machine: all-gather before forward,
//     all-gather again before backward (a separate call -- the whole reason
//     resharding after forward is a real memory win rather than a no-op),
//     reduce-scatter after backward, immediate release of the full
//     (unsharded) buffer on both ends.
//
// What's modeled on `Reducer.cpp` (FSDP-DESIGN.md Sec 3.1), NOT copied from
// it (Reducer is DDP's, unsharded -- one all-reduce per bucket; FSDP needs
// two distinct, shard-aware collective shapes at two distinct points):
//   - Autograd hook registration pattern: a post-accumulate-grad hook per
//     original parameter, registered once at construction via
//     `torch::autograd::impl::grad_accumulator(...)->add_post_hook(...)`,
//     exactly `Reducer::autograd_hook`'s mechanism, firing once per
//     parameter per backward pass.
//   - "Bucket readiness" becomes "all of this unit's original parameters
//     have reported a gradient" -> fire the unit's single reduce_scatter.
//
// What's new, not present in either Reducer or Python FSDP's exact shape:
//   - The "re-unshard before backward" trigger is a `Tensor::register_hook`
//     on the unit's forward *output* tensor(s), not a dedicated
//     `_pre_backward_hook` module method (libtorch's `torch::nn::Module` has
//     no generic forward-hook registration API the way Python's
//     `nn.Module.register_forward_hook` does) -- functionally equivalent:
//     it fires exactly when the autograd engine is about to deliver a
//     gradient to that tensor, i.e. immediately before backward executes
//     through this unit's internal ops, which is the same timing Python
//     FSDP's `_pre_backward_hook` achieves via `register_hook` on its own
//     output tensor(s) (`_runtime_utils.py::_register_pre_backward_hooks`).
//
// Parameter-swap mechanism: `Tensor::set_data()` (the same primitive behind
// Python's `param.data = x`, and specifically the same primitive behind the
// `flat_param_to`/`_use_unsharded_views` mechanism PLAN.md's incident
// traces to). `set_data()` does NOT bump the target Variable's version
// counter and does NOT change the Variable's identity -- the same
// `AccumulateGrad` node (and therefore the same post-hook registered once at
// construction) keeps firing correctly across every step even though the
// Variable's underlying storage is swapped out and back in every forward/
// backward. This is why hooks are registered once in the constructor, not
// per-step.
//
// Scope / known limitations (acceptable for v1 per the task brief):
//   - One flat parameter per wrapped unit (classic FSDP1 "use_orig_params=
//     False" shape) is literally what the optimizer trains -- not FSDP2's
//     per-original-parameter DTensor sharding. Simpler, and sufficient: the
//     validated workload wraps whole transformer blocks / linear stacks,
//     not individual tensors that need independent optimizer hyperparams.
//   - No prefetch/overlap of the next unit's all-gather with the current
//     unit's compute (FSDP2's async scheduling, PLAN.md row 12). Units are
//     unsharded and resharded strictly one at a time -- correct and
//     synchronous, not yet throughput-optimal. Explicitly out of scope per
//     the task's priority ordering (correctness/no-hang first).
//   - Equal-shard-size only (mirrors collective_transport's own current
//     scope -- `host_staged_transport.cpp` asserts equal shards). Padding
//     (`padded_numel_`) handles any wrapped unit whose true numel isn't a
//     multiple of world_size; FSDP's true last-shard-uneven-size handling
//     is not implemented.
//   - Single-node only (inherits this scope directly from
//     collective_transport's `make_transport`).

#include <ATen/core/Tensor.h>
#include <torch/csrc/distributed/c10d/collective_transport/transport.h>
#include <torch/nn/module.h>

#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace torch::autograd {
struct Node;
} // namespace torch::autograd

namespace torch::distributed::fsdp {

// The wrapped unit's actual compute, e.g. `[&](std::vector<Tensor> in) {
// return std::vector<Tensor>{block->forward(in[0])}; }`. Deliberately the
// same shape as `torch::utils::CheckpointFunction`
// (torch/csrc/api/include/torch/utils/checkpoint.h) -- both exist for the
// same reason: `torch::nn::Module` has no polymorphic base-class `forward()`
// (unlike Python's `nn.Module.__call__`), so anything that needs to invoke
// an arbitrary wrapped module generically has to go through a type-erased
// closure the caller provides, not a virtual dispatch this layer can do
// itself. This also composes directly with `torch::utils::checkpoint()`
// (task item 5): the closure is free to call `checkpoint(...)` internally
// around its real forward instead of calling it directly -- see
// fsdp_unit.cpp's `forward()` for exactly where/why that composes for free.
using UnitForwardFn =
    std::function<std::vector<at::Tensor>(std::vector<at::Tensor>)>;

// One "wrapped unit" -- the FSDP equivalent of one `FlatParamHandle` /
// FSDP2 `FSDPParamGroup`. Typically one transformer block, one Linear layer,
// or any other `torch::nn::Module` whose parameters should be sharded and
// communicated together as a single collective per forward/backward.
class TORCH_API FSDPUnit {
 public:
  // `module`'s CURRENT parameter values (via `module->parameters()`) are
  // read once, at construction, to seed this rank's shard -- so `module`
  // must already be fully initialized (and, in a real multi-rank job,
  // identically initialized/broadcast-synced across ranks by the caller
  // *before* wrapping; this layer does not itself broadcast initial values,
  // mirroring collective_transport's scope of all_gather/reduce_scatter
  // only, no broadcast -- see FSDP-DESIGN.md Sec 3.2's note that broadcast
  // is a distinct, not-yet-built op). `forward_fn` is the closure actually
  // invoked by `forward()` below -- see `UnitForwardFn`'s doc comment for
  // why this layer can't just call `module->forward(...)` generically.
  FSDPUnit(
      std::shared_ptr<torch::nn::Module> module,
      UnitForwardFn forward_fn,
      std::shared_ptr<torch::collectives::CollectiveTransport> transport);

  ~FSDPUnit();
  FSDPUnit(const FSDPUnit&) = delete;
  FSDPUnit& operator=(const FSDPUnit&) = delete;

  // The transparent wrapper per the task brief: call this instead of
  // `module_->forward(...)`. Internally: unshard (all_gather + set_data into
  // every original parameter), run the real module forward, immediately
  // reshard (release the full buffer -- the actual memory-saving point of
  // FSDP). Registers the re-unshard-before-backward hook on every output
  // tensor that requires grad, and (once, lazily, on the very first call)
  // the per-parameter post-accumulate-grad hooks that drive reduce_scatter.
  std::vector<at::Tensor> forward(std::vector<at::Tensor> inputs);

  // The ONE Tensor this unit contributes to the optimizer's param list --
  // this rank's 1/world_size shard of the flattened parameters, with
  // `.grad()` populated by reduce_scatter after backward. Feed this directly
  // into `torch::optim::AdamW`'s param group.
  at::Tensor& shard_param();

  // Diagnostics / the test harness's logging requirement (task item 6c):
  // which transport this unit's all-gather/reduce-scatter calls are
  // actually using.
  torch::collectives::TransportKind transport_kind() const;

  int64_t shard_numel() const {
    return shard_numel_;
  }
  int64_t padded_numel() const {
    return padded_numel_;
  }

  // Diagnostic counter, not used by the orchestration logic itself: counts
  // real calls to unshard() (each one a real all_gather). Used by the
  // regression test (task item 5) to confirm a checkpointed region inside
  // this unit's forward_fn does NOT trigger a second all_gather during its
  // recompute -- exactly one unshard() per forward() and exactly one more
  // per backward pass through this unit, regardless of how many times
  // forward_fn's own internal function is invoked (twice, if it uses
  // torch::utils::checkpoint internally: once for the real forward, once
  // for the recompute).
  int64_t unshard_call_count() const {
    return unshard_call_count_;
  }
  void reset_unshard_call_count() {
    unshard_call_count_ = 0;
  }

 private:
  void unshard(); // all_gather + set_data into every original parameter
  void reshard(); // set_data back to a tiny placeholder + release full buffer
  void register_grad_hooks(); // once, lazily, at first forward()
  void on_param_grad_ready(size_t param_index);
  void pack_and_reduce_scatter(); // called once all params report a grad

  std::shared_ptr<torch::nn::Module> module_; // kept alive; not called directly
  UnitForwardFn forward_fn_;
  std::shared_ptr<torch::collectives::CollectiveTransport> transport_;

  // Original parameter Tensors (handles into `module_`'s own registered
  // Parameters -- same TensorImpl identity for the module's lifetime;
  // `set_data()` mutates them in place, never reassigns which Variable
  // backs them, which is exactly what keeps the once-registered
  // AccumulateGrad post-hooks valid across every step).
  std::vector<at::Tensor> orig_params_;
  std::vector<int64_t> offsets_;
  std::vector<int64_t> numels_;
  std::vector<std::vector<int64_t>> shapes_;

  int64_t total_numel_ = 0; // sum of real numels, unpadded
  int64_t padded_numel_ = 0; // rounded up to a multiple of world_size
  int64_t shard_numel_ = 0; // padded_numel_ / world_size
  int rank_ = 0;
  int world_size_ = 1;

  at::Tensor shard_param_; // persistent leaf; what the optimizer sees
  at::Tensor full_buffer_; // transient scratch; reset (freed) after each use

  bool hooks_registered_ = false;
  int grads_ready_count_ = 0;
  int64_t unshard_call_count_ = 0;
  bool unsharded_for_backward_ = false; // guards double-unshard per backward
  // Keeps every param's grad_accumulator Node alive (same reason
  // Reducer::grad_accumulators_ exists -- it's only weakly referenced from
  // the Variable's AutogradMeta otherwise).
  std::vector<c10::intrusive_ptr<torch::autograd::Node>> grad_accumulators_;
};

} // namespace torch::distributed::fsdp
