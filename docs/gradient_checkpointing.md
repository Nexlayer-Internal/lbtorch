# Gradient checkpointing: a C++ port of the reentrant path

Implements step 2 of `PLAN.md`'s prioritized plan (row 16, "Gradient checkpointing"):
a C++ equivalent of `torch.utils.checkpoint.checkpoint(..., use_reentrant=True)`.
Per the plan, this is **new work, not extend-existing** -- the survey found no
existing C++ equivalent anywhere in libtorch, so unlike step 1's `AdamW` (which
extended an existing `torch::optim::AdamW`), this is a from-scratch port.

## Scope

Only the **reentrant** path (`CheckpointFunction(torch.autograd.Function)` in
`torch/utils/checkpoint.py`, the original/default implementation). The newer
**non-reentrant** path (`use_reentrant=False`, the current Python default,
built on saved-tensor hooks and `__torch_dispatch__`-based selective-recompute
policies) is explicitly out of scope for this PR. Reentrant is what
FSDP-wrapped actor models in verl-style GRPO training actually use (confirmed
against the validated GRPO smoke test's config:
`actor_rollout_ref.model.enable_gradient_checkpointing=True`, which routes
through FSDP's own checkpoint wrapper calling `torch.utils.checkpoint.checkpoint`
with the reentrant default in the verl version referenced by this fork's prior
survey) -- see `PLAN.md` row 16 and the executive summary for why this ordering
(checkpointing before FSDP) was chosen: FSDP's own C++ port (PLAN.md step 3)
can't be properly exercised end-to-end without checkpointing already working.

## What's new

- `torch/csrc/api/include/torch/utils/checkpoint.h` +
  `torch/csrc/api/src/utils/checkpoint.cpp` --
  `torch::utils::checkpoint(CheckpointFunction fn, std::vector<Tensor> inputs,
  bool preserve_rng_state = true) -> std::vector<Tensor>`, where
  `CheckpointFunction = std::function<std::vector<Tensor>(std::vector<Tensor>)>`.
  Registered in `caffe2/CMakeLists.txt` and in `all.h` (so `#include
  <torch/torch.h>` picks it up, same as `torch::optim`/`torch::nn`).
- `test/cpp/api/checkpoint.cpp` -- GTest cases, registered in
  `test/cpp/api/CMakeLists.txt`'s `TORCH_API_TEST_SOURCES` the same way
  `optim.cpp` already is.

Unlike Python's `checkpoint()`, which takes arbitrary `*args` (tensors and
non-tensors freely mixed) because it needs to be a fully generic drop-in
replacement for calling any Python callable, the C++ API only accepts
`std::vector<Tensor>` in and out. Any non-Tensor configuration a checkpointed
region needs should be captured in the closure itself (lambda capture); this
is both simpler to implement correctly and more idiomatic for the rest of
`torch::optim`/`torch::nn`'s C++ API style, which generally favors explicit,
narrowly-typed signatures over Python's duck-typed generality.

## Implementation vehicle: `torch::autograd::Function<T>`

Confirmed via `torch/csrc/autograd/custom_function.h` and
`test/cpp/api/autograd.cpp` (dozens of existing `Function<T>` usages, including
`CustomFunctionWithTensorList` at line 375, which is the precedent this port's
calling convention for a variable-length tensor input list is copied from
directly): `torch::autograd::Function<T>` is a real, first-class, pure-C++
mechanism for exactly this ("run some forward computation, and supply a custom
backward"), with zero Python-specific scaffolding needed. This confirms
PLAN.md's assessment in row 3 and the GRPO section of §3.

One real wrinkle PLAN.md's summary doesn't mention, because it only shows up
once you try to actually implement something stateful with it:
`Function<T>::forward`/`::backward` are **static** methods -- `T` itself never
has a live instance holding your data, so nothing about the mechanism gives
you a natural place to stash a runtime `std::function` closure (the
checkpointed function itself) so it survives from the forward call to the
later backward call on the same node. The only persistent-across-calls slot
`AutogradContext` offers is `saved_data`, a `map<string, at::IValue>`, and a
`std::function` doesn't fit naturally into an `IValue` the way an `int64_t` or
`Tensor` does.

The fix used here: wrap the closure (plus the RNG bookkeeping that also needs
to survive forward->backward -- see below) in one small
`CheckpointState : public torch::CustomClassHolder` struct, and stash it as a
single `at::IValue` "Capsule" (`IValue::make_capsule`/`toCapsule`,
`aten/src/ATen/core/ivalue.h`) in `ctx->saved_data`. `ivalue.h` carries a "no
new callsites" comment on that API -- it's there to steer people towards
`torch::class_<T>`-registered custom classes for anything that needs to be
introspectable by TorchScript/torch.export/serialization/compiled autograd.
Nothing here is ever any of those: the capsule is created and consumed within
a single eager backward pass and never escapes it, which is exactly the
"internal implementation detail" use case `ivalue.h`'s own doc comment for the
Capsule tag describes one line above that caution. Going through
`torch::class_<T>` registration instead would add global-registration-order
risk and visibility in `torch.classes.*` reflection for zero benefit here, so
the direct Capsule route was used deliberately, not out of not knowing the
alternative existed.

A second, smaller wrinkle: passing a variable-length list of input tensors as
a single argument to `Function<T>::apply()` only gets tracked correctly in the
autograd graph if that argument's **static type** is exactly `at::TensorList`
(`custom_function.h`'s `ExtractVariables` has a dedicated overload for it) --
passing a `std::vector<Tensor>`/`variable_list` directly, instead, silently
falls through to the generic "not a variable" case and the tensors inside it
would not be registered as graph inputs at all. `checkpoint()` explicitly
constructs a named `at::TensorList` local from the input vector before calling
`apply()` to get this right, copying the exact pattern
`CustomFunctionWithTensorList` in `test/cpp/api/autograd.cpp` uses.

## RNG state handling

Mirrors `torch/utils/checkpoint.py`'s `get_device_states`/`set_device_states`
and the bare `torch.get_rng_state()`/`set_rng_state()` calls exactly:

1. In `forward()`, if `preserve_rng_state` (default `true`): save the CPU RNG
   state, plus the RNG state of every distinct non-CPU/meta device seen among
   `inputs`, **before** calling `fn` the first time.
2. In `backward()`: save whatever RNG state is *ambient* right now (the state
   at backward-entry, not the forward-time one) so it can be restored once the
   recompute below is done -- this is the save/restore pair
   `torch.random.fork_rng()`'s `with` block does on enter/exit. Then overwrite
   the CPU + saved devices' RNG state with the state captured in step 1, run
   `fn` a second time (now under `c10::AutoGradMode(true)`), and finally
   restore the ambient state from the start of this step.

The actual C++ calls this bottoms out on
(`at::globalContext().defaultGenerator(device)` -> `at::Generator::get_state()`
/ `set_state()`, each under `Generator::mutex()`) are the exact same
device-agnostic `at::Generator`/`at::Context` abstraction Python's
`torch.get_rng_state()`/`device_module.get_rng_state()` themselves bottom out
on (confirmed by reading `torch/random.py` and `aten/src/ATen/Context.h`'s
`manual_seed()`, which uses the identical "copy Generator to a local, lock its
mutex, call get/set_state" pattern this port copies) -- not a reimplementation
of RNG state capture, the same mechanism Python uses, called directly.

Because this is device-agnostic (no `at::cuda::*` symbols are referenced
anywhere in `checkpoint.cpp`), the code compiles and is exercised by the CPU
tests regardless of whether CUDA is enabled; the device-state-saving branch
would take over identically for CUDA tensors without needing a separate
`#ifdef USE_CUDA` path, consistent with how the Python version is written.

**Known simplification vs. Python**: Python's `_infer_device_type` restricts
RNG bookkeeping to a single inferred device *type* among the inputs
(preferring CUDA if present, warning if more than one non-CPU type is seen,
since e.g. mixed CUDA+XPU tensors in one checkpointed region is an unusual and
mostly academic case). This port instead saves/restores every distinct
non-CPU device seen among `inputs`, full stop, without that
single-type-preference/warning logic. This is a strict superset in terms of
what gets saved (so it cannot be a source of missed RNG state), just not a
byte-identical reimplementation of Python's disambiguation heuristic.

**Known gap vs. Python**: autocast context (`torch.amp.autocast`) is not
saved/restored around the recompute forward. Not needed for the RNG
correctness problem this PR is scoped to, but worth listing as missing rather
than silently absent.

## What was verified, and how

Full build/test status: **cmake (4.4.4) and ninja (1.13.2) were available in
this environment** (installed via `pip install cmake ninja`, after the
environment step 1 (`AdamW`) was developed in did not have them) -- a real
difference from step 1's environment. **No CUDA GPU was available** (macOS,
no `nvcc`/`nvidia-smi`), so only a CPU build was attempted; any CUDA-specific
code path (the `#ifdef USE_CUDA` memory test below) is unexecuted.

See the PR description for the exact build/test commands run and their
results -- this doc covers the design, not the verification log, so it
doesn't go stale if the build is re-run later with different results.

## Tests (`test/cpp/api/checkpoint.cpp`)

- `MatchesNonCheckpointedGradients` -- a 6-layer `Linear`+`ReLU` stack, split
  into two checkpointed segments of 3 layers each, must produce exactly the
  same input gradient *and* every layer's weight/bias gradient as running the
  same stack with no checkpointing at all.
- `MultipleInputsAndOutputs` -- basic plumbing: 2 inputs, 2 outputs, one input
  genuinely unused (gradient must come back undefined, not zero or an error).
- `PreserveRngStateFalseIsCorrectForDeterministicFn` -- `preserve_rng_state=false`
  only opts out of the save/restore bookkeeping; must still be correct for a
  function that doesn't use randomness.
- `DropoutRecomputeReproducesForwardMask` -- the direct RNG test: captures the
  output of the checkpointed function on *every* call (the original no_grad
  forward, and the recompute during backward) and asserts they're bit-exact.
  A broken RNG restore would, with overwhelming probability, draw a different
  dropout mask the second time and fail this specific assertion -- this is the
  test designed to catch exactly the "silently wrong gradients" failure mode
  called out in the PR description, rather than a crash.
- `DropoutGradientMatchesNonCheckpointedBaseline` -- end-to-end version of the
  same property: with the global RNG seeded identically, a checkpointed
  forward+backward through a dropout layer must produce exactly the same
  input/weight gradients as the same network run with no checkpointing.
- `ReducesPeakMemoryOnCuda` (guarded by `#ifdef USE_CUDA`, skips at runtime if
  no CUDA device is present) -- a 64-layer, dim-2048 stack checkpointed as one
  segment must show a lower `c10::cuda::CUDACachingAllocator` peak
  `allocated_bytes` than the same stack run without checkpointing. Unexecuted
  in this environment (no GPU); the allocator API calls were checked against
  `c10/cuda/CUDACachingAllocator.h` and `c10/core/Allocator.h`, not guessed.
