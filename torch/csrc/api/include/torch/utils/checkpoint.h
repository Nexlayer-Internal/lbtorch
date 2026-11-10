#pragma once

#include <torch/csrc/Export.h>
#include <torch/types.h>

#include <functional>
#include <vector>

namespace torch::utils {

/// The user-provided function being checkpointed. Takes the (possibly
/// re-materialized) inputs and returns the outputs. Both ends are
/// `std::vector<Tensor>` -- an arbitrary fixed number of leading/trailing
/// non-Tensor arguments is not supported (unlike Python's
/// `torch.utils.checkpoint.checkpoint`, which accepts arbitrary `*args`);
/// capture any non-Tensor configuration in the closure itself.
using CheckpointFunction =
    std::function<std::vector<Tensor>(std::vector<Tensor>)>;

/// C++ port of `torch.utils.checkpoint.checkpoint(..., use_reentrant=True)`
/// -- i.e. the original/"reentrant" gradient-checkpointing implementation
/// (`CheckpointFunction(torch.autograd.Function)` in `torch/utils/checkpoint.py`).
/// The newer non-reentrant / selective-activation-checkpointing path
/// (`use_reentrant=False`, built on saved-tensor hooks and
/// `__torch_dispatch__`-based recompute policies) is NOT implemented here;
/// see PLAN.md row 16 for why the reentrant path was chosen for this port
/// (it's what FSDP-wrapped actor models in verl-style GRPO training actually
/// use via `enable_gradient_checkpointing=True`).
///
/// Mechanics (matches the Python reentrant path exactly):
///   1. `fn(inputs)` is run once, under `torch::NoGradGuard`, to produce the
///      real outputs. No intermediate activations inside `fn` are kept --
///      only `inputs` themselves (and whatever RNG state is needed, see
///      below) are saved.
///   2. When backward reaches this checkpointed region, `fn` is run a
///      *second* time, this time with grad tracking enabled
///      (`c10::AutoGradMode(true)`), on freshly `.detach()`-ed copies of
///      `inputs`, to rebuild a small, throwaway autograd graph just for this
///      region. `torch::autograd::backward()` is then run on that graph (a
///      reentrant call back into the autograd engine, hence "reentrant
///      checkpointing") to get gradients w.r.t. the detached inputs, which
///      become this node's output gradients.
///
/// This trades one extra forward pass through `fn` for not having to keep
/// `fn`'s internal activations alive from forward to backward -- the whole
/// point of checkpointing.
///
/// RNG state: if `preserve_rng_state` is true (the default), the CPU RNG
/// state -- and, for every distinct non-CPU device seen among `inputs`, that
/// device's accelerator RNG state too -- is saved right before step 1's
/// forward call and restored right before step 2's recompute, so that any
/// randomness inside `fn` (dropout being the common case) produces identical
/// results both times. Without this, the recomputed activations silently
/// diverge from what the first forward pass actually produced, and the
/// gradients that come out are numerically plausible but wrong. The RNG
/// state ambient at the *start* of backward (i.e. whatever it was before
/// this function stepped in to replay the forward-time state) is always
/// restored once the recompute is done, mirroring
/// `torch.random.fork_rng()`'s enter/exit pairing -- so a chain of several
/// checkpointed regions hit during one backward pass doesn't leak RNG state
/// between them, and whatever comes after the whole backward pass sees the
/// RNG exactly as if checkpointing had never been used.
///
/// Known gaps vs. the Python implementation (acceptable for this port, but
/// worth knowing about):
///   - autocast context (`torch.amp.autocast`) is not saved/restored around
///     the recompute forward.
///   - Mixed accelerator-device inputs: every distinct non-CPU device seen
///     among `inputs` gets its RNG state saved/restored. Python restricts
///     this to a single inferred device *type* (preferring CUDA) and warns
///     if more than one is present; this port does not replicate that
///     disambiguation/warning, since within a single process you'd need
///     multiple distinct accelerator backends live simultaneously to hit it.
///
/// \param fn The function to checkpoint.
/// \param inputs Inputs to `fn`. Entries that don't require grad are passed
///     through untouched (unlike Python there is no non-Tensor-argument
///     support, so there's no analogous "non-Tensor passthrough").
/// \param preserve_rng_state See above. Defaults to `true`; pass `false`
///     only if `fn` is known to be deterministic (no dropout, no RNG use at
///     all) and you want to skip the (small) overhead of saving/restoring
///     RNG state.
TORCH_API std::vector<Tensor> checkpoint(
    CheckpointFunction fn,
    std::vector<Tensor> inputs,
    bool preserve_rng_state = true);

} // namespace torch::utils
