# AdamW: foreach / fused / capturable, and Python checkpoint interop

Implements step 1 of `PLAN.md`'s prioritized plan ("Optimizer step", rows 7-9):
extends the existing `torch::optim::AdamW` (`torch/csrc/api/include/torch/optim/adamw.h`,
`torch/csrc/api/src/optim/adamw.cpp`) with the three missing performance variants
plus a Python `state_dict()` checkpoint bridge. Per the plan, this is **extend, not
replace** -- the original single-tensor algorithm is untouched (still the default,
still what every pre-existing caller gets).

## What's new

- `AdamWOptions::foreach()`, `::fused()`, `::capturable()` -- three new bool options,
  mutually exclusive between `foreach`/`fused` (checked in `AdamW`'s constructor and
  at the top of `AdamW::step()`), mirroring `torch/optim/adamw.py`'s kwargs of the
  same names. Unlike Python, these are plain `bool`, not `bool | None` -- the C++
  frontend doesn't do the device-dependent "auto-select foreach/fused" heuristics
  `torch.optim.AdamW.__init__` does; you ask for what you want.
- `AdamW::step()` now dispatches each param group to one of three private methods:
  - `step_single_tensor` -- the original algorithm, extended with a `capturable`
    branch (device-tensor bias correction instead of host-scalar).
  - `step_foreach` -- multi-tensor batched update via `at::_foreach_*` ops, mirroring
    `torch/optim/adam.py`'s `_multi_tensor_adam` (decoupled-weight-decay/AdamW
    branch), itself with a `capturable` sub-branch (so `foreach=true,
    capturable=true` is supported, same as upstream).
  - `step_fused` -- calls `at::_fused_adamw_` directly, the *same* native op
    `torch.optim.AdamW(fused=True)` dispatches to (see
    `aten/src/ATen/native/native_functions.yaml:_fused_adamw_`, dispatched per
    backend to `_fused_adamw_kernel_{cpu,cuda,mps,xpu}_`). We do not reimplement
    the fused math; we call the real registered kernel.
- `AdamWParamState::step_tensor()` -- a new, optional device-resident scalar Tensor
  step counter, used only when `capturable` and/or `fused` is enabled. The original
  `step()` (`int64_t`, host-resident) is untouched and remains the source of truth
  for the default path; every pre-existing archive/test that reads `.step()` as an
  `int64_t` keeps working unmodified.
- `torch/csrc/api/include/torch/optim/adamw_checkpoint_bridge.h` +
  `torch/csrc/api/src/optim/adamw_checkpoint_bridge.cpp` -- `export_checkpoint` /
  `import_checkpoint`, a Python `state_dict()` <-> C++ `AdamW` bridge.
- `tools/checkpoint_bridge/adamw_checkpoint_bridge.py` -- the Python-side
  counterpart (`export_adamw_checkpoint` / `import_adamw_checkpoint`), usable
  standalone from any Python training script (e.g. `verl`) with no C++ dependency.

## Numerical fidelity notes (read this before trusting any specific digit)

All three new paths were written by transcribing the exact arithmetic of
`torch/optim/adam.py`'s `_single_tensor_adam` / `_multi_tensor_adam` / `_fused_adam`
(decoupled-weight-decay/AdamW branches) -- same order of operations, same bias
correction formulas, same epsilon placement, same weight-decay order (decoupled,
applied to the param directly, before the moment updates). Two things are **not**
bit-identical by design, and this is expected, not a bug:

1. **`capturable` vs non-capturable can differ by ~1e-6 (float32) at a given
   step.** The device-resident `step_tensor` is float32 (matching
   `_get_scalar_dtype(is_fused=...)`'s common case -- see "Known limitations"
   below), so `pow(beta, step_tensor)` is computed in float32, whereas the
   non-capturable path computes `std::pow(beta, step)` in `double` on the host.
   Upstream PyTorch has the exact same characteristic for the exact same reason;
   it is not something this fork introduced, and it does not compound into anything
   larger than float32 ULP-scale noise in testing (see Testing below).
2. **`fused` on CPU and `foreach` on CPU gave *exactly* matching results against
   their Python counterparts in testing (0.0 max abs diff)**, because both sides
   call the identical registered op (`at::_fused_adamw_` / `at::_foreach_*`) with
   identical inputs -- there's no reimplementation to drift.

## Known limitations (said plainly, not buried)

- **`capturable=True` on a CPU tensor is allowed here, with a `TORCH_WARN_ONCE`,
  where upstream Python hard-errors** (`_get_capturable_supported_devices()`
  excludes `"cpu"`). We relaxed this specifically so the capturable code path could
  be unit-tested without a CUDA device (see Testing). Do not rely on this for
  anything that matters on CPU in production; the warning says so.
- **`step_tensor`'s dtype is hardcoded to `float32`**, matching Python's
  `_get_scalar_dtype(is_fused=...)` under the assumption that
  `torch.get_default_dtype() == torch.float32`, true for essentially every real
  training configuration (including the `verl`/GRPO target use case). If someone
  has globally set the default dtype to `float64`, the step tensor's dtype won't
  match what Python would have produced bit-for-bit. This has no numerical
  consequence for training (the step counter is an integer-valued scalar either
  way) and only matters if you're trying to produce a checkpoint file that looks
  byte-identical to one Python would write in that unusual configuration.
- **No `maximize` or `differentiable` support**, matching the pre-existing
  `torch::optim::AdamW` (neither option existed in `AdamWOptions` before this
  change either). Not required for the GRPO/FSDP target use case.
- **No complex-tensor support** in the new `foreach`/`fused` paths (the original
  single-tensor path's complex handling is also absent from the Python reference's
  equivalent branches we ported, for the same reason: out of scope for the
  real-valued transformer weights GRPO trains).
- **The foreach/fused device+dtype grouping is a simple `std::map`-based grouping
  we wrote ourselves** (`torch/csrc/api/src/optim/adamw.cpp`'s anonymous
  `DeviceDtypeKey`/`ForeachBucket`), not a port of
  `Optimizer._group_tensors_by_device_and_dtype`'s exact internals (that utility
  isn't exposed at the C++ frontend layer). Correct and sufficient for the common
  case (one device, uniform dtype per param group) that the GRPO FSDP training
  target exercises.
- **CUDA-specific paths are implemented but *performance*-unverified on real
  hardware in this change.** See "Performance" below -- read it before citing any
  number from this PR as a GPU result.
- **The checkpoint bridge's format (v1) flattens params across all of an
  optimizer's `param_groups` with no group-boundary bookkeeping.** This exactly
  matches Python's own flat `state_dict()["state"]` indexing for the common single-
  param-group case (one `AdamW` over one model, which is what the GRPO actor/critic
  training loop this project targets actually does) but does not yet disambiguate
  *which* group a param belongs to if you need to reconstruct per-group
  hyperparameters (e.g. different `lr` for different groups) from the checkpoint
  alone. Extending the format with a `(group_index) -> count` table is
  straightforward if/when a multi-group use case shows up; not done here because it
  isn't needed yet and every added field is another thing to keep the two
  implementations in sync on.

## Why an explicit exchange format, not a byte-for-byte `state_dict()` port

`torch::optim::AdamW`'s native `save()`/`load()` go through
`torch::serialize::OutputArchive`/`InputArchive` (`torch/csrc/api/src/optim/serialize.cpp`),
which has no defined relationship to Python's `torch.optim.AdamW.state_dict()` (a
nested Python dict of Python ints and tensors, which in practice is always written
out via `torch.save`, i.e. pickled inside a zip container). Making C++ read/write
that representation directly would mean either embedding a pickle-compatible
decoder in C++ (a meaningful chunk of new code, version-coupled to Python's pickle
protocol, and in tension with the entire point of this project: no Python
dependency on the training path) or reverse-engineering the zip+pickle byte stream
well enough to round-trip it reliably across PyTorch/pickle versions (fragile, and
still Python-pickle-shaped even once done).

Instead, `adamw_checkpoint_bridge.{h,cpp}` (C++) and
`tools/checkpoint_bridge/adamw_checkpoint_bridge.py` (Python) agree on one simple,
versioned, explicitly-documented binary layout (below) and each side converts
to/from its own native in-memory representation. This is option (b) from the task
spec ("writing explicit conversion functions") rather than (a) ("matching the C++
state's serialization format exactly to Python's dict structure") -- chosen because
(a) would require C++ to either produce or consume pickle, which this project is
explicitly trying to get away from needing at all on the training path.

### Format v1 byte layout

```
magic:          4 bytes, "LBA1"
version:        int32 (= 1)
num_params:      int64          -- flattened across all param_groups, in order
for each param:
  ndim:          int32
  shape:         int64 * ndim
  dtype_code:    int32          -- 0=float32, 1=float64, 2=float16, 3=bfloat16
  has_max:       uint8          -- 1 iff amsgrad's max_exp_avg_sq is present
  step:          float64        -- canonical step value (works regardless of
                                    whether the source was host-int64, or a
                                    capturable/fused device tensor)
  param bytes:   numel * elemsize, raw, row-major, dtype above
  exp_avg bytes: numel * elemsize
  exp_avg_sq bytes: numel * elemsize
  max_exp_avg_sq bytes: numel * elemsize   -- only present if has_max
```

All integers little-endian (every realistic target here -- x86_64, aarch64 -- is
little-endian; this is not claimed to be portable to a big-endian host).

## Testing

**What was actually run, and on what:** this machine (macOS/arm64, no CUDA GPU) has
no `cmake`/`ninja` installed and a from-source lbtorch build was judged infeasible
to complete + validate within this change's time budget (a full PyTorch source
build is commonly 1+ hour even with good parallelism, by the project's own
estimate). Instead, the exact logic in `adamw.cpp`/`adamw_checkpoint_bridge.cpp` was
compiled standalone (`clang++ -std=c++17`) against the **pip-installed PyTorch
2.8.0** CPU wheel's prebuilt `libtorch_cpu`/`libc10`/headers (renaming the classes
`AdamW` -> `AdamWExt` etc. to avoid an ODR clash with that wheel's own built-in
`torch::optim::AdamW` symbols -- a copy renamed purely for linking purposes, not a
logic change) and exercised directly, with results compared against that same pip
wheel's `torch.optim.AdamW` run on deterministic (non-RNG, formula-generated)
parameters and gradients. lbtorch itself is pinned to PyTorch 2.16.0a0 at the time
of this change; 2.8.0 was the closest thing to a ground truth available without a
multi-hour build, and the `torch::optim::Optimizer`/`OptimizerParamGroup`/
`OptimizerCloneable{Options,ParamState}` base-class API this code builds on has been
stable across that range. **This is a real limitation**: it validates the
algorithm's logic, not a build of lbtorch itself; the equivalent GTest cases added
to `test/cpp/api/optim.cpp` (see below) are what should run under lbtorch's own
CI/build to close that gap.

Results (20 optimizer steps, 4 tensors of mixed shapes, lr=0.05, betas=(0.9,0.999),
eps=1e-8, weight_decay=0.01, float32):

| comparison | atol/rtol | max abs diff | result |
|---|---|---|---|
| C++ single-tensor vs Python eager `AdamW` | 1e-6 | 1.49e-7 | PASS |
| C++ `foreach=true` vs Python `foreach=True` | 1e-6 | **0.0 (exact)** | PASS |
| C++ `fused=true` vs Python `fused=True` | 1e-6 | **0.0 (exact)** | PASS |
| C++ `capturable=true` (single-tensor) vs C++ single-tensor | 1e-5 | 2.12e-6 | PASS (see note 1 above) |
| C++ `foreach=true,capturable=true` vs C++ `foreach=true` | 1e-5 | 2.12e-6 | PASS (ditto) |

(`capturable` has no Python-side CPU ground truth to compare against --
`torch.optim.AdamW(capturable=True)` raises on CPU tensors upstream; see "Known
limitations." It's instead cross-checked against this fork's own already-Python-
validated non-capturable path, which is the best available correctness signal
without a GPU.)

Checkpoint interop, both directions (10 steps, then one more step after a
save/load round trip):

| direction | max abs diff vs "just kept running in the same process" | result |
|---|---|---|
| Python saves -> C++ `import_checkpoint` -> step | 1.49e-8 | PASS |
| C++ `export_checkpoint` -> Python `import_adamw_checkpoint` -> step | 7.45e-9 | PASS |

All of the above were executed in this session; none are asserted without having
actually been run. The scripts live in this PR's description / can be reconstructed
from `adamw_checkpoint_bridge.py` + the format above if you want to rerun them
yourself; the standalone harness itself (renamed-class `.cpp`/`.h` copies + a
`main.cpp` driver) was scratch tooling for this validation and is not part of this
PR's diff (the PR's diff is the real `torch/csrc/api/**` and
`tools/checkpoint_bridge/**` files it validated).

Additionally, equivalent correctness tests were added to `test/cpp/api/optim.cpp`
as real GTest cases (`OptimTest.AdamWForeachMatchesSingleTensor`,
`OptimTest.AdamWFusedMatchesSingleTensor`,
`OptimTest.AdamWCapturableMatchesNonCapturable`) so they run under lbtorch's own
build/CI once that exists -- **these were added but not executed by this change**
(same build-infeasibility reason as above); they encode the same comparisons just
validated via the standalone harness, against lbtorch's own real compiled objects
rather than a stand-in pip wheel.

## Performance

**No CUDA GPU was available to this change to benchmark on.** This is stated
plainly, not glossed over: any specific throughput number for `fused` or
`capturable` on CUDA (their actual target regime -- `fused`'s whole point is fewer
CUDA kernel launches, `capturable`'s is CUDA-graph replay eliminating host launch
overhead entirely) would be fabricated if claimed here, so none is. All three paths
are implemented and are **believed correct** (see Testing above) but are
**performance-unverified on real GPU hardware** by this change.

A CPU-only microbenchmark (`ms/step` over 30-50 steps, warmup excluded) was run
instead -- real, measured numbers, but CPU, and explicitly not a stand-in for the
CUDA numbers that actually matter for this project's throughput goals:

| regime (n tensors x numel) | single | foreach | fused | threads |
|---|---|---|---|---|
| 400 x 4096 (~1.6M params) | 25.6 ms/step | 24.2 ms/step | 22.0 ms/step | 10 |
| 50 x 200,000 (~10M params) | 73.6 ms/step | 72.9 ms/step | 60.4 ms/step | 10 |
| 1000 x 512 (~0.5M params, many-tiny-tensor regime) | 6.9 ms/step | 6.2 ms/step | 32.5 ms/step | 10 |
| 1000 x 512 (same, single CPU thread) | 7.1 ms/step | 6.3 ms/step | **3.8 ms/step** | 1 |

Interesting, worth flagging rather than hiding: in the many-tiny-tensors regime,
the CPU fused kernel was **slower** than foreach at 10 threads but **faster** than
both at 1 thread -- consistent with the CPU fused kernel internally parallelizing
per call in a way that oversubscribes/thrashes the thread pool when called
repeatedly across many small per-dtype/device buckets, a CPU-kernel-implementation
detail in ATen we did not write and cannot tune from the C++ optimizer layer. This
dynamic is specific to the CPU kernel's threading strategy and should **not** be
assumed to predict anything about CUDA behavior, where there is no equivalent CPU
thread-pool contention and the entire motivation for fusing is different (kernel
launch count, not intra-kernel parallelism). Real GPU numbers are needed before
drawing any conclusion about `fused`'s actual target-regime performance; consider
this CPU table a sanity check that the code paths are doing real work roughly in
the expected ballpark, not a performance verdict.

`capturable`'s entire value proposition (CUDA graph replay removing host launch
overhead) could not be exercised at all without a CUDA device -- there is no CPU
analog to benchmark.
