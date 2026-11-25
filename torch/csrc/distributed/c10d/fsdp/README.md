# fsdp

C++ FSDP orchestration layer (`torch::distributed::fsdp::FSDPUnit`), per
`FSDP-DESIGN.md` Sec 4 "Phase 1-3" and `PLAN.md` row 11/12 (the ~13,000-line
Python FSDP subsystem this replaces for the validated GRPO use case). Builds
on top of `../collective_transport` (Phase 0, already merged/validated) and
is itself standalone/not wired into the monolithic libtorch build -- same
`CMakeLists.txt` pattern as `collective_transport`'s own.

## What's here

- `fsdp_unit.h` / `fsdp_unit.cpp` -- the orchestration layer itself: one
  `FSDPUnit` per wrapped module (one transformer block / one Linear layer /
  etc). See `fsdp_unit.h`'s header comment for the full design (FlatParameter
  construction + sharding at wrap time, all-gather before forward/backward,
  reduce-scatter after backward, autograd-hook-driven timing, why
  `Tensor::set_data()` is the parameter-swap primitive, and how gradient
  checkpointing composes with zero extra all-gathers).
- `test/test_fsdp_training.cpp` -- the real, 2-rank, end-to-end regression
  test: a 4-layer Linear stack (one layer internally gradient-checkpointed),
  wrapped unit-per-layer, trained for 30 real steps with `torch::optim::
  AdamW` against a synthetic regression target. Confirms: no hang (the
  original IP66 incident this whole project traces back to), loss decreases,
  which transport was selected, and that the checkpointed unit unshards
  exactly twice per step (not three times).

## What this does not do (v1 scope, see fsdp_unit.h for the full list)

- One flat parameter per wrapped unit (classic FSDP1 shape), not FSDP2's
  per-original-parameter DTensor sharding.
- No prefetch/overlap of the next unit's all-gather with the current unit's
  compute -- units are unsharded/resharded strictly one at a time.
- Equal-shard-size only (inherits this from `collective_transport`'s current
  scope); single-node only (ditto).
- No broadcast-at-init: every rank must construct the wrapped module with
  identical initial values itself (e.g. identical seeding, as the test does)
  -- `CollectiveTransport` has no `broadcast` op yet (transport.h's own
  scope note).

## Build

```
cmake -G Ninja -DTORCH_SRC_DIR=/workspace -DTORCH_BUILD_DIR=/workspace/build -B build .
ninja -C build
```

## Run the regression test

```
cd build
MASTER_PORT=29550  # pick a free port
FORCE_TRANSPORT=  # unset -> real auto-probe, matches production selection
../../collective_transport/test/run_multiproc.sh test_fsdp_training 2 29550 180
```
