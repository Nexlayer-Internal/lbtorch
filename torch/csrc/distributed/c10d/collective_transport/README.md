# collective_transport

Transport-agnostic, N-rank collective transport layer (`torch::collectives`),
per `FSDP-DESIGN.md`'s Phase 0 ("transport layer first"). Standalone
infrastructure: nothing here is wired into FSDP, autograd, or anything else
in this change. Build and test it on its own via `CMakeLists.txt` in this
directory (see that file's header comment for the exact invocation).

## What's here

- `topology.h` / `topology.cpp` -- live, bounded P2P probe
  (`probe_peer_pair`). Not a static `nvidia-smi`/`cudaDeviceCanAccessPeer`
  query alone: it actually attempts a P2P-enabled transfer and bounds the
  wait by wall clock, because IP66's own incident history shows the static
  query can report P2P as usable while the real transfer hangs. Also
  provides `bounded_cuda_device_synchronize()`, a general bounded-wait
  primitive added after validation found an even subtler version of the
  same lesson -- see "Validation findings" below.
- `transport.h` -- the transport-agnostic interface
  (`torch::collectives::CollectiveTransport`: `all_gather`,
  `reduce_scatter`, `abort`) and the selection entry point
  (`make_transport`).
- `nccl_fast_transport.h` / `.cpp` -- fast path, a thin wrapper around stock
  `ProcessGroupNCCL`.
- `host_staged_transport.h` / `.cpp` -- fallback path, a from-scratch,
  N-rank, multi-dtype, pinned-host-memory-staged implementation of
  all-gather/reduce-scatter, adopting the *mechanism and safety philosophy*
  (not the code) of `liquid-brain-native`'s `direct_collectives`.
- `transport_factory.cpp` -- `make_transport()`'s implementation: probes
  topology for every rank pair, reaches a group-wide consensus via the
  `Store`, and constructs whichever transport is selected.
- `test/` -- multi-process correctness, bounded-timeout/abort, and the
  IP66 hang-reproduction regression tests. See each file's header comment
  and `test/run_multiproc.sh`.

## What this explicitly does not do

Per `FSDP-DESIGN.md` Sec 4, this is Phase 0 only:

- No FSDP orchestration, hook registration, bucketing, or sharding/view-
  reconstruction logic -- that is a separate, later task (Phase 1-3).
- No multi-node topology model -- `make_transport`'s topology probe assumes
  every rank's device is visible on a single node (matches IP66 and
  matches FSDP's per-node sharding groups; see `transport.h`'s doc
  comment).
- No uneven/non-equal shard sizes (`host_staged_transport.cpp` asserts
  equal-sized shards; FSDP's actual last-shard-uneven-size handling is left
  for the orchestration-layer work).
- `all_reduce` / `broadcast` are not implemented in this phase: only
  `all_gather` and `reduce_scatter`, the two operations FSDP actually needs
  per `FSDP-DESIGN.md` Sec 2.
- The host-staged fallback's `all_gather`/`reduce_scatter` calls are
  synchronous (block until their own call completes). The staging memory
  is double-buffered so a future async API could pipeline call N+1's
  staging against call N's drain, but exposing that overlap at the
  collective-API level is explicitly FSDP orchestration's job, not this
  layer's, per `FSDP-DESIGN.md` Sec 3.1.

## Validation findings on IP66 (2x RTX PRO 6000 Blackwell, confirmed SYS topology)

**`ProcessGroupNCCL::Work::wait()` is not sufficient evidence that a
collective finished, on this hardware.** Direct instrumentation of a real
`all_gather_single` (bypassing this layer, same as
`repro_vanilla_nccl_hang.cpp`) showed: the collective enqueues, and
`work->wait()` (no explicit timeout) returns *successfully* in ~370-800ms --
but a plain `cudaDeviceSynchronize()` called immediately afterward then
hangs. This is corroborated independently by PyTorch's own NCCL watchdog
when `wait()` is given an explicit timeout instead: it reports `"Watchdog
caught collective operation timeout ... ran for 30000 milliseconds before
timing out"` followed by `ncclSystemError`, i.e. PyTorch's own stack agrees
the collective never genuinely finished -- it just doesn't surface that
through `wait()`'s default (no-explicit-timeout) return value.

This meant the topology probe's first version had a real blind spot: it
only exercised a raw `cudaMemcpyPeerAsync` (a different code path from
NCCL's own P2P/CUMEM channel setup), which genuinely succeeds fast on this
hardware, so the probe wrongly cleared the fast path even though a real
collective on the same hardware does not reliably finish. Fixed in
`transport_factory.cpp`'s `verify_live_nccl_collective()`: a mandatory
second stage that builds a real (throwaway) group-wide `ProcessGroupNCCL`,
runs one tiny `all_gather_single`, and requires a bounded
`cudaDeviceSynchronize()` to *also* succeed -- not trusting `Work::wait()`
alone -- before the fast path is selected.

That fix surfaced a second, related hazard: the first version of "abandon a
timed-out probe's throwaway `ProcessGroupNCCL` instead of risking a hang on
teardown" stored it in a `static std::vector`, whose destructor still runs
at normal process exit -- so it still called `~ProcessGroupNCCL()` on a
wedged communicator and crashed (SIGABRT/SIGSEGV) during static teardown.
This is the same hazard family as `FSDP-DESIGN.md`'s incident #2
(`ncclCommDestroy()` hanging on an already-wedged comm), manifesting as a
crash instead of a hang. Fixed with a true heap leak (`new`'d, never
deleted) so the destructor genuinely never runs, not even at exit.

(One detail worth not being alarmed by when reading the raw logs: the
watchdog's `"To avoid data inconsistency, we are taking the entire process
down"` line, read in `ProcessGroupNCCL.cpp`'s `handleException()`, is
followed by `std::rethrow_exception`, called from `Work::wait()` itself on
the *calling* thread, synchronously, as part of `wait()`'s own return path
-- not an independent, uncatchable abort from a background thread. That is
exactly why `verify_live_nccl_collective()`'s own `try`/`catch` around
`work->wait(timeout)` reliably catches it. `wait()` does, however, also
synchronously call `abort()`/`abortComms()` on the wedged communicator
*before* throwing -- a documented, deliberate PyTorch behavior (its own
comment: "If throwing timed out exception without aborting nccl
communicators here, it was observed that CUDA GPU will have 100%
utilization and can not run new events successfully"). That first abort
call is what actually matters here, and it completed without incident in
every run. The crash this project hit and fixed (see above) came from a
*second* teardown attempt on the same already-aborted communicator, later,
via `~ProcessGroupNCCL()` during normal static destruction at process
exit -- which is also why a true heap leak, not just avoiding an explicit
`abort()` call, was the fix: it ensures that second attempt never happens
at all.)

With both fixes, `test_collective_correctness` and `repro_transport_no_hang`
run repeatedly and cleanly in real (non-forced) auto-probe mode: the group
consistently selects `host_staged_fallback` on IP66, with no hang and no
crash, while `all_gather`/`reduce_scatter` on the fallback complete
correctly in milliseconds at FSDP-realistic (hundreds-of-MB) payload sizes.
Forcing `FORCE_TRANSPORT=fast` to bypass the probe still hangs, as
expected -- that demonstrates the probe's rejection is correct, not a
regression.
