# FSDP-DESIGN: a C++ FSDP replacement informed by `liquid-brain-native`'s `direct_collectives`

**Status:** design research only, no implementation in this document or branch (per the task brief).
**Supersedes:** `PLAN.md` §4 step 3's original recommendation to model the FSDP rewrite purely on
DDP's `Reducer.cpp` calling stock `ProcessGroupNCCL`. That recommendation undercounted a real,
production-confirmed hazard class on non-NVLink hardware; this document folds in Nexlayer's own
proven fix for that hazard class from `liquid-brain-native` (lbx), while being honest about how much
of lbx's actual code is, and is not, reusable as-is.

---

## 0. Why this revision exists

IP66 (2x NVIDIA RTX PRO 6000 Blackwell, confirmed via `nvidia-smi topo -m` to be `SYS` topology — no
NVLink, traversal through the SMP/QPI-UPI interconnect, cross-NUMA/cross-CPU-socket) hit a real
deadlock running standard PyTorch FSDP: both worker processes froze solid, confirmed via `py-spy dump`
with identical stack traces 5 seconds apart, inside `FlatParamHandle.flat_param_to` — a GPU-to-GPU
device-transfer deadlock on exactly the class of hardware NCCL's own P2P logic does not handle
correctly. The practical fix at the time, `NCCL_P2P_DISABLE=1`, is an environment workaround, not a
real fix — it forces NCCL onto a slower socket-based fallback transport.

Separately, and independently, `liquid-brain-native`'s own engineers hit the **same hazard class** on
the **same hardware** for their own inference-serving tensor/pipeline-parallel code, and did not stop
at the environment-variable workaround. They built `direct_collectives`: a from-scratch, direct-CUDA
communication mechanism that never uses `cudaMemcpyPeerAsync` (GPU-to-GPU P2P DMA) at all, staging all
cross-GPU data movement through `cudaHostAlloc`'d pinned host memory instead.

This document studies that mechanism in depth, honestly assesses how much of it transfers to FSDP's
actual communication pattern, and proposes a concrete, topology-aware architecture for the C++ FSDP
replacement that borrows what genuinely transfers and builds, from scratch, what does not.

**One correction to the task's own framing, worth stating plainly up front:** lbx's engineers did not
hit the *identical* bug to IP66's `flat_param_to` deadlock — they hit a closely related **second**
incident in the same hazard family. The full, honest history, reconstructed from
`docs/native/design/tensor-parallel-v1.md` and `docs/native/design/direct-collectives-v1.md`:

1. **First (matches IP66's bug mechanism):** `ncclAllReduce` with P2P enabled hung indefinitely on this
   exact 2x RTX PRO 6000 topology, even though `nvidia-smi topo -p2p r` reported P2P as `OK` — NCCL's
   own runtime detects `isAllDirectP2p 1 directMode 1` and tries to use direct device-to-device
   pointers over a link that does not actually work for it at runtime on this hardware. Fixed, at the
   time, with `NCCL_P2P_DISABLE=1` — exactly IP66's workaround, for exactly the same reason.
2. **Second (the one that directly motivated building `direct_collectives`):** even with
   `NCCL_P2P_DISABLE=1` forced, every TP=2/PP=2 process's **main thread never exited** after finishing
   real, correct work — a genuine orphaned busy-spin (confirmed via `/proc/<pid>/status` = state `R`,
   100% CPU, indefinitely) inside NCCL's own non-P2P socket-fallback teardown path. An explicit
   `ncclCommDestroy()` call, added as a minimal fix, did **not** help — the spin simply moved inside
   that call, confirming the defect lives in NCCL's own communicator-teardown machinery over this
   degraded transport, not in a missing destroy call.

So `direct_collectives` is lbx's fix for incident #2 (a teardown hang under the P2P-disabled fallback),
built using the same underlying philosophy (bypass NCCL's P2P path entirely, stage through host memory)
that `NCCL_P2P_DISABLE=1` was already a partial, unsatisfying answer to for incident #1. The mechanism
is real and directly applicable to IP66's hazard class; the specific bug is not byte-for-byte identical
to `flat_param_to`'s. This matters because it sets the right expectation: `direct_collectives` is
strong evidence this hardware class needs a non-NCCL-P2P transport, not a pre-validated fix for FSDP's
specific collective pattern (see §2).

---

## 1. What `direct_collectives` actually is

Source studied (read-only, via IP66, `/home/nexlayer/liquid-brain-native`):
`kernels/direct_collectives/direct_collectives.cu` (389 lines), `include/lbx/engine/direct_collectives.h`
(66 lines), `tools/pp/direct_p2p_smoke.cpp`, `tools/tp/direct_smoke.cpp`, `tests/test_direct_collectives.cpp`
(212 lines), `docs/native/design/direct-collectives-v1.md` (301 lines), `docs/native/design/tensor-parallel-v1.md`,
`include/lbx/engine/dsv4_stage.h`, `include/lbx/engine/dsv4_runner.h`.

(Note: `tools/pp/nccl_p2p_smoke.cpp` no longer exists under that name — the design doc records it was
renamed in-place to `tools/pp/direct_p2p_smoke.cpp` as part of the NCCL→direct-CUDA port, same tool,
new name, confirmed by `find`.)

### 1.1 API surface (`extern "C"`, in `direct_collectives.h`)

```
int  lbx_direct_create_pair(const int devices[2], size_t slot_capacity, void* groups_out[2]);
void lbx_direct_destroy_pair(const int devices[2], void* groups[2]);
int  lbx_direct_allreduce_bf16(void* group, void* buf, size_t count, cudaStream_t stream);
int  lbx_direct_send_bf16(void* group, const void* buf, size_t count, cudaStream_t stream);
int  lbx_direct_recv_bf16(void* group, void* buf, size_t count, int peer_rank, cudaStream_t stream);
int  lbx_direct_check_status(void* group);
void lbx_direct_abort(void* group);
```

**Collective operations supported: all-reduce (sum, bf16 only) and point-to-point send/recv.**
Nothing else — no all-gather, no reduce-scatter, no broadcast as a distinct op, no any/avg/max
reductions.

### 1.2 Data-staging mechanism

- One **shared, pinned host buffer**, sized `num_ranks * slot_capacity * sizeof(bf16)`, allocated once
  per pair via `cudaHostAlloc(cudaHostAllocMapped | cudaHostAllocPortable)`. `slot_capacity` is fixed at
  bootstrap to the max element count any single call on that pair will ever move (sized by the caller
  from the model's real shapes, e.g. `max_step_tokens * hidden_size`).
- Each rank calls `cudaHostGetDevicePointer` **while its own device is current** to get its own
  device-side virtual-address mapping into that *same* underlying host allocation — this is the crux of
  the mechanism: real cross-PCIe DMA issued by each SM's own memory system against host RAM, never a
  GPU-to-GPU transaction, never a socket.
- A small `arrived` flags array (one `uint32_t` per rank, host-allocated, mapped the same way) plus one
  `aborted` flag word serve as the synchronization primitive. Each rank also owns its own **private**
  `cudaMalloc`'d device-side status word (not shared) recording the per-call outcome.
- **No double-buffering.** One staging buffer is reused call after call; an `epoch` counter
  (incremented once per call, by the issuing rank) tags each arrival so a stale flag from a previous
  call can't be mistaken for the current one. Correctness comes from the epoch tag, not from buffer
  rotation — there is no mechanism to let call *N+1*'s staging write begin while call *N*'s consumer
  (reduce/copy-in kernel) hasn't finished reading the buffer.
- **Synchronization:** `__threadfence_system()` after a payload write and before the corresponding
  arrival-flag write (CUDA's documented primitive for host↔device visibility ordering across devices
  over PCIe — a plain store gives no such cross-device guarantee). Waits are **bounded**: a fixed spin
  budget (`kSpinIterations = 4,000,000`) with a `__nanosleep(1000ns)` backoff between checks, checking an
  externally-settable abort flag every iteration — never an unbounded wait. On timeout or abort, the
  dependent finishing kernel (reduce or copy-in) is skipped entirely, leaving the destination buffer
  untouched rather than writing from data that never arrived. This bounded-wait discipline is explicitly
  a lesson learned from incident #2 above: an unbounded hand-rolled wait would just relocate the same
  failure mode into new code.
- **Status is read back lazily, never inline.** Every collective call is a pure stream enqueue (never
  synchronizes). The caller is expected to call `lbx_direct_check_status` only at a point that *already*
  does a host sync (e.g. an existing D2H logits copy) — this preserves the original `ncclAllReduce`/
  `ncclSend`/`ncclRecv` contract of adding no round-trip latency per call.
- **Documented, known gap (`direct-collectives-v1.md` §7):** in a tight pipelined loop with several
  back-to-back collectives before any host sync, only the *last* call's status is observable — the
  per-call status word is overwritten by each new call's `cudaMemsetAsync` before an earlier one could be
  read. A mid-loop failure is masked until the final check. Stated plainly as unsolved in lbx's own docs.
- **Rank count is hardcoded to 2** in `lbx_direct_create_pair` (`constexpr int kNumRanks = 2`). The
  *kernels themselves* are rank-count-generic (`direct_wait_all_kernel`/`direct_reduce_bf16_kernel`
  already loop over an arbitrary `num_ranks`) — lbx's own docs call out that extending the bootstrap to
  N ranks would be a smaller step than the original NCCL port was, but it has not been done.
- **Dtype: bf16 only.** A deliberate scope decision (lbx's engine only ever moves bf16 residual-stream
  activations over this path) — explicitly not a generic byte-mover.
- **Process/thread model:** this entire mechanism assumes **one host process, one host thread, driving
  both ranks' CUDA contexts directly** (`cudaSetDevice` calls interleaved on one thread) — this is lbx's
  whole engine architecture (one serving process per node, not one process per GPU), not an incidental
  detail. It is baked into the API: the caller "enqueues both ranks' calls from one host thread." There
  is no cross-process IPC, shared-memory-across-processes, or rendezvous logic anywhere in this
  mechanism, because lbx has never needed any — its two "ranks" are two device contexts in one process.

### 1.3 Measured performance (real numbers, from `direct-collectives-v1.md` §5, 2x RTX PRO 6000, this exact hardware)

Point-to-point, vs the old `NCCL_P2P_DISABLE=1` path (same element counts; byte-size caveat noted
honestly in the source — old test moved fp32 stand-ins, new test moves real bf16, so part of the gap is
simply fewer bytes, not a pure transport win):

| size | NCCL (`NCCL_P2P_DISABLE=1`) | direct-CUDA | verdict |
|---|---|---|---|
| decode-sized, 2048 elem | 0.029 ms | 0.0155 ms | ~1.9x faster (half the bytes moved) |
| prefill-sized, 2.1M elem | 0.270 ms | 0.190 ms | ~1.4x faster (half the bytes moved) |

All-reduce (no prior isolated NCCL all-reduce microbenchmark existed in lbx's history to compare
against — stated as a real, named gap rather than fabricated):

| size | direct-CUDA all-reduce median |
|---|---|
| decode-sized, 2048 elem bf16 | 0.0277 ms |
| prefill-sized, 2.1M elem bf16 | 0.633 ms |

All-reduce costs ~1.8–3.3x point-to-point at matched element counts — expected, since all-reduce does a
mutual two-way barrier plus a read of every rank's slot, vs send/recv's one-way signal plus a single
copy.

**Important scale caveat, stated plainly:** every one of these numbers is at lbx's own *inference
activation* sizes — a few KB to a few MB per call. FSDP's actual payload per all-gather is a full
flat-parameter shard: realistically many MB to low GB per FSDP unit, on **every** forward and **every**
backward pass, for every unit in the model (dozens to hundreds of units in a real model). These
measurements are real and honestly reported, but they do not by themselves validate that this mechanism
is fast enough, or scales the way FSDP needs, at FSDP's actual payload sizes and call frequency. New
measurement at FSDP-realistic sizes is required before treating this as production-ready for training
(see the test plan in §4).

---

## 2. Honest fit-for-purpose assessment

FSDP's real communication pattern (per `PLAN.md`'s survey, rows 11/12): **all-gather** to unshard a flat
parameter before forward/backward use (every rank contributes its own shard; the output is the full,
concatenated tensor, replicated identically on every rank — not a reduction), and **reduce-scatter** to
shard-and-reduce gradients after backward (every rank's local gradient is summed across all ranks, then
the sum is split so each rank keeps only its own slice — the output is *not* identical across ranks).
Both are genuine N-rank collectives, where N is the FSDP world size (realistically far more than 2 in
production training).

`direct_collectives` was built for lbx's own inference-time tensor/pipeline-parallelism: TP=2's
all-reduce (Megatron-style column→row-parallel sharding, one all-reduce per sync point) and PP=2's
point-to-point activation handoff between exactly two pipeline stages
(`dsv4_stage.h`/`dsv4_runner.h`: "PP=2 hands the stream over through host memory (no P2P on this box)").

Point by point, against FSDP's actual needs:

1. **Rank count.** Hard-coded to 2, with a single-process/single-host-thread architecture baked into the
   whole design, not just a bootstrap parameter. FSDP needs arbitrary world size. The reduction/barrier
   *kernels* already generalize (confirmed by lbx's own code and docs), but the surrounding bootstrap,
   and especially the "one process drives every rank" assumption, does not — real FSDP training deploys
   one process per GPU (`torchrun`-style), which is a fundamentally different rendezvous and
   shared-memory model (cross-process shared memory / `mmap`'d region per node, not one process's
   `cudaHostAlloc`) than anything this code does today. **This is a real redesign, not a parameter bump.**
2. **Collective operations.** All-reduce (sum) and send/recv only — no all-gather, no reduce-scatter.
   Neither exists in `direct_collectives` today, in any form.
   - **All-gather is actually a *good* structural fit for the transport mechanism**, and arguably
     *simpler* than what's already there: skip the fp32-accumulate-and-reduce step of
     `direct_reduce_bf16_kernel` entirely — each rank copies its own shard into its own slot of the
     shared staging buffer, the same bounded-spin barrier fires, then every rank copies *all* slots
     (not just its own) into its full-size destination buffer. The staging-buffer-plus-epoch-tagged-
     barrier pattern transfers directly; only the "finishing" kernel's logic changes (copy-the-whole-
     buffer vs sum-the-whole-buffer).
   - **Reduce-scatter is a believable but non-trivial adaptation**: logically "all-reduce then keep only
     your own slice," so the barrier and the per-rank-slot staging transfer cleanly, but the reduce
     kernel needs to both sum across all ranks' full-size inputs *and* know which output slice belongs
     to the calling rank — a real, if modest, piece of new kernel logic, not a drop-in reuse of
     `direct_reduce_bf16_kernel` as written (that kernel computes the full sum, identically, into every
     rank's own full-size buffer — FSDP needs each rank to end up with only its shard of that sum).
3. **Dtype.** bf16-only, a deliberate scope choice for lbx's own activations. FSDP training realistically
   needs at least fp32 (master weights / optimizer state, in mixed-precision training) in addition to
   bf16/fp16 — this needs a real, generalized byte/dtype-templated path, not an assumption that bf16-only
   is enough.
4. **Scale.** Validated only at lbx's own inference-activation sizes (KB–few MB). FSDP's payloads are
   larger by one to three orders of magnitude and occur far more frequently (every training step, every
   unit, both directions) — unvalidated at this scale, as stated in §1.3.
5. **No double-buffering / overlap.** FSDP's actual throughput model depends on overlapping the *next*
   unit's all-gather with the *current* unit's compute (FSDP2's async prefetch scheduling, `PLAN.md` row
   12). `direct_collectives`' single shared buffer, gated only by an epoch counter, has no mechanism to
   let one call's staging begin while a prior call's consumer is still reading — a real gap for FSDP's
   performance model, even though it was never a problem for lbx's own use (one call at a time, blocking
   on the result almost immediately in a tight serving loop).
6. **Status/error semantics.** `direct_collectives`' deferred, last-call-wins status model is a
   deliberate, well-reasoned simplification *for lbx's own inline inference call sites* (a handful of
   calls per request, checked at an already-existing sync point). FSDP's backward pass triggers
   potentially many bucketed reduce-scatter calls from autograd hooks before any natural host sync
   point — the same "only the last status is visible" gap documented in lbx's own §7 would be worse,
   not better, under FSDP's call pattern, and needs its own design (see §3).

### Verdict

**`direct_collectives` is not a drop-in FSDP communication backend, and claiming otherwise would be
overclaiming.** What it *is*: a genuine, production-validated answer to the exact hazard class that
caused IP66's `flat_param_to` deadlock — "GPU-to-GPU P2P DMA hangs on this non-NVLink, cross-socket
topology; route through pinned host memory instead, with bounded, abortable waits, and no persistent
communicator object to hang in on teardown." That answer, and the underlying mechanism, generalizes well
in principle (the reduction kernels are already rank-count-generic; all-gather is a natural, arguably
easier extension of the existing staging pattern).

**The honest path forward is to adopt lbx's host-memory-staging mechanism and safety philosophy as the
underlying transport primitive for the non-P2P fallback path, and build genuine N-rank, multi-dtype,
double-buffered all-gather/reduce-scatter collective logic on top of it from scratch** — not to reuse
`direct_collectives`' code 1:1, and not to pretend it already does what FSDP needs. Concretely: borrow
(a) `cudaHostAlloc(Mapped|Portable)` + `cudaHostGetDevicePointer` per-rank mapping as the data-movement
primitive instead of `cudaMemcpyPeerAsync`, (b) epoch-tagged arrival flags + `__threadfence_system` +
bounded-spin-with-abortable-backoff as the synchronization primitive, and (c) "no persistent
communicator object, teardown is a plain synchronous free" as the lifecycle philosophy. Do **not** borrow
the 2-rank/single-process bootstrap, the bf16-only payload path, the single-buffer (no overlap) staging
model, or the deferred/last-call-wins status model — each of those needs to be redesigned for FSDP's
actual shape.

---

## 3. Proposed architecture for the C++ FSDP replacement

Four layers, each independently testable:

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. FSDP orchestration layer (new C++, Reducer-patterned)              │
│    - per-FSDP-unit autograd hook registration                         │
│    - WHEN to communicate: prefetch next unit's all-gather during      │
│      current unit's forward/backward; bucket gradients as they become │
│      ready during backward                                            │
│    - sharding-offset math + flat-parameter view reconstruction        │
│      (ported from FSDP1/_flat_param.py + FSDP2's                      │
│      _fsdp_param_group.py STATE, not from Reducer.cpp's code)         │
├──────────────────────────────────────────────────────────────────────┤
│ 2. Collective API (new, transport-agnostic)                           │
│    all_gather(shard, out, group) / reduce_scatter(in, out_shard,      │
│    group) / all_reduce(...) [HSDP/optimizer-state sync] /             │
│    broadcast(...) [initial sync]                                      │
├──────────────────────────────────────────────────────────────────────┤
│ 3. Transport, selected per process-group at init time                 │
│    ┌────────────────────────┐   ┌───────────────────────────────────┐ │
│    │ Fast path:              │   │ Fallback path:                   │ │
│    │ ProcessGroupNCCL        │   │ new host-staged backend, modeled  │ │
│    │ (libtorch, as-is,       │   │ on direct_collectives' MECHANISM  │ │
│    │ PLAN.md row 13 –        │   │ (not its code): N-rank, multi-    │ │
│    │ free win)               │   │ dtype, double-buffered all-gather │ │
│    │                          │   │ / reduce-scatter                 │ │
│    └────────────────────────┘   └───────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────┤
│ 4. Topology detection (init-time, real CUDA/NVML APIs, no shelling    │
│    out to nvidia-smi) — picks which transport each pair/group gets    │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.1 Layer 1 — orchestration, modeled on `Reducer.cpp`'s *pattern*, not its code

`torch/csrc/distributed/c10d/reducer.cpp` proves the right shape already exists in this codebase: C++-
registered post-accumulate-grad autograd hooks → bucket-readiness tracking → a call into the collective
layer once a bucket is ready. That hook-registration/bucketing *pattern* is what FSDP's orchestration
layer should copy. What it must **not** copy is `Reducer`'s actual bucket semantics: `Reducer` is
DDP's, unsharded — a bucket's contents *are* the gradient, communicated with a single all-reduce.
FSDP's buckets need shard-aware slicing (each rank holds only its 1/N shard of each parameter) and two
distinct collective shapes at two distinct points in the step (all-gather before use, reduce-scatter
after backward) rather than one all-reduce. The real, genuine Python logic PLAN.md flagged in rows 11/12
— per-rank shard-offset math, unshard/reshard view-reconstruction, the prefetch state machine deciding
*when* to fire each collective — is ported from FSDP1's `_flat_param.py`/`_init_utils.py` and FSDP2's
`_fsdp_param_group.py`/`_fsdp_collectives.py`, reusing the already-C++ `ChunkCat.cpp`/`CollectiveCopy.cpp`
buffer-packing kernels those files already call into (`PLAN.md` row 12 — a free win, unaffected by any
of this transport discussion).

### 3.2 Layer 2 — a transport-agnostic collective API

A small interface (`all_gather`, `reduce_scatter`, `all_reduce`, `broadcast`) that the orchestration
layer calls without knowing which transport backs it. This is also where the explicit fix for the
*other* root cause PLAN.md §3 identified lives: `_init_utils.py`'s per-rank, per-parameter loop of
blocking, uncoalesced `.to(device)` calls, with no explicit cross-rank barrier before the first
collective. The C++ replacement should coalesce device-transfer calls and insert an explicit barrier at
that handoff — a correctness fix that is independent of which transport (NCCL or host-staged) ends up
underneath, and should not be allowed to regress just because the new transport exists.

### 3.3 Layer 3 — the two transports

- **Fast path — `ProcessGroupNCCL`, used as-is.** Already pure C++, already has `allgather`/
  `reduce_scatter`/`allreduce`/`broadcast`, comm caching, a watchdog, and a flight recorder
  (`PLAN.md` row 13). No reason to touch this when real P2P works.
- **Fallback path — a new implementation, not a port of `direct_collectives`.** Generalizes rank count
  (N, not 2), supports both bf16 and fp32 (at minimum), adds double-buffering so one unit's all-gather
  can be staged while the previous unit's compute (and the transport's own consumer kernel) is still
  running, and implements all-gather/reduce-scatter as first-class operations rather than adapting
  all-reduce. Reuses, deliberately, three specific design decisions from `direct_collectives` because
  they are the parts that are actually about the hazard, not about lbx's specific 2-rank shape:
  1. `cudaHostAlloc(Mapped|Portable)` + per-rank `cudaHostGetDevicePointer` instead of
     `cudaMemcpyPeerAsync`, for the same documented reason: direct GPU-to-GPU P2P is the confirmed hang
     hazard on this hardware class, even when the driver claims P2P is available.
  2. Bounded, abortable waits (fixed spin budget + `__nanosleep` backoff + externally-settable abort
     flag) instead of an unbounded wait — the lesson from lbx's own incident #2 (§0): a hand-rolled
     synchronization primitive that isn't itself bounded can reintroduce the same failure class in a new
     shape.
  3. No persistent "communicator" object with its own teardown protocol — lifecycle is plain
     `cudaHostAlloc`/`cudaFreeHost` and `cudaMalloc`/`cudaFree`, torn down with ordinary synchronous
     calls, so there is nothing with internal socket/thread state left to hang on exit.
  A real multi-process rendezvous mechanism (since real FSDP training runs one process per GPU, not
  lbx's one-process-drives-every-device model) is new work this layer must add — e.g., a POSIX shared-
  memory segment per node (`shm_open`/`mmap`) in place of a single process's `cudaHostAlloc`, opened by
  every rank's process on that node, plus the normal c10d-style rendezvous (store-based rank/address
  exchange) to agree on the shared-memory name/layout before mapping it.

### 3.4 Layer 4 — topology detection, using real APIs, probed not just queried

Two things are needed, and the design deliberately does **not** trust the first one alone, because
IP66's own incident history shows it can lie:

1. **Static capability check:** `cudaDeviceCanAccessPeer(&canAccess, device, peerDevice)` (real CUDA
   runtime API — no shelling out to `nvidia-smi`), plus NVML (`nvmlDeviceGetNvLinkState` /
   `nvmlDeviceGetNvLinkCapability`) as the programmatic equivalent of `nvidia-smi topo -m`'s NVLink
   column, for logging/diagnostics and as a first-pass filter.
2. **Live, bounded correctness probe — the part that actually matters.** `tensor-parallel-v1.md`
   records, as a real, dated fact: NCCL's own P2P detection on this exact hardware said P2P was usable
   (`nvidia-smi topo -p2p r` reported `OK`, NCCL's runtime detected `isAllDirectP2p 1 directMode 1`) and
   it hung anyway. A static query is therefore **not sufficient** to trust the fast path. At
   process-group init time, run a small, timed `cudaMemcpyPeerAsync` round-trip between every rank pair
   that will need to communicate, on a watchdog-bounded deadline (a host thread with a hard wall-clock
   timeout, independent of CUDA's own synchronization — exactly the "every wait must be bounded and
   externally abortable" discipline `direct_collectives` itself was built around). Only select the fast
   `ProcessGroupNCCL` path for a given pair if **both** the static check *and* the live probe succeed
   within the deadline; fall back to the host-staged transport otherwise. This probe is itself a direct,
   practical lesson from the `direct_collectives` incident history, independent of whether any of its
   code is reused: don't make every future NVLink-equipped node (the expected shape of real B200
   production nodes) pay the host-staging tax, but don't trust a topology query that has already been
   observed to be wrong on real hardware either.

---

## 4. Phased implementation sub-plan (FSDP port itself — separate from the transport design above)

**Phase 0 — Transport layer, standalone, no FSDP dependency.**
Build the generalized N-rank, multi-dtype, double-buffered host-staged all-gather/reduce-scatter
backend, the topology-probe-based selection logic, and the `ProcessGroupNCCL` fast-path wiring, as an
independently buildable/testable C++ library. Validation:
- Correctness goldens at multiple rank counts and sizes, mirroring `test_direct_collectives.cpp`'s own
  discipline (exact output across every rank, bounded-timeout and abort behavior exercised
  deliberately, argument-refusal cases covered) — but for all-gather/reduce-scatter specifically, not
  all-reduce/send-recv.
- **The regression test that matters most, stated explicitly per the task brief:** reproduce IP66's
  original deadlock scenario as an automated regression test. On IP66's exact hardware (confirmed `SYS`
  topology), force the fallback transport, run a sustained loop of all-gather/reduce-scatter calls at
  FSDP-realistic sizes (full flat-parameter-shard-sized payloads, not lbx's KB–MB activation sizes) for
  an extended duration, and confirm the process completes and exits cleanly with no lingering process
  (the same method `direct-collectives-v1.md` §5a used: wall-clock `time` plus
  `nvidia-smi --query-compute-apps` showing empty after exit) — i.e., directly confirm the new
  implementation does **not** hang where the old Python FSDP did.
- New throughput measurement at FSDP-realistic payload sizes (§1.3's scale caveat) — do not assume
  lbx's inference-activation-scale numbers transfer.

**Phase 1 — FSDP orchestration skeleton, no real distributed run yet.**
Port the sharding/offset math and flat-parameter view-reconstruction logic (PLAN.md rows 11/12's
"genuine Python logic") as pure C++ data structures, unit-testable deterministically given shapes and
world size, without needing real multi-GPU hardware.

**Phase 2 — Wire orchestration to transport.**
Autograd hook registration per FSDP unit, bucketing, and the prefetch/overlap state machine (layer 1),
calling the Phase 0 collective API (layer 2). This is where the `_init_utils.py` device-transfer-
batching/barrier fix (§3.2) is implemented and tested.

**Phase 3 — End-to-end validation on real hardware.**
Run the new C++ FSDP-equivalent on IP66 with a real (even if small) model:
- Confirms no hang, by the same direct process-level evidence as Phase 0's regression test, now at the
  level of full training steps rather than isolated collective calls.
- Numerical correctness vs the old Python FSDP path on the same model/data, same discipline as
  `direct-collectives-v1.md` §5b (byte-for-byte or within-stated-tolerance agreement, not "it ran
  without crashing").
- Throughput, on both IP66 (host-staged fallback — now a real, quantified number instead of an assumed
  one) and, once available, real NVLink-equipped hardware (confirm the fast path is actually selected
  there and is not regressed vs plain `ProcessGroupNCCL`).

**Phase 4 — feature completeness.**
HSDP (hybrid-shard) if required, checkpoint-format compatibility work (bundles with the optimizer
checkpoint-bridge work from `PLAN.md` step 1/row 8-9), and a gradual rollout plan.

---

## 5. Summary

- `direct_collectives` is real, production-validated, and directly relevant: it is Nexlayer's own proof
  that on this non-NVLink, cross-socket hardware class, routing GPU-to-GPU data through
  `cudaHostAlloc`'d pinned host memory — with bounded, abortable synchronization and no persistent
  communicator object — avoids the exact hazard family that caused both IP66's `flat_param_to` deadlock
  and lbx's own two NCCL incidents.
- It is **not** a drop-in FSDP transport: it is hardcoded to 2 ranks in a single-process architecture,
  supports only all-reduce and point-to-point send/recv (not all-gather/reduce-scatter), only bf16, has
  no double-buffering, and has only been measured at inference-activation scale, not FSDP's
  flat-parameter-shard scale.
- The honest, non-overclaiming path: **adopt the mechanism and safety philosophy, not the code.** Build
  a new, genuinely N-rank, multi-dtype, double-buffered all-gather/reduce-scatter backend using the same
  pinned-host-memory-staging primitive, as the fallback transport selected only when a live, bounded P2P
  probe fails — keeping `ProcessGroupNCCL` as the fast path on real NVLink hardware, so NVLink-equipped
  B200 production nodes never pay a tax they don't need.
- The FSDP rewrite itself stays modeled on `Reducer.cpp`'s hook/bucket *pattern* for scheduling, but its
  actual sharding/view-reconstruction logic is ported from FSDP1/FSDP2's own Python, not from `Reducer`
  (which is architecturally unsharded and solves a simpler problem).
