# PLAN: Eliminating the Python Layer from PyTorch's Training Path

**Project:** Nexlayer `pytorch-native-training` (fork of `pytorch/pytorch`)
**Goal:** make a GRPO-style RL post-training step (FSDP actor forward/backward +
optimizer step + weight sync) runnable as 100% C/C++, no Python interpreter at
runtime. First step toward LB Native GRPO folding RL post-training into `lbx`.
**Scope:** training path only. Not inference, not TorchScript/JIT export, not
mobile. Survey + plan only — no rewrite code in this document or this branch.
**Method:** full source survey of this fork (`torch/`, `aten/`, `c10/`,
`torch/csrc/`), cross-checked against `torch/csrc/api/` (libtorch's pure-C++
frontend) to see how much of "the rewrite" already exists and just needs to be
linked rather than written.

---

## 1. Executive summary

The single biggest finding: **a large fraction of a Python-free training loop
already exists today as libtorch**, PyTorch's pure-C++ frontend
(`torch/csrc/api/`). Tensor storage, the dispatcher, the autograd engine, NCCL
process groups, and a real `torch::nn` module library (Linear, Conv, LayerNorm,
RNN, Transformer blocks) are already C++-only, with zero Python dependency,
and already ship as `libc10.so` / `libtorch_cpu.so` as separate link targets
from the Python bindings (`libtorch_python.so`). None of that needs a rewrite
— it needs to be *used directly*, bypassing Python entirely rather than
"stripped" of Python.

The two places with real, load-bearing Python logic directly on the GRPO hot
path are:

1. **FSDP** (`torch/distributed/fsdp/`, both FSDP1 and FSDP2/`_fully_shard`) —
   ~13,000 lines, 55-70% genuine scheduling/bookkeeping logic (sharding,
   view-reconstruction, prefetch/overlap, hook orchestration), with **zero C++
   counterpart anywhere in libtorch**. This is the largest rewrite and also
   the one that actually bit us in production (the `flat_param_to`
   non-NVLink deadlock).
2. **Optimizers** (`torch/optim/`) — real per-step math (~400-450 LOC for
   AdamW across its single/multi-tensor/fused variants), and while libtorch
   *does* already have a C++ `torch::optim::AdamW`, it's not a drop-in
   replacement: it lacks `foreach`/`fused`/`capturable` support and its
   serialization format is incompatible with Python checkpoints.

Everything else (data loading, the Python `Tensor`/`Module` pybind11 bindings)
either shouldn't be eliminated (data loading) or doesn't need to be touched at
all because we're not going through it in the first place (pybind11 bindings
— irrelevant once the training loop is written directly against libtorch).

---

## 2. Subsystem survey

| # | Subsystem | Current Python/C++ split | Verdict | Effort | Why it matters for GRPO |
|---|---|---|---|---|---|
| 1 | **c10 / ATen core** (`c10/`, `aten/src/ATen/`) — tensor storage, dispatcher, op registration/kernels | Pure C++. `grep` for `Python.h`/`pybind11` across `c10/` hits only an opaque `PyObject*` interop pointer in `SafePyObject.h`/`intrusive_ptr.h`; no CPython dependency. Codegen (`torchgen`) emits C++ registration glue from `native_functions.yaml`, but kernels and the dispatcher engine itself are hand-written C++. Ships as `libc10.so` + `libtorch_cpu.so`, separate CMake targets from `torch_python`. | **REUSE AS-IS** | None | Every tensor op in the GRPO step (rollout tensors, log-probs, advantages, FSDP forward/backward, optimizer math) bottoms out here. Zero risk, zero work — just link. |
| 2 | **Autograd engine — execution** (`torch/csrc/autograd/engine.{h,cpp}`, `function.h`/`node.h`, `variable.cpp`) | Pure C++ worker-thread/queue/topological-traversal engine. `Node::apply` is a pure-virtual C++ interface. Graph construction in a normal eager script goes through codegen'd `VariableType` wrappers (already C++, invoked transparently under every Python op call). | **REUSE AS-IS** | None | FSDP forward/backward is autograd-driven; the actual backward execution needs zero changes. |
| 3 | **Autograd engine — C++-authored graph construction & custom ops** (`torch/csrc/api/include/torch/autograd.h`, `custom_function.h`) | `torch::autograd::backward()`/`grad()` and `torch::autograd::Function<T>` (C++ custom-autograd-op base class) already exist and are used internally. A libtorch C++ program can build and run a full backward graph without ever touching Python. | **REUSE AS-IS** (free win) | None | This is the mechanism a C++ training loop uses for `.backward()` — already solved. |
| 4 | **Python-only autograd glue** (`python_function.cpp` 2418 lines, `python_variable.cpp` 3984 lines, `python_engine.cpp` 500 lines) | This is specifically the bridge for *Python-authored* `torch.autograd.Function` subclasses and the GIL-release wrapper around `Engine::execute`. Not needed if nothing is authored in Python. | **IRRELEVANT — skip, don't rewrite** | None (not on critical path once training loop is C++-native) | N/A if the actor/optimizer code is written in C++ from the start. |
| 5 | **Python `torch.Tensor`/`torch.nn.Module` pybind11 bindings** (`torch/csrc/Module.cpp` 4003 lines, `python_variable.cpp`) | `THPVariable` is a thin wrapper (`PyObject_HEAD` + `at::Tensor cdata` + 2 hook pointers); the 4000-line files are mostly argument-marshaling/module-init glue, not business logic. The one real piece of Python-side state is backward-hook storage on the Python object itself. | **DON'T TOUCH — not required at all**, not a rewrite target | N/A | We don't need to strip these bindings; we simply never link `torch_python` and write the training loop directly against the C++ frontend (row 6). Stripping pybind11 glue is only a project if someone insists on reusing the *Python object model* in C++, which we don't. |
| 6 | **libtorch C++ frontend** (`torch/csrc/api/include/torch/{nn,optim,data}/`) | Full, real, actively maintained module library: `Linear`, `Conv2d`, `LayerNorm`, `BatchNorm`, `RNN/LSTM/GRU`, Transformer encoder/decoder/layer, `Embedding`, `Dropout`, pooling, plus `Sequential`/`ModuleList`/`ModuleDict` containers, plus a C++ `optim` and `data` namespace. Presence of Transformer building blocks indicates active maintenance, not a legacy/abandoned shim. | **REUSE AS-IS (free win)** | None to adopt; extension work happens per-row below (optimizer, FSDP) | This is literally "the rewrite" for the module/tensor API layer — it already exists. A GRPO actor model (transformer decoder stack) is directly expressible here. |
| 7 | **Optimizer step — algorithm** (`torch/optim/adamw.py`, `adam.py` 998 lines, `sgd.py` 544 lines) | Real control-flow (~400-450 LOC for AdamW: bias correction, capturable/differentiable branches, amsgrad), but triplicated across `_single_tensor_*`, `_multi_tensor_*` (foreach), and `_fused_*` variants implementing the *same* math three ways, plus a device/dtype dispatch chooser. | **REWRITE CANDIDATE** | Medium | Every GRPO step calls this exactly once per param group; it's on the hot path every iteration, but it's self-contained (no distributed dependency) so it can be built and validated independently of FSDP. |
| 8 | **Optimizer step — base plumbing** (`torch/optim/optimizer.py` 1274 lines) | Real but non-algorithmic: param-group merging, `state_dict`/`load_state_dict` (dtype/device-preserving), hooks, `add_param_group` validation. ~150-200 real LOC. | **REWRITE CANDIDATE (bundle with #7)** | Small–Medium | Needed so checkpoints/resume work; not hot-path CPU time but blocks reuse of existing Python checkpoints if skipped. |
| 9 | **Optimizer step — existing C++ implementation** (`torch/csrc/api/src/optim/adamw.cpp`, `.../adam.cpp`, `.../sgd.cpp`) | Correct algorithm (decoupled weight decay, bias correction, amsgrad all present and verified in `adamw.cpp`), but: (a) **no `foreach`/`fused`/`capturable`** — confirmed absent from `AdamWOptions` in `adamw.h` and absent from `src/optim/` entirely (no `torch::_foreach_*` calls); `state.step()` is a plain `int64_t`, architecturally incompatible with CUDA-graph capture; (b) serialization is `OutputArchive`/`InputArchive` (`torch/csrc/api/src/optim/serialize.cpp`), structurally incompatible with Python's dict-based `state_dict()` — no bridging utility exists. | **NOT a free win — REWRITE/EXTEND CANDIDATE** | Medium (add foreach batching + a state-dict-format shim); **Large** if `fused`/`capturable` CUDA-graph support is actually required for throughput parity | Correctly flagged by the user's instinct that optim is "almost entirely Python" — but note the surprise: C++ already has *a* correct AdamW, it's just not performance- or checkpoint-compatible yet. This changes the work from "write an optimizer" to "extend an existing one," which is meaningfully less risky. |
| 10 | **Data loading** (`torch/utils/data/dataloader.py` 1705 lines, `collate.py`, `sampler.py`) | Multiprocessing worker pool, pickling batches/exceptions across process boundaries, `pin_memory` thread — all exists specifically to parallelize arbitrary **Python** `Dataset.__getitem__`/`collate_fn` around the GIL. A separate, long-unmaintained C++ `torch::data::DataLoader` exists (`torch/csrc/api/include/torch/data/`) but only parallelizes over a C++ `Dataset`, with no path to running user Python code. | **LEAVE AS PYTHON BOUNDARY — do not rewrite** | N/A (by design) | Rollout/reference-logprob/advantage data in a GRPO step is typically user/framework (verl) composed Python logic anyway; the right boundary is "Python hands collated tensors to the C++ training core," not full elimination. |
| 11 | **FSDP1** (`torch/distributed/fsdp/fully_sharded_data_parallel.py` 2192 lines, `_flat_param.py` 2853 lines — class `FlatParamHandle`, `_init_utils.py` 1236 lines, `_runtime_utils.py` 1654 lines) | ~55-70% genuine logic depending on file: per-rank shard-offset math, unshard/reshard view-reconstruction (180-280 line functions), user-mutated-storage writeback, wrap-policy module-tree traversal, hybrid-shard PG construction, and **full hook orchestration + an explicit prefetch state machine** deciding *when* to fire all-gather/reduce-scatter (the collectives themselves are thin c10d pass-throughs). `flat_param_to` itself (`_flat_param.py:2507`) is an 8-10 line wrapper — **not** where the real complexity is. | **REWRITE CANDIDATE** | Large | This is the direct cause of the production incident (see §3) and the single densest block of Python control-flow directly in the GRPO forward/backward hot path. |
| 12 | **FSDP2 / `_fully_shard`** (`torch/distributed/fsdp/_fully_shard/_fsdp_param.py` 1765 lines, `_fsdp_param_group.py` 1423 lines — densest file, 65-70% real, `_fsdp_collectives.py` 1082 lines) | Same character as FSDP1 but DTensor-based per-param sharding-spec construction, explicit CUDA-stream-overlapped async scheduling, backward prefetch, HSDP. `_fsdp_collectives.py`'s `foreach_all_gather`/`foreach_reduce` call into **already-C++** packing kernels (`torch/csrc/distributed/fsdp/ChunkCat.cpp`, `CollectiveCopy.cpp`) before handing off to `ProcessGroupNCCL` — confirms bucketing/overlap/scheduling *decisions* are 100% Python, only the buffer-packing and the collective call itself are C++. | **REWRITE CANDIDATE** | Large | This is `torch.distributed.fsdp.FullyShardedDataParallel`'s modern successor and is what a current `verl`-based actor would actually be using. |
| 13 | **c10d / NCCL process groups** (`torch/csrc/distributed/c10d/ProcessGroupNCCL.{cpp,hpp}` 6276/1573 lines, `ProcessGroup.hpp`, `Backend.hpp`) | Pure C++, confirmed zero Python entanglement in the collective path (only a generic lazy-cache `PyObjectSlot`, same pattern as c10 elsewhere). Full `allreduce`/`allgather`/`reduce_scatter`/`broadcast`/NCCL-comm-caching/watchdog/flight-recorder already here. | **REUSE AS-IS (free win)** | None | This is the actual communication substrate under FSDP and weight-sync; it's already exactly what a C++ FSDP-equivalent would call into. |
| 14 | **DDP's `Reducer`** (`torch/csrc/distributed/c10d/reducer.cpp`, 2724 lines) | Proof-of-pattern: C++-registered autograd hooks → bucket-ready tracking → `all_reduce_bucket`, i.e. **sync-orchestration logic already exists in C++ in this codebase**, just for DDP's simpler (non-sharded) case, not FSDP's. | **REUSE AS A DESIGN TEMPLATE** for the FSDP rewrite | N/A | Directly informs how to build an FSDP-equivalent: hook-based bucket/shard state machine in C++, modeled on `Reducer`, calling `ProcessGroupNCCL` + the existing `ChunkCat`/`CollectiveCopy` kernels. |
| 15 | **Process-group init/rendezvous** (`torch/distributed/distributed_c10d.py`, 8531 lines) | ~30-40% real logic in `init_process_group`/`_new_process_group_helper`/`new_group` (backend dispatch, rendezvous); most of the rest is thin per-collective Python pass-through to the C++ `Work` object. `device_mesh.py` (1874 lines) has genuine mesh-slicing/sub-PG-caching logic for multi-dim parallelism. | **REWRITE CANDIDATE** | Small–Medium | Needed once FSDP rewrite requires a pure-C++ process-group bring-up path (no Python `init_process_group` call at all). Bundle into the FSDP phase — don't do standalone. |
| 16 | **Gradient checkpointing** (`torch/utils/checkpoint.py`, 2135 lines) | Reentrant path (`CheckpointFunction`, ~105 real LOC): CPU+CUDA RNG save/restore, `no_grad`-wrapped forward, `enable_grad`+restored-RNG recompute in backward — self-contained, no distributed dependency. Non-reentrant/selective path adds `__torch_dispatch__`-based policy-driven recompute caching, substantially more complex but separable/skippable for v1. No existing C++ equivalent anywhere (only an unrelated 15-line thread-local guard in `engine.cpp`). | **REWRITE CANDIDATE** | Small (reentrant path only, ~200-300 real LOC) | Needed for any realistic actor-model FSDP forward/backward (checkpointing is standard for memory under FSDP); small enough to build and validate standalone before layering FSDP on top. |

---

## 3. GRPO-specific question: what does libtorch's C++ API already cover?

Per the validated IP66 smoke test, a GRPO step is:
**generate rollouts (vLLM, out of scope of this fork) → reference log-probs →
advantage computation → actor FSDP forward/backward → optimizer step (AdamW)
→ weight sync.**

Checking the three PyTorch-fork pieces of that step against libtorch's pure
C++ API specifically:

- **Actor FSDP forward/backward (`torch.distributed.fsdp.FullyShardedDataParallel`
  / FSDP2 `fully_shard`)**: `grep -ril "sharding|fsdp|fully_shard"
  torch/csrc/api/` returns **zero matches**. The only parallel-training
  primitive in the C++ frontend is `torch::nn::parallel::DataParallel`
  (single-process multi-GPU replication, not sharding — unrelated). **FSDP
  requires the Python module today; there is no libtorch C++ counterpart at
  all.** Building one means writing new C++ from scratch (row 12 above),
  though it can reuse `ProcessGroupNCCL`, the `ChunkCat`/`CollectiveCopy`
  packing kernels, and the `Reducer` pattern as building blocks — it is not a
  from-nothing greenfield effort, but it is a genuine rewrite, not a link job.
- **Optimizer step (`torch.optim.AdamW`)**: libtorch's `torch::optim::AdamW`
  exists and is algorithmically correct, **but is not drop-in**: no
  `foreach`/`fused`/`capturable`, and an incompatible checkpoint format (row
  9). Usable as a starting point, not a free win.
- **Gradient checkpointing**: no C++ counterpart; small, contained,
  from-scratch C++ port needed (row 16).

**The production incident context**: the `FlatParamHandle.flat_param_to`
trace point from py-spy is a thin 8-10 line wrapper
(`self.flat_param.data = self.flat_param.to(*args, **kwargs)` +
view-refresh); the real suspects for the non-NVLink deadlock are upstream of
it — specifically `_init_utils.py`'s `_move_states_to_device` (a **per-rank,
per-parameter loop of blocking, uncoalesced `.to(device)` calls**, followed by
a rank-0 broadcast in `_sync_module_params_and_buffers`) and `_flat_param.py`'s
`pre_unshard` (a non-blocking H2D copy immediately preceding
`_all_gather_flat_param`, with no explicit cross-rank sync barrier at the
handoff). On NVLink hardware, GPU-fabric collectives and PCIe H2D traffic are
physically separate, which masks per-rank timing skew; on PCIe-only topology
they share a fabric, and per-rank copy-completion skew without batching is a
plausible deadlock/stall mechanism. This is a second, independent reason (on
top of pure Python-elimination) that FSDP init is the highest-value rewrite
target: a C++ rewrite modeled on `Reducer`'s hook/bucket pattern would
naturally batch and barrier these transfers instead of looping blocking calls
per-parameter.

---

## 4. Prioritized, ordered plan

**Do these in this order:**

### 1st — Optimizer step (rows 7, 8, 9)
**Why first:** fully self-contained (no distributed/FSDP dependency), so it
can be designed, built, and correctness-tested in isolation against a plain
(non-distributed) C++ training loop before any FSDP work begins. It's real
Python logic directly in the hot path (every single training step), but the
risk and scope are both bounded: extend the *existing* `torch::optim::AdamW`
C++ class rather than write one from nothing. Concretely:
  - Add `foreach`/multi-tensor batched update support to `torch::optim`
    (mirrors `torch/optim/adam.py`'s `_multi_tensor_adam`).
  - Add a `state_dict`-format bridge so C++-trained checkpoints can be saved/
    loaded compatibly with existing Python-trained checkpoints (needed for
    weight sync / eval / handoff to vLLM).
  - Treat `fused`/`capturable` (CUDA graph support) as a stretch goal, not a
    v1 requirement — defer unless profiling shows it's needed.

### 2nd — Gradient checkpointing (row 16)
**Why second:** small and contained (~200-300 real LOC for the reentrant
path), no distributed dependency, and it's a *prerequisite* for the FSDP work
in step 3 — FSDP-wrapped actor models will be checkpointed, so the mechanism
needs to exist and be validated (RNG state save/restore, no_grad/enable_grad
recompute) against the plain C++ autograd API before adding sharding
complexity on top. Build the reentrant path only; skip the non-reentrant/
selective `__torch_dispatch__`-based path for v1 unless a specific memory
target requires it.

### 3rd — FSDP sharding/scheduling (rows 11, 12, 15)
**Why third, despite being the actual pain point:** it is the largest rewrite
(large effort, ~13,000 lines of Python to replace, though only 55-70% of that
is real logic to port) and it depends on both prior steps being done first —
an FSDP-equivalent needs a working C++ optimizer step and working C++
checkpointing to wrap. It is also the subsystem with the most direct
production relevance (the non-NVLink `flat_param_to`/`_move_states_to_device`
deadlock, §3) and the densest Python control-flow directly on the GRPO
forward/backward hot path, so it must be tackled, just not first.
  - Target FSDP2's architecture (`_fsdp_param_group.py`'s state machine), not
    FSDP1 — it's the one `verl`/modern actors actually use.
  - Build it as a new C++ component modeled on `Reducer`'s hook/bucket
    pattern (row 14), calling directly into `ProcessGroupNCCL` (row 13) and
    reusing the existing `ChunkCat.cpp`/`CollectiveCopy.cpp` packing kernels
    (row 12) — these are free-win building blocks, not things to rewrite.
  - Explicitly fix the batching/barrier gap identified in §3 as part of this
    work (coalesce per-parameter device-transfer calls, add an explicit sync
    point before the first collective) — this both eliminates Python *and*
    fixes the known production bug in one pass.
  - Bundle `distributed_c10d.py`'s process-group init/rendezvous logic (row
    15) into this phase rather than doing it standalone, since FSDP needs
    process groups to exist first.

### Not prioritized — don't rewrite at all
- **Data loading** (row 10): leave as a Python boundary. Design the C++
  training core to receive collated tensors from a Python (or
  framework-level, e.g. `verl`) DataLoader process rather than reimplementing
  multiprocessing/pickling around arbitrary user datasets in C++.
- **Python `Tensor`/`Module` pybind11 bindings** (row 5): not a rewrite
  target — simply don't link `torch_python`, and write the training loop
  directly against the C++ frontend (row 6). There is nothing to strip
  because there is nothing we need from that layer.

---

## 5. Free wins — link against these, write nothing

- `libc10.so` + `libtorch_cpu.so`: tensor storage, dispatcher, op kernels,
  CUDA kernels (row 1).
- `torch::autograd::backward()`/`grad()`, `torch::autograd::Function<T>`:
  full C++ autograd, graph construction and execution, no Python (rows 2-3).
- `torch::nn::*` module library (Linear, Conv2d, LayerNorm, BatchNorm,
  RNN/LSTM/GRU, Transformer encoder/decoder, Embedding, Dropout, pooling,
  `Sequential`/`ModuleList`/`ModuleDict`): a complete, maintained C++ module
  API — the actor model itself can likely be expressed here directly (row 6).
- `ProcessGroupNCCL`/`Backend`/`Work` (`torch/csrc/distributed/c10d/`): the
  entire NCCL collective layer — allreduce/allgather/reduce_scatter/
  broadcast, comm caching, watchdog — pure C++, link directly (row 13).
- `ChunkCat.cpp`/`CollectiveCopy.cpp` (`torch/csrc/distributed/fsdp/`):
  already-C++ buffer-packing kernels used by FSDP2's collectives today —
  reusable primitives for the FSDP rewrite, not things to redo (row 12).
- `Reducer` (`torch/csrc/distributed/c10d/reducer.cpp`): not literally
  reusable for FSDP (it's DDP-specific, unsharded), but a working, in-tree
  proof that "C++ hook-based collective-orchestration engine" is a pattern
  this codebase already has — use it as the architectural template for the
  FSDP rewrite rather than designing from a blank page (row 14).
- `torch::optim::AdamW/Adam/SGD`: algorithmically correct starting point for
  the optimizer rewrite — extend, don't replace (row 9).

---

## 6. What this plan deliberately does not cover

- Inference-path/TorchScript/JIT export elimination — out of scope per the
  project brief.
- Mobile/edge deployment — out of scope.
- vLLM's rollout-generation path — separate project, not part of this
  PyTorch fork.
- `verl`'s own Python orchestration logic (advantage computation, reference
  log-prob bookkeeping) — this is framework code on top of PyTorch, not part
  of PyTorch itself, and isn't in this repository. It will need its own C++
  reimplementation as a separate, later project once the PyTorch-layer work
  above lands, but surveying it was out of scope here.
- Any actual rewrite code — this document is survey + plan only, per
  instructions.
