# LAND NOTES: land-lbtorch-nemotron-qwen-20261125

Base: gh/main 18cf32c2 (Nexlayer-Internal/lbtorch). Source: feature/nemotron-h-trainable (ee8d4b81925) which contains,
as a linear stack: feature/qwen35-training (2 commits: Qwen3 loader overlay + Qwen3.5 hybrid training model) ->
feature/nemotron-h-training (+2: model + CPU gradient gate) -> feature/nemotron-h-trainable (+2: block-level
trainer API, AttnFn hook). All three feature branches are ancestors of the tip. merge-base with main: 85522842
(= origin feature/gradient-checkpointing, already in main).

## Conflicts and side taken
- torch/csrc/api/include/torch/utils/safetensors.h and torch/csrc/api/src/utils/safetensors.cpp (add/add):
  main carries the first loader (commit 3748311afce); the feature branch carries the same file extended with
  FromFile, shape(), dtype(), and a device-direct get(). A diff of main's version against the feature's has NO
  removed lines (strict superset), so the branch side was taken (git checkout --ours on both).
- qwen3.h / qwen3.cpp / test qwen3.cpp are byte-identical on both sides (no conflict).
Everything else merged cleanly: main's additions (FSDP orchestration, collective_transport, adamw eager fuse,
nccl.cmake, caffe2/CMakeLists) are disjoint from the feature files. AI_POLICY.md deletion is main's.

## Scope
New model files only (torch::models::nemotron_h, qwen35) + safetensors extensions; no change to existing
ATen/autograd/optim code paths besides main's own.


---

# LAND NOTES: land-lbtorch-nemotron-slots-20261125

Base: gh/main dd51f8b2c0c. Source: local branch nemotron-g3 (79a1cf83b5f, 1 commit on feature/nemotron-h-trainable ee8d4b81925,
made by the GRPO agent, not pushed before): `nemotron_h: base_slots() lists the slots the loader recorded, not the defined
ones` (stable slot list across park / spill level 2, which empty the tensors; the trainer restores by slot index).
Merge: `git merge --no-ff` onto main, no conflicts (cec31a6437c). Diff against main: torch/models/nemotron_h.{h,cpp} only
(+21 / -8). A QWeight built by hand keeps the old behaviour (a defined tensor counts as present). Qwen sources untouched.

## Gates
- test/nemotron_h CPU gate (HF autograd oracle regenerated into a private fixture dir): 91 checks, 0 failed, 0 skipped.
- Through the trainer landing (lbtorch-train-app land-trainer-nemotron-20261125 compiled against this tree):
  test_nemotron_trainable 110/110, test_nemotron_wire_sites (base_slots stable across emptying every slot; park / restore
  emulation) PASS, queue jobs ld_n_nemo (nemotron_wire_check, resume_check, lbx_train_check through the Nemotron
  executable, bitwise) PASS and ld_n_qwen (Qwen lbx_train_check bitwise vs the pre-refactor trainer, attach, mb_fp32,
  spill, resume) PASS.

## Not included
perf-nemotron-20261124 (work in progress of another agent).
