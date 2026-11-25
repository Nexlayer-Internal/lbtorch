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
