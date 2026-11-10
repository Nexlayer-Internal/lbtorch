# Loading a real Qwen3-4B checkpoint in pure C++ (no Python)

**Status:** implemented and validated against Python's `transformers` on real
hardware (IP66, 2x RTX PRO 6000 Blackwell). Independent of the FSDP
orchestration work in `FSDP-DESIGN.md` -- this covers the model-construction
and checkpoint-loading step only (PLAN.md's survey didn't flag this
specifically, since it assumed a model object already exists in memory by
the time training starts; this closes that gap for a Python-free path).

## What this adds

Three new pieces, consistent with the rest of this fork's extend-don't-
duplicate approach to libtorch's C++ API:

- `torch/csrc/api/include/torch/utils/safetensors.h` +
  `torch/csrc/api/src/utils/safetensors.cpp` --
  `torch::utils::safetensors::SafetensorsCollection`, a from-scratch reader
  for the `safetensors` checkpoint format (8-byte LE header length + JSON
  header + raw tensor bytes) and HuggingFace's sharded-checkpoint index
  convention (`model.safetensors.index.json`). No third-party safetensors
  library is linked -- the header is parsed with `nlohmann::json` (already
  vendored and linked into `torch_cpu` for other purposes, e.g.
  `ProcessGroupNCCL.cpp`'s flight recorder and
  `torch/csrc/inductor/aoti_package/model_package_loader.cpp`; see
  `caffe2/CMakeLists.txt`), and tensor bytes are mapped via `torch::from_blob`
  with a deleter that captures a `shared_ptr` to the backing mmap, so the
  mapping outlives any view created from it.
- `torch/csrc/api/include/torch/models/qwen3.h` +
  `torch/csrc/api/src/models/qwen3.cpp` -- the actual Qwen3 decoder-only
  transformer (`torch::models::qwen3::Qwen3ForCausalLM`), built on
  `torch::nn::Module`/`Linear`/`Embedding`/`ModuleList` (PLAN.md row 6's
  "free win"), plus `Qwen3Config::FromFile()` (parses `config.json`, also via
  `nlohmann::json`) and `LoadQwen3ForCausalLM(model_dir)`, the top-level
  entry point.
- `test/cpp/api/qwen3.cpp` -- GTest cases for the parts that are easy to get
  subtly wrong in isolation (safetensors round-trip, RMSNorm math, RoPE
  angle closed-form check, GQA `repeat_kv` head-mapping order, config
  parsing against a bundled real config, and a small random-weight
  build-and-run smoke test). Registered in `test/cpp/api/CMakeLists.txt`.
  Deliberately does NOT load the real 4B-parameter checkpoint (an ~8GB
  download, unsuitable for CI) -- that validation is below, run by hand on
  IP66 against the real cached checkpoint.

## API

```cpp
#include <torch/models/qwen3.h>

using namespace torch::models::qwen3;

// model_dir must contain config.json plus either model.safetensors or
// model.safetensors.index.json + its shards (standard HF repo/cache layout).
Qwen3ForCausalLM model = LoadQwen3ForCausalLM(model_dir);
model->eval();

torch::Tensor input_ids = ...;        // [batch, seq_len], int64
torch::Tensor logits;
{
  torch::NoGradGuard no_grad;
  logits = model->forward(input_ids); // [batch, seq_len, vocab_size]
}
```

`LoadQwen3ForCausalLM` throws (`TORCH_CHECK`) on any mismatch between the
module's parameters and the checkpoint's tensors -- a missing tensor, an
unconsumed checkpoint tensor, or a shape mismatch -- rather than silently
loading a partially-initialized model.

## Architecture details verified against the real checkpoint and the real
## `transformers` source (not assumed)

Checked directly against the real, published `Qwen/Qwen3-4B` `config.json`
and tensor names (via its `model.safetensors.index.json`), and against
`transformers` 4.51.0's actual `modeling_qwen3.py` (the version the
checkpoint's `config.json` itself declares):

- **`head_dim` is explicit, not derived.** Qwen3-4B: `hidden_size=2560`,
  `num_attention_heads=32` -> a naive `hidden_size / num_attention_heads`
  would give 80, but the real `head_dim` is 128 (so
  `q_proj`/`k_proj`/`v_proj`'s output width is `heads * head_dim`, not
  `hidden_size`). Confirmed both from `config.json`'s explicit `head_dim`
  field and from the real `q_proj.weight` tensor shape (`[4096, 2560]` =
  `[32*128, 2560]`).
- **Tied embeddings.** `config.json` has `tie_word_embeddings: true`,
  confirmed directly against the real safetensors index: there is **no**
  `lm_head.weight` tensor in the checkpoint at all (398 tensors total, none
  named `lm_head.weight`). `Qwen3ForCausalLMImpl` does not register a
  separate `lm_head` parameter in this case -- `forward()` reuses
  `model->embed_tokens->weight` directly via
  `torch::nn::functional::linear`.
- **Per-head QK-norm.** Qwen3 (unlike Qwen2/Llama) applies an RMSNorm
  (`q_norm`/`k_norm`, weight shape `[head_dim]` = `[128]`) to Q/K *before*
  RoPE, on the `[B,S,heads,head_dim]` layout (before transposing heads next
  to the sequence dim) -- confirmed from both the real tensor names
  (`self_attn.q_norm.weight`/`k_norm.weight`) and the real
  `Qwen3Attention.forward` source.
- **RoPE.** `inv_freq = 1 / theta^(arange(0,head_dim,2)/head_dim)`,
  `theta = rope_theta = 1,000,000` (not the common 10,000 default);
  `rope_scaling` is `null` for Qwen3-4B (plain/unscaled RoPE). `cos`/`sin`
  are built via `cat({freqs, freqs}, -1)` (not pairwise-interleaved), and
  `rotate_half` splits `head_dim` into two contiguous halves (also not
  pairwise). `Qwen3Config::FromFile` explicitly rejects a non-null
  `rope_scaling` rather than silently assuming the default formula still
  applies.
- **GQA head mapping.** `num_attention_heads=32`, `num_key_value_heads=8` ->
  `n_rep=4`. `repeat_kv`'s expand+reshape order means kv head `i` maps to
  query heads `[i*4, (i+1)*4)` (a "repeat_interleave" order), not an
  interleaved or tiled order -- verified both against the real
  `transformers` source and with a standalone unit test
  (`RepeatKvUsesRepeatInterleaveOrderNotTiled`) that checks the mapping
  directly rather than only end-to-end.
- **No biases anywhere.** `attention_bias: false`, and the real safetensors
  index has zero tensors with `bias` in the name across all 398 tensors --
  confirmed, not assumed, before writing `Linear(..., bias=false)`
  everywhere.
- **RMSNorm, not LayerNorm**, computed in float32 regardless of input dtype,
  weight multiply applied after casting back to the input dtype (matters for
  mixed-precision training; irrelevant for this validation since both sides
  run entirely in float32 -- see below).

## Numerical validation

Procedure: both sides load the real, already-cached `Qwen/Qwen3-4B`
checkpoint (`/mnt/raid/grpo-smoketest/hf_cache/hub/models--Qwen--Qwen3-4B`,
~8GB, reused read-only rather than re-downloaded) in **float32** (weights
upcast from their on-disk bf16), run an identical forward pass over the same
fixed token ids (no sampling, no dropout, `model.eval()` /
`torch.no_grad()`), and compare logits directly:

- C++ side: `forward_test` (standalone tool, not part of the CMake-built
  `test_api` target -- see "Build note" below), loads via
  `LoadQwen3ForCausalLM`, writes each test case's logits to a raw float32
  `.bin` file.
- Python side: `scripts/run_python_reference.py` (Python/`transformers`,
  `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32,
  attn_implementation="eager")` -- eager attention specifically, so both
  sides compute attention the same (mathematically equivalent but not
  necessarily bit-identical) way rather than comparing against SDPA/
  flash-attention's different accumulation order) runs the same cases and
  diffs against the C++ `.bin` files.

Test cases (`scripts/gen_test_cases.py`, deterministic/seeded, shared
verbatim between both sides so token ids are byte-identical):

| case | batch | seq_len | max abs diff | mean abs diff | argmax match |
|---|---|---|---|---|---|
| tiny_b1_s3 | 1 | 3 | 3.05e-05 | 2.47e-06 | yes (3/3 positions) |
| short_b1_s16 | 1 | 16 | 4.10e-05 | 3.74e-06 | yes (16/16) |
| medium_b1_s128 | 1 | 128 | 2.38e-04 | 4.92e-06 | yes (128/128) |
| long_b1_s512 | 1 | 512 | 1.27e-04 | 3.02e-06 | yes (512/512) |
| batch4_s32 | 4 | 32 | 1.70e-04 | 3.86e-06 | yes (4x32/4x32) |
| batch3_s64 | 3 | 64 | 7.15e-05 | 3.43e-06 | yes (3x64/3x64) |

(max-relative-diff is not reported here: with logits whose true value is
near zero, a tiny absolute difference produces a huge and meaningless
relative ratio -- it was computed during validation and confirmed to track
exactly the near-zero-logit positions, not a real divergence; the absolute
diff and the 100% argmax match across every single position, in every case,
are the numbers that actually matter.)

**Result: max abs diff across all 6 cases (including the 512-token
long-sequence case, batch size up to 4) is 2.38e-4, mean abs diff is
~3-5e-6, and the argmax token matches Python at literally every position in
every case.** For float32 accumulation through 36 decoder layers of a
4-billion-parameter model (different summation order between my hand-
written eager attention/MLP and `transformers`' own eager implementation,
not a difference in dtype or algorithm), this is exactly the "extremely
close, not just roughly similar" bar the task called for -- these are not
hand-picked best cases; all 6 generated cases passed, including the one
specifically sized to exercise RoPE across a wide range of relative
positions.

Full timing (CPU-only, single process, this build's intraop threading):
loading the checkpoint took 29.8s; forward passes ranged from 12.5s
(3 tokens) to 966.6s (512 tokens, the quadratic-in-sequence-length eager
attention cost dominating); total wall time for all 6 cases plus loading was
~30 minutes. No GPU path was exercised in this validation (CPU was used for
exact, deterministic comparability with the Python reference); this is a
correctness validation, not a performance benchmark, and no throughput
optimization work (e.g. SDPA/flash-attention-equivalent fused kernels) has
been done here.

## Build note (standalone validation harness, same caveat as the
## optimizer/checkpointing PRs)

Like the AdamW foreach/fused/capturable work, this was validated against an
**already-built** `libtorch_cpu.so`/`libc10.so` (from
`/mnt/raid/lbtorch-checkpoint-build`, the gradient-checkpointing PR's build
output) rather than a fresh `pip install -e . -v --no-build-isolation` of
this exact commit -- a full rebuild from this commit wasn't feasible in this
session's time budget. The three new files were compiled directly against
that prebuilt library's headers/ABI (same compiler, same flags, extracted
from that build's own `compile_commands.json`/`build.ninja`) in both their
final repo locations and namespaces, including linking and running the real
`qwen3.cpp` GTest cases against the real `libgtest`. `caffe2/CMakeLists.txt`
and `test/cpp/api/CMakeLists.txt` are updated so a full rebuild picks them up
normally; that full-rebuild path itself has not been executed end-to-end in
this session.
