#pragma once

/// Qwen3 decoder-only transformer (config parsing + architecture + real
/// checkpoint loading), built directly on libtorch's C++ `torch::nn::Module`
/// API -- no Python anywhere in this path. Part of PLAN.md's Python-
/// elimination effort: the C++ equivalent of
/// `transformers.AutoModelForCausalLM.from_pretrained(qwen3_dir)`, which
/// normally reads a HuggingFace `config.json` + `safetensors` weights in
/// Python and builds the module graph there.
///
/// Verified against the real, published `Qwen/Qwen3-4B` config.json and the
/// real `transformers` 4.51.0 `modeling_qwen3.py` source (not assumed) --
/// see LoadQwen3ForCausalLM's doc comment and docs/qwen3_cpp_loading.md for
/// the full verification writeup and numerical validation results against
/// Python's `transformers` on the same checkpoint.
///
/// Architecture highlights that do NOT generalize from more familiar
/// (Llama-style) decoder-only transformers, verified rather than assumed:
///   - `head_dim` is an explicit config field, NOT hidden_size /
///     num_attention_heads (Qwen3-4B: hidden_size=2560, num_attention_heads=32
///     -> 80, but head_dim is actually 128).
///   - Attention applies a per-head RMSNorm (q_norm/k_norm, weight shape
///     [head_dim]) to Q/K *before* RoPE, on the [B,S,heads,head_dim] layout
///     (i.e. before transposing heads next to the sequence dim), so one
///     shared [head_dim] weight normalizes each head independently.
///   - RMSNorm computes in float32 regardless of input dtype, and multiplies
///     by the weight only after casting back to the input dtype.
///   - RoPE: inv_freq = 1/theta^(arange(0,head_dim,2)/head_dim); cos/sin are
///     built via cat({freqs, freqs}, -1) (NOT pairwise-interleaved);
///     rotate_half splits head_dim in half (not pairwise either).
///   - GQA: key/value heads are repeated via unsqueeze+expand+reshape
///     ("repeat_interleave" order -- kv head i maps to query heads
///     [i*n_rep, (i+1)*n_rep) -- not a tiled/concatenated order).
///   - Tied embeddings: Qwen3-4B's config.json has tie_word_embeddings=true,
///     confirmed directly against the real safetensors index containing no
///     `lm_head.weight` tensor at all -- the LM head reuses embed_tokens'
///     weight directly rather than owning a separate parameter.

#include <torch/csrc/Export.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>
#include <string>
#include <utility>

namespace torch::models::qwen3 {

struct TORCH_API Qwen3Config {
  int64_t vocab_size = 0;
  int64_t hidden_size = 0;
  int64_t intermediate_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0; // explicit in config; NOT hidden_size / num_attention_heads
  double rms_norm_eps = 1e-6;
  double rope_theta = 10000.0;
  bool attention_bias = false;
  bool tie_word_embeddings = false;
  int64_t max_position_embeddings = 0;

  static Qwen3Config FromFile(const std::string& config_json_path);
};

/// RMSNorm (not LayerNorm): x * rsqrt(mean(x^2, -1) + eps) * weight,
/// computed in float32 regardless of input dtype.
class TORCH_API RMSNormImpl : public torch::nn::Module {
 public:
  RMSNormImpl(int64_t dim, double eps);
  torch::Tensor forward(const torch::Tensor& x);

  torch::Tensor weight;
  double eps_;
};
TORCH_MODULE(RMSNorm);

class TORCH_API MLPImpl : public torch::nn::Module {
 public:
  explicit MLPImpl(const Qwen3Config& cfg);
  torch::Tensor forward(const torch::Tensor& x);

  torch::nn::Linear gate_proj{nullptr}, up_proj{nullptr}, down_proj{nullptr};
};
TORCH_MODULE(MLP);

/// Precomputes RoPE cos/sin tables for a batch of positions. Stateless aside
/// from the (non-trained) inv_freq buffer.
class TORCH_API RotaryEmbeddingImpl : public torch::nn::Module {
 public:
  RotaryEmbeddingImpl(int64_t head_dim, double theta);

  // positions: int64 tensor of shape [batch, seq_len]. Returns {cos, sin},
  // each [batch, seq_len, head_dim], float32.
  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& positions);

  torch::Tensor inv_freq; // [head_dim / 2], float32, registered as a buffer
};
TORCH_MODULE(RotaryEmbedding);

class TORCH_API AttentionImpl : public torch::nn::Module {
 public:
  explicit AttentionImpl(const Qwen3Config& cfg);

  // x: [B, S, hidden_size]; cos/sin: [B, S, head_dim]; causal_mask:
  // [1, 1, S, S] additive mask (0 or -inf).
  torch::Tensor forward(
      const torch::Tensor& x,
      const torch::Tensor& cos,
      const torch::Tensor& sin,
      const torch::Tensor& causal_mask);

  torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, o_proj{nullptr};
  RMSNorm q_norm{nullptr}, k_norm{nullptr};

  int64_t num_heads_, num_kv_heads_, head_dim_, n_rep_;
  double scaling_;
};
TORCH_MODULE(Attention);

class TORCH_API DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit DecoderLayerImpl(const Qwen3Config& cfg);

  torch::Tensor forward(
      const torch::Tensor& x,
      const torch::Tensor& cos,
      const torch::Tensor& sin,
      const torch::Tensor& causal_mask);

  RMSNorm input_layernorm{nullptr}, post_attention_layernorm{nullptr};
  Attention self_attn{nullptr};
  MLP mlp{nullptr};
};
TORCH_MODULE(DecoderLayer);

/// Corresponds to HF's `Qwen3Model` (the "model." prefix in tensor names):
/// embed_tokens + decoder stack + final norm. Does not include the LM head.
class TORCH_API Qwen3ModelImpl : public torch::nn::Module {
 public:
  explicit Qwen3ModelImpl(const Qwen3Config& cfg);

  // input_ids: [B, S] int64. Returns final hidden states [B, S, hidden_size].
  torch::Tensor forward(const torch::Tensor& input_ids);

  Qwen3Config cfg_;
  torch::nn::Embedding embed_tokens{nullptr};
  torch::nn::ModuleList layers;
  RMSNorm norm{nullptr};
  RotaryEmbedding rotary_emb{nullptr};
};
TORCH_MODULE(Qwen3Model);

/// Corresponds to HF's `Qwen3ForCausalLM`: adds the LM head on top of
/// Qwen3Model. If cfg.tie_word_embeddings, no separate lm_head parameter is
/// registered -- forward() reuses model->embed_tokens->weight directly, so
/// there is nothing for LoadQwen3ForCausalLM to (mis)match against a
/// nonexistent `lm_head.weight` safetensors entry.
class TORCH_API Qwen3ForCausalLMImpl : public torch::nn::Module {
 public:
  explicit Qwen3ForCausalLMImpl(const Qwen3Config& cfg);

  // Returns logits [B, S, vocab_size].
  torch::Tensor forward(const torch::Tensor& input_ids);

  Qwen3Config cfg_;
  Qwen3Model model{nullptr};
  torch::nn::Linear lm_head{nullptr}; // null iff cfg_.tie_word_embeddings
};
TORCH_MODULE(Qwen3ForCausalLM);

/// Loads a real Qwen3 checkpoint directly in C++: `model_dir` must contain
/// `config.json` plus either `model.safetensors` or
/// `model.safetensors.index.json` + its shards (standard HuggingFace
/// repo/cache layout). All parameters are loaded as float32 (weights are
/// upcast from their on-disk dtype, typically bf16, during the copy).
///
/// Throws (TORCH_CHECK) on any mismatch between the module's expected
/// parameters and the checkpoint's actual tensors (missing tensor,
/// unexpected leftover tensor, or shape mismatch) -- loading never silently
/// proceeds with a partially- or mis-initialized model.
TORCH_API Qwen3ForCausalLM LoadQwen3ForCausalLM(const std::string& model_dir);

} // namespace torch::models::qwen3
