#pragma once

/// Qwen3.5 ("qwen3_5" hybrid: gated-delta-net linear attention + gated full attention) decoder for TRAINING, in C++ ATen + autograd.
/// No Python anywhere on this path. The transformers `modeling_qwen3_5.py` is the spec; every deviation is noted in qwen35.cpp.
///
/// Compute modes:
///   BF16: activations bf16, fp32 where HF does (RMSNorm, gates, the delta rule, LoRA branch) -- the production-like mode.
///   FP32: bf16 weights are kept resident and upcast per use inside a custom autograd Function that saves ONLY the bf16 weight
///         (so memory stays ~bf16 weights + activations); all activations fp32. This is the mode used to validate logic against the
///         HF fp32 oracle at tight tolerance.
/// GDN here is the pure-ATen reference (conv1d + chunked gated delta rule, autograd through ATen ops); a fast kernel replaces it later.

#include <torch/csrc/Export.h>
#include <torch/types.h>
#include <torch/utils/safetensors.h>

#include <map>
#include <string>
#include <vector>

namespace torch::models::qwen35 {

struct TORCH_API Config {
  int64_t vocab_size = 0, hidden_size = 0, intermediate_size = 0, num_hidden_layers = 0;
  int64_t num_attention_heads = 0, num_key_value_heads = 0, head_dim = 0;
  double rms_norm_eps = 1e-6, rope_theta = 1e7, partial_rotary_factor = 0.25;
  int64_t linear_num_key_heads = 0, linear_num_value_heads = 0, linear_key_head_dim = 0, linear_value_head_dim = 0, linear_conv_kernel_dim = 4;
  bool tie_word_embeddings = false;
  std::vector<bool> is_full_attention;  // per layer
  int64_t rotary_dim() const { return static_cast<int64_t>(head_dim * partial_rotary_factor); }
  /// Accepts the multimodal wrapper config.json (reads "text_config") or a bare text config.
  static Config FromFile(const std::string& config_json_path);
};

enum class Compute { BF16, FP32 };

struct TORCH_API Lora {
  Tensor A, B;   // fp32 leaves, A [r,in], B [out,r]
  double scale = 0;
  bool defined() const { return A.defined(); }
};

struct TORCH_API Linear {
  Tensor w;  // bf16 [out,in], frozen
  Lora lora;
  std::string name;  // e.g. layers.3.self_attn.q_proj
};

struct TORCH_API Layer {
  bool full = false;
  Tensor input_norm, post_norm;  // [hidden] (offset-weight RMSNorm: scale = 1 + w)
  Linear gate_proj, up_proj, down_proj;
  // full attention
  Linear q_proj, k_proj, v_proj, o_proj;
  Tensor q_norm, k_norm;  // [head_dim], offset-weight
  // gated delta net
  Linear in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj;
  Tensor conv_w;          // [conv_dim,1,k]
  Tensor A_log, dt_bias;  // [num_v_heads]
  Tensor gdn_norm;        // [head_v_dim], plain weight (NOT offset)
};

/// Optional recorder: layer outputs (detached fp32 CPU) and dL/d(layer output) (fp32 CPU, filled during backward).
struct Probe {
  bool acts = true, grads = true;
  std::map<std::string, Tensor> act, grad;
};

struct TORCH_API ForwardOptions {
  Compute compute = Compute::BF16;
  bool checkpoint = true;  // per-layer reentrant gradient checkpointing (torch::utils::checkpoint)
  Probe* probe = nullptr;
};

class TORCH_API Model {
 public:
  Config cfg;
  Tensor embed, final_norm, lm_head;  // bf16
  std::vector<Layer> layers;
  torch::Device device = torch::kCUDA;

  /// Loads every language-model tensor (bf16) straight to `device`. Visual/MTP tensors are ignored.
  static Model Load(const std::string& dir, torch::Device device);

  /// Attach LoRA (fp32 leaves) to every language-model Linear except lm_head ("all-linear"): per layer q,k,v,o | in_proj_qkv,z,b,a,out_proj
  /// plus gate,up,down. Default init: A ~ U(-1/sqrt(in),1/sqrt(in)) (PEFT kaiming(a=sqrt5)), B = 0. `init` (optional) overrides from
  /// safetensors entries "lora_init.<name>.A/.B".
  void attach_lora(int64_t rank, double alpha, uint64_t seed = 0, const utils::safetensors::SafetensorsCollection* init = nullptr);
  std::vector<std::pair<std::string, Tensor>> named_lora_params();  // "<linear name>.A" / ".B"

  /// ids [B,T] int64 on device -> logits [B,T,V] in compute dtype.
  Tensor forward(const Tensor& ids, const ForwardOptions& opt = {});

 private:
  std::vector<Linear*> all_linears();
};

}  // namespace torch::models::qwen35
