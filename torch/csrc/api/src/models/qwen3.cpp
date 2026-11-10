#include <torch/models/qwen3.h>

#include <torch/nn/functional/linear.h>
#include <torch/utils/safetensors.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <limits>
#include <set>
#include <sstream>

namespace torch::models::qwen3 {

// ============================== Qwen3Config ================================

Qwen3Config Qwen3Config::FromFile(const std::string& path) {
  std::ifstream f(path);
  TORCH_CHECK(f, "qwen3: could not open config '", path, "'");
  std::stringstream ss;
  ss << f.rdbuf();
  nlohmann::json j = nlohmann::json::parse(ss.str());

  Qwen3Config c;
  c.vocab_size = j.at("vocab_size").get<int64_t>();
  c.hidden_size = j.at("hidden_size").get<int64_t>();
  c.intermediate_size = j.at("intermediate_size").get<int64_t>();
  c.num_hidden_layers = j.at("num_hidden_layers").get<int64_t>();
  c.num_attention_heads = j.at("num_attention_heads").get<int64_t>();
  c.num_key_value_heads = j.at("num_key_value_heads").get<int64_t>();
  // Qwen3 sets head_dim explicitly and it does NOT generally equal
  // hidden_size / num_attention_heads (e.g. Qwen3-4B: hidden_size=2560,
  // num_attention_heads=32 -> 80, but head_dim is actually 128). Fall back
  // to the naive ratio only if the field is genuinely absent.
  if (j.contains("head_dim") && !j.at("head_dim").is_null()) {
    c.head_dim = j.at("head_dim").get<int64_t>();
  } else {
    c.head_dim = c.hidden_size / c.num_attention_heads;
  }
  c.rms_norm_eps = j.at("rms_norm_eps").get<double>();
  c.rope_theta = j.at("rope_theta").get<double>();
  c.attention_bias = j.value("attention_bias", false);
  c.tie_word_embeddings = j.value("tie_word_embeddings", false);
  c.max_position_embeddings = j.value("max_position_embeddings", int64_t{0});

  TORCH_CHECK(
      !(j.contains("rope_scaling") && !j.at("rope_scaling").is_null()),
      "qwen3: rope_scaling is set in config.json but this loader only implements "
      "the default (unscaled) RoPE formula -- verify against transformers' "
      "modeling_rope_utils.py before proceeding");

  return c;
}

// ================================ RMSNorm ===================================

RMSNormImpl::RMSNormImpl(int64_t dim, double eps) : eps_(eps) {
  weight = register_parameter("weight", torch::ones({dim}));
}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& x) {
  torch::Dtype input_dtype = x.scalar_type();
  torch::Tensor xf = x.to(torch::kFloat32);
  torch::Tensor variance = xf.pow(2).mean(-1, /*keepdim=*/true);
  xf = xf * torch::rsqrt(variance + eps_);
  return weight * xf.to(input_dtype);
}

// ================================== MLP =====================================

MLPImpl::MLPImpl(const Qwen3Config& cfg) {
  gate_proj = register_module(
      "gate_proj",
      torch::nn::Linear(
          torch::nn::LinearOptions(cfg.hidden_size, cfg.intermediate_size).bias(false)));
  up_proj = register_module(
      "up_proj",
      torch::nn::Linear(
          torch::nn::LinearOptions(cfg.hidden_size, cfg.intermediate_size).bias(false)));
  down_proj = register_module(
      "down_proj",
      torch::nn::Linear(
          torch::nn::LinearOptions(cfg.intermediate_size, cfg.hidden_size).bias(false)));
}

torch::Tensor MLPImpl::forward(const torch::Tensor& x) {
  return down_proj->forward(torch::silu(gate_proj->forward(x)) * up_proj->forward(x));
}

// ============================ RotaryEmbedding ===============================

RotaryEmbeddingImpl::RotaryEmbeddingImpl(int64_t head_dim, double theta) {
  torch::Tensor exponent =
      torch::arange(0, head_dim, 2, torch::kFloat32) / static_cast<double>(head_dim);
  inv_freq = register_buffer("inv_freq", torch::pow(theta, -exponent));
}

std::pair<torch::Tensor, torch::Tensor> RotaryEmbeddingImpl::forward(
    const torch::Tensor& positions) {
  int64_t b = positions.size(0);
  torch::Tensor inv_freq_expanded =
      inv_freq.unsqueeze(0).unsqueeze(-1).expand({b, inv_freq.size(0), 1}).to(torch::kFloat32);
  torch::Tensor positions_expanded = positions.unsqueeze(1).to(torch::kFloat32); // [B, 1, S]
  torch::Tensor freqs =
      torch::matmul(inv_freq_expanded, positions_expanded).transpose(1, 2); // [B, S, hd/2]
  torch::Tensor emb = torch::cat({freqs, freqs}, -1); // [B, S, hd]
  return {emb.cos(), emb.sin()};
}

// ================================ Attention =================================

namespace {

torch::Tensor RotateHalf(const torch::Tensor& t) {
  int64_t half = t.size(-1) / 2;
  torch::Tensor x1 = t.narrow(-1, 0, half);
  torch::Tensor x2 = t.narrow(-1, half, half);
  return torch::cat({-x2, x1}, -1);
}

// repeat_interleave along the head dimension: kv head i maps to query heads
// [i*n_rep, (i+1)*n_rep) -- matches transformers' repeat_kv exactly (NOT a
// tiled/"cat n_rep copies" order).
torch::Tensor RepeatKV(const torch::Tensor& t, int64_t n_rep) {
  if (n_rep == 1)
    return t;
  auto sizes = t.sizes();
  int64_t b = sizes[0], kvh = sizes[1], s = sizes[2], hd = sizes[3];
  torch::Tensor expanded = t.unsqueeze(2).expand({b, kvh, n_rep, s, hd});
  return expanded.reshape({b, kvh * n_rep, s, hd});
}

} // namespace

AttentionImpl::AttentionImpl(const Qwen3Config& cfg)
    : num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.num_key_value_heads),
      head_dim_(cfg.head_dim),
      n_rep_(cfg.num_attention_heads / cfg.num_key_value_heads),
      scaling_(1.0 / std::sqrt(static_cast<double>(cfg.head_dim))) {
  bool bias = cfg.attention_bias;
  q_proj = register_module(
      "q_proj",
      torch::nn::Linear(
          torch::nn::LinearOptions(cfg.hidden_size, num_heads_ * head_dim_).bias(bias)));
  k_proj = register_module(
      "k_proj",
      torch::nn::Linear(
          torch::nn::LinearOptions(cfg.hidden_size, num_kv_heads_ * head_dim_).bias(bias)));
  v_proj = register_module(
      "v_proj",
      torch::nn::Linear(
          torch::nn::LinearOptions(cfg.hidden_size, num_kv_heads_ * head_dim_).bias(bias)));
  o_proj = register_module(
      "o_proj",
      torch::nn::Linear(
          torch::nn::LinearOptions(num_heads_ * head_dim_, cfg.hidden_size).bias(bias)));
  q_norm = register_module("q_norm", RMSNorm(head_dim_, cfg.rms_norm_eps));
  k_norm = register_module("k_norm", RMSNorm(head_dim_, cfg.rms_norm_eps));
}

torch::Tensor AttentionImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& causal_mask) {
  int64_t b = x.size(0), s = x.size(1);

  torch::Tensor q = q_proj->forward(x).view({b, s, num_heads_, head_dim_});
  q = q_norm->forward(q).transpose(1, 2); // [B, heads, S, hd]

  torch::Tensor k = k_proj->forward(x).view({b, s, num_kv_heads_, head_dim_});
  k = k_norm->forward(k).transpose(1, 2); // [B, kv_heads, S, hd]

  torch::Tensor v = v_proj->forward(x).view({b, s, num_kv_heads_, head_dim_}).transpose(1, 2);

  torch::Tensor cos_b = cos.unsqueeze(1); // [B, 1, S, hd]
  torch::Tensor sin_b = sin.unsqueeze(1);
  torch::Tensor q_rot = q * cos_b + RotateHalf(q) * sin_b;
  torch::Tensor k_rot = k * cos_b + RotateHalf(k) * sin_b;

  torch::Tensor k_rep = RepeatKV(k_rot, n_rep_);
  torch::Tensor v_rep = RepeatKV(v, n_rep_);

  torch::Tensor attn_weights = torch::matmul(q_rot, k_rep.transpose(-2, -1)) * scaling_;
  attn_weights = attn_weights + causal_mask;
  attn_weights = torch::softmax(attn_weights, /*dim=*/-1);

  torch::Tensor attn_out = torch::matmul(attn_weights, v_rep); // [B, heads, S, hd]
  attn_out = attn_out.transpose(1, 2).contiguous().reshape({b, s, num_heads_ * head_dim_});
  return o_proj->forward(attn_out);
}

// ============================== DecoderLayer ================================

DecoderLayerImpl::DecoderLayerImpl(const Qwen3Config& cfg) {
  input_layernorm = register_module("input_layernorm", RMSNorm(cfg.hidden_size, cfg.rms_norm_eps));
  self_attn = register_module("self_attn", Attention(cfg));
  post_attention_layernorm =
      register_module("post_attention_layernorm", RMSNorm(cfg.hidden_size, cfg.rms_norm_eps));
  mlp = register_module("mlp", MLP(cfg));
}

torch::Tensor DecoderLayerImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& causal_mask) {
  torch::Tensor residual = x;
  torch::Tensor h = input_layernorm->forward(x);
  h = self_attn->forward(h, cos, sin, causal_mask);
  h = residual + h;

  residual = h;
  torch::Tensor h2 = post_attention_layernorm->forward(h);
  h2 = mlp->forward(h2);
  return residual + h2;
}

// ================================ Qwen3Model ================================

Qwen3ModelImpl::Qwen3ModelImpl(const Qwen3Config& cfg) : cfg_(cfg) {
  embed_tokens = register_module(
      "embed_tokens", torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg.vocab_size, cfg.hidden_size)));
  layers = register_module("layers", torch::nn::ModuleList());
  for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
    layers->push_back(DecoderLayer(cfg));
  }
  norm = register_module("norm", RMSNorm(cfg.hidden_size, cfg.rms_norm_eps));
  rotary_emb = register_module("rotary_emb", RotaryEmbedding(cfg.head_dim, cfg.rope_theta));
}

torch::Tensor Qwen3ModelImpl::forward(const torch::Tensor& input_ids) {
  int64_t b = input_ids.size(0), s = input_ids.size(1);
  torch::Tensor h = embed_tokens->forward(input_ids); // [B, S, hidden]

  torch::Tensor positions =
      torch::arange(0, s, torch::TensorOptions().dtype(torch::kInt64)).unsqueeze(0).expand({b, s});
  auto [cos, sin] = rotary_emb->forward(positions);

  torch::Tensor causal_mask = torch::zeros({s, s}, torch::kFloat32);
  torch::Tensor upper = torch::triu(torch::ones({s, s}, torch::kBool), /*diagonal=*/1);
  causal_mask.masked_fill_(upper, -std::numeric_limits<float>::infinity());
  causal_mask = causal_mask.unsqueeze(0).unsqueeze(0); // [1, 1, S, S]

  for (int64_t i = 0; i < cfg_.num_hidden_layers; ++i) {
    auto* layer = layers->ptr(i)->as<DecoderLayerImpl>();
    h = layer->forward(h, cos, sin, causal_mask);
  }
  return norm->forward(h);
}

// ============================= Qwen3ForCausalLM ==============================

Qwen3ForCausalLMImpl::Qwen3ForCausalLMImpl(const Qwen3Config& cfg) : cfg_(cfg) {
  model = register_module("model", Qwen3Model(cfg));
  if (!cfg.tie_word_embeddings) {
    lm_head = register_module(
        "lm_head",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.hidden_size, cfg.vocab_size).bias(false)));
  }
}

torch::Tensor Qwen3ForCausalLMImpl::forward(const torch::Tensor& input_ids) {
  torch::Tensor h = model->forward(input_ids);
  if (cfg_.tie_word_embeddings) {
    return torch::nn::functional::linear(h, model->embed_tokens->weight);
  }
  return lm_head->forward(h);
}

// ============================ LoadQwen3ForCausalLM ===========================

Qwen3ForCausalLM LoadQwen3ForCausalLM(const std::string& model_dir) {
  Qwen3Config cfg = Qwen3Config::FromFile(model_dir + "/config.json");
  Qwen3ForCausalLM module(cfg);

  torch::utils::safetensors::SafetensorsCollection weights =
      torch::utils::safetensors::SafetensorsCollection::FromDirectory(model_dir);

  torch::NoGradGuard no_grad;

  std::set<std::string> consumed;
  auto params = module->named_parameters(/*recurse=*/true);
  for (auto& item : params) {
    const std::string& name = item.key();
    torch::Tensor& param = item.value();

    TORCH_CHECK(
        weights.has(name),
        "qwen3: module parameter '",
        name,
        "' has no matching tensor in the checkpoint");
    torch::Tensor loaded = weights.get(name, torch::kFloat32);
    TORCH_CHECK(
        loaded.sizes() == param.sizes(),
        "qwen3: shape mismatch for '",
        name,
        "': module expects ",
        param.sizes(),
        " but checkpoint has ",
        loaded.sizes());
    param.copy_(loaded);
    consumed.insert(name);
  }

  // Every checkpoint tensor should have been consumed, except lm_head.weight
  // when embeddings are tied (it genuinely does not exist in the checkpoint
  // at all -- verified directly against the real Qwen3-4B safetensors index,
  // which has no lm_head.weight entry whatsoever when tie_word_embeddings is
  // true). Any other leftover name indicates the loader's naming assumptions
  // don't actually match this checkpoint, which is a bug, not a thing to
  // silently ignore.
  for (const std::string& name : weights.names()) {
    TORCH_CHECK(
        consumed.find(name) != consumed.end(),
        "qwen3: checkpoint tensor '",
        name,
        "' was never consumed by any module parameter");
  }

  module->eval();
  return module;
}

} // namespace torch::models::qwen3
