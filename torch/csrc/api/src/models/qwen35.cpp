#include <torch/models/qwen35.h>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <sstream>

namespace torch::models::qwen35 {

namespace {
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
constexpr auto kF = torch::kFloat32;

// ---------------------------------------------------------------------------------------------------------------------
// fp32 view of a frozen bf16 weight: forward y = x.float() @ w.float()^T, backward dx = g.float() @ w.float(); saves ONLY the bf16 w.
// ---------------------------------------------------------------------------------------------------------------------
struct F32Linear : public torch::autograd::Function<F32Linear> {
  static Tensor forward(AutogradContext* ctx, const Tensor& x, const Tensor& w) {
    ctx->save_for_backward({w});
    return at::linear(x.to(kF), w.to(kF));
  }
  static variable_list backward(AutogradContext* ctx, variable_list go) {
    Tensor w = ctx->get_saved_variables()[0];
    return {at::matmul(go[0].to(kF), w.to(kF)), Tensor()};
  }
};

Tensor lin(const Tensor& x, const Linear& L, Compute c) {
  Tensor y = (c == Compute::FP32) ? F32Linear::apply(x, L.w) : at::linear(x, L.w);
  if (L.lora.defined()) {
    // PEFT semantics: adapter kept in fp32, input cast to fp32, delta added to the base result, result cast back to the base dtype.
    Tensor d = at::linear(at::linear(x.to(kF), L.lora.A), L.lora.B) * L.lora.scale;
    y = (c == Compute::FP32) ? (y + d) : (y + d).to(y.scalar_type());
  }
  return y;
}

// weights that HF keeps as bf16 params: in FP32 mode they are used as fp32 (exact upcast of the bf16 value)
Tensor P(const Tensor& t, Compute c) { return c == Compute::FP32 ? t.to(kF) : t; }

// Qwen3_5RMSNorm: x*rsqrt(mean(x^2)+eps) in fp32, times (1 + w), cast back to x dtype.
Tensor rms_offset(const Tensor& x, const Tensor& w, double eps) {
  Tensor xf = x.to(kF);
  Tensor out = xf * at::rsqrt(xf.pow(2).mean(-1, true) + eps) * (1.0 + w.to(kF));
  return out.to(x.scalar_type());
}

// Qwen3_5RMSNormGated: norm BEFORE gate; plain weight (not offset); gate = silu(gate.float()).
Tensor rms_gated(const Tensor& h, const Tensor& gate, const Tensor& w, double eps) {
  auto in_dt = h.scalar_type();
  Tensor hf = h.to(kF);
  hf = hf * at::rsqrt(hf.pow(2).mean(-1, true) + eps);
  Tensor o = w * hf.to(in_dt);
  o = o * at::silu(gate.to(kF));
  return o.to(in_dt);
}

struct Rope {
  Tensor cos, sin;  // [1,1,T,rotary_dim], in activation dtype
};

Rope make_rope(const Config& cfg, int64_t T, torch::Device dev, ScalarType dt) {
  const int64_t rd = cfg.rotary_dim();
  auto fopt = torch::TensorOptions().dtype(kF).device(dev);
  // inv_freq = 1 / base ** (arange(0, rd, 2) / rd)   (compute_default_rope_parameters); text-only => all 3 mrope rows are identical.
  Tensor ar = at::arange(0, rd, 2, fopt).div(static_cast<double>(rd));
  Tensor inv_freq = 1.0 / at::pow(at::scalar_tensor(cfg.rope_theta, fopt), ar);
  Tensor pos = at::arange(T, fopt);
  Tensor freqs = pos.unsqueeze(1) * inv_freq.unsqueeze(0);  // [T, rd/2]
  Tensor emb = at::cat({freqs, freqs}, -1);                 // [T, rd]: pairs i <-> i + rd/2
  return {emb.cos().to(dt).view({1, 1, T, rd}), emb.sin().to(dt).view({1, 1, T, rd})};
}

// apply_rotary_pos_emb: rotate_half over the FIRST rd channels (halves of rd/2), remaining channels pass through.
Tensor apply_rope(const Tensor& t, const Rope& r, int64_t rd) {
  const int64_t hd = t.size(-1);
  Tensor tr = t.narrow(-1, 0, rd), tp = t.narrow(-1, rd, hd - rd);
  Tensor x1 = tr.narrow(-1, 0, rd / 2), x2 = tr.narrow(-1, rd / 2, rd / 2);
  Tensor rh = at::cat({-x2, x1}, -1);
  Tensor e = tr * r.cos + rh * r.sin;
  return at::cat({e, tp}, -1);
}

Tensor attn_fwd(const Tensor& x, const Layer& L, const Rope& rope, const Config& cfg, Compute c) {
  const int64_t B = x.size(0), T = x.size(1), nH = cfg.num_attention_heads, nKV = cfg.num_key_value_heads, hd = cfg.head_dim;
  Tensor qg = lin(x, L.q_proj, c).view({B, T, nH, 2 * hd});
  auto parts = qg.chunk(2, -1);
  Tensor gate = parts[1].reshape({B, T, nH * hd});
  Tensor q = rms_offset(parts[0], L.q_norm, cfg.rms_norm_eps).transpose(1, 2);
  Tensor k = rms_offset(lin(x, L.k_proj, c).view({B, T, nKV, hd}), L.k_norm, cfg.rms_norm_eps).transpose(1, 2);
  Tensor v = lin(x, L.v_proj, c).view({B, T, nKV, hd}).transpose(1, 2);
  q = apply_rope(q, rope, cfg.rotary_dim());
  k = apply_rope(k, rope, cfg.rotary_dim());
  const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
  Tensor o;
  if (c == Compute::FP32) {
    // fp32 has no flash path; expand KV (repeat_interleave order: kv head i -> q heads [i*g,(i+1)*g)) and let the efficient/math kernel run.
    const int64_t g = nH / nKV;
    o = at::scaled_dot_product_attention(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), c10::nullopt, 0.0, true, scale, false);
  } else {
    o = at::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, scale, /*enable_gqa=*/true);  // flash, native GQA, hd256
  }
  o = o.transpose(1, 2).reshape({B, T, nH * hd}) * at::sigmoid(gate);
  return lin(o, L.o_proj, c);
}

Tensor l2norm(const Tensor& x) { return x * at::rsqrt((x * x).sum(-1, true) + 1e-6); }

// Pure-ATen chunked gated delta rule (port of HF torch_chunk_gated_delta_rule; the forward-substitution loop is kept verbatim). q,k,v [B,T,H,D], g [B,T,H] (fp32, log-decay <= 0), beta [B,T,H].
// Returns [B,T,H,Dv] fp32.
Tensor delta_rule_chunk(const Tensor& q_in, const Tensor& k_in, const Tensor& v_in, const Tensor& g_in, const Tensor& beta_in, int64_t C = 64) {
  Tensor q = l2norm(q_in.to(kF)), k = l2norm(k_in.to(kF));
  auto tr = [&](const Tensor& t) { return t.transpose(1, 2).contiguous().to(kF); };
  q = tr(q); k = tr(k);
  Tensor v = tr(v_in), beta = tr(beta_in), g = tr(g_in);  // [B,H,T,D] / [B,H,T]
  const int64_t B = k.size(0), H = k.size(1), T = k.size(2), Dk = k.size(3), Dv = v.size(3);
  const int64_t pad = (C - T % C) % C;
  if (pad) {
    q = at::pad(q, {0, 0, 0, pad}); k = at::pad(k, {0, 0, 0, pad}); v = at::pad(v, {0, 0, 0, pad});
    beta = at::pad(beta, {0, pad}); g = at::pad(g, {0, pad});
  }
  const int64_t Tp = T + pad, N = Tp / C;
  q = q * (1.0 / std::sqrt(static_cast<double>(Dk)));
  Tensor v_beta = v * beta.unsqueeze(-1), k_beta = k * beta.unsqueeze(-1);
  auto ch = [&](const Tensor& t) { return t.reshape({B, H, N, C, t.size(-1)}); };
  q = ch(q); k = ch(k); v = ch(v); k_beta = ch(k_beta); v_beta = ch(v_beta);
  g = g.reshape({B, H, N, C}).cumsum(-1);
  Tensor decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril();
  // attn = -(strictly-lower part of (k_beta k^T * decay)); forward substitution gives T = (I + M)^-1 (HF loop, verbatim semantics).
  // (at::linalg_solve_triangular would be the cheap equivalent but needs libtorch_cuda_linalg.so, which this prebuilt tree lacks.)
  Tensor attn0 = -(at::matmul(k_beta, k.transpose(-1, -2)) * decay).tril(-1);
  Tensor Tm = attn0.clone();
  for (int64_t i = 1; i < C; ++i) {
    Tensor row = Tm.select(-2, i).narrow(-1, 0, i).clone();
    Tensor sub = Tm.narrow(-2, 0, i).narrow(-1, 0, i).clone();
    Tm.select(-2, i).narrow(-1, 0, i).copy_(row + (row.unsqueeze(-1) * sub).sum(-2));
  }
  Tm = Tm + at::eye(C, Tm.options());
  Tensor value = at::matmul(Tm, v_beta);
  Tensor k_cumdecay = at::matmul(Tm, k_beta * g.exp().unsqueeze(-1));
  Tensor S = at::zeros({B, H, Dk, Dv}, q.options());
  std::vector<Tensor> outs;
  outs.reserve(N);
  for (int64_t i = 0; i < N; ++i) {
    Tensor q_i = q.select(2, i), k_i = k.select(2, i), v_i = value.select(2, i), g_i = g.select(2, i), dm_i = decay.select(2, i);
    Tensor attn = at::matmul(q_i, k_i.transpose(-1, -2)) * dm_i;
    Tensor v_new = v_i - at::matmul(k_cumdecay.select(2, i), S);
    Tensor attn_inter = at::matmul(q_i * g_i.exp().unsqueeze(-1), S);
    outs.push_back(attn_inter + at::matmul(attn, v_new));
    Tensor g_last = g_i.select(-1, C - 1);  // [B,H]
    S = S * g_last.exp().unsqueeze(-1).unsqueeze(-1) +
        at::matmul((k_i * (g_last.unsqueeze(-1) - g_i).exp().unsqueeze(-1)).transpose(-1, -2), v_new);
  }
  Tensor out = at::stack(outs, 2).reshape({B, H, Tp, Dv}).narrow(2, 0, T);  // [B,H,T,Dv]
  return out.transpose(1, 2).contiguous();
}

Tensor gdn_fwd(const Tensor& x, const Layer& L, const Config& cfg, Compute c) {
  const int64_t B = x.size(0), T = x.size(1), Hk = cfg.linear_num_key_heads, Hv = cfg.linear_num_value_heads, Dk = cfg.linear_key_head_dim,
                Dv = cfg.linear_value_head_dim, key_dim = Hk * Dk, value_dim = Hv * Dv, conv_dim = 2 * key_dim + value_dim;
  Tensor mixed = lin(x, L.in_proj_qkv, c).transpose(1, 2);  // [B,C,T]
  Tensor z = lin(x, L.in_proj_z, c).reshape({B, T, -1, Dv});
  Tensor b = lin(x, L.in_proj_b, c);
  Tensor a = lin(x, L.in_proj_a, c);
  // causal depthwise conv + silu
  mixed = at::silu(at::conv1d(mixed, P(L.conv_w, c), {}, {1}, {cfg.linear_conv_kernel_dim - 1}, {1}, conv_dim).narrow(2, 0, T));
  mixed = mixed.transpose(1, 2);
  Tensor q = mixed.narrow(-1, 0, key_dim).reshape({B, T, Hk, Dk});
  Tensor k = mixed.narrow(-1, key_dim, key_dim).reshape({B, T, Hk, Dk});
  Tensor v = mixed.narrow(-1, 2 * key_dim, value_dim).reshape({B, T, Hv, Dv});
  Tensor beta = b.sigmoid();
  Tensor g = -L.A_log.to(kF).exp() * at::softplus(a.to(kF) + L.dt_bias.to(kF));  // [B,T,Hv] fp32
  if (Hv / Hk > 1) {
    q = q.repeat_interleave(Hv / Hk, 2);
    k = k.repeat_interleave(Hv / Hk, 2);
  }
  Tensor core = delta_rule_chunk(q, k, v, g, beta).to(x.scalar_type());  // [B,T,Hv,Dv]
  core = core.reshape({-1, Dv});
  core = rms_gated(core, z.reshape({-1, Dv}), P(L.gdn_norm, c), cfg.rms_norm_eps).reshape({B, T, -1});
  return lin(core, L.out_proj, c);
}

Tensor layer_fwd(const Layer& L, const Tensor& x, const Rope& rope, const Config& cfg, Compute c) {
  Tensor h = rms_offset(x, L.input_norm, cfg.rms_norm_eps);
  h = L.full ? attn_fwd(h, L, rope, cfg, c) : gdn_fwd(h, L, cfg, c);
  Tensor r = x + h;
  Tensor m = rms_offset(r, L.post_norm, cfg.rms_norm_eps);
  m = lin(at::silu(lin(m, L.gate_proj, c)) * lin(m, L.up_proj, c), L.down_proj, c);
  return r + m;
}
}  // namespace

Config Config::FromFile(const std::string& path) {
  std::ifstream f(path);
  TORCH_CHECK(f, "qwen35: cannot open ", path);
  std::stringstream ss;
  ss << f.rdbuf();
  auto j = nlohmann::json::parse(ss.str());
  const auto& t = j.contains("text_config") ? j.at("text_config") : j;
  Config c;
  c.vocab_size = t.at("vocab_size"); c.hidden_size = t.at("hidden_size"); c.intermediate_size = t.at("intermediate_size");
  c.num_hidden_layers = t.at("num_hidden_layers"); c.num_attention_heads = t.at("num_attention_heads");
  c.num_key_value_heads = t.at("num_key_value_heads"); c.head_dim = t.at("head_dim"); c.rms_norm_eps = t.at("rms_norm_eps");
  const auto& rp = t.at("rope_parameters");
  c.rope_theta = rp.at("rope_theta");
  c.partial_rotary_factor = rp.value("partial_rotary_factor", 1.0);
  c.linear_num_key_heads = t.at("linear_num_key_heads"); c.linear_num_value_heads = t.at("linear_num_value_heads");
  c.linear_key_head_dim = t.at("linear_key_head_dim"); c.linear_value_head_dim = t.at("linear_value_head_dim");
  c.linear_conv_kernel_dim = t.at("linear_conv_kernel_dim");
  c.tie_word_embeddings = t.value("tie_word_embeddings", j.value("tie_word_embeddings", false));
  TORCH_CHECK(t.value("attn_output_gate", true), "qwen35: only attn_output_gate=true is implemented");
  for (const auto& s : t.at("layer_types")) c.is_full_attention.push_back(s.get<std::string>() == "full_attention");
  TORCH_CHECK((int64_t)c.is_full_attention.size() == c.num_hidden_layers, "qwen35: layer_types size mismatch");
  return c;
}

Model Model::Load(const std::string& dir, torch::Device device) {
  Model m;
  m.cfg = Config::FromFile(dir + "/config.json");
  m.device = device;
  auto coll = utils::safetensors::SafetensorsCollection::FromDirectory(dir);
  const std::string pre = coll.has("model.language_model.embed_tokens.weight") ? "model.language_model." : "model.";
  auto W = [&](const std::string& n) { return coll.get(n, torch::kBFloat16, device); };
  m.embed = W(pre + "embed_tokens.weight");
  m.final_norm = W(pre + "norm.weight");
  m.lm_head = m.cfg.tie_word_embeddings ? m.embed : W("lm_head.weight");
  m.layers.resize(m.cfg.num_hidden_layers);
  for (int64_t i = 0; i < m.cfg.num_hidden_layers; ++i) {
    Layer& L = m.layers[i];
    const std::string p = pre + "layers." + std::to_string(i) + ".";
    const std::string nm = "layers." + std::to_string(i) + ".";
    L.full = m.cfg.is_full_attention[i];
    L.input_norm = W(p + "input_layernorm.weight");
    L.post_norm = W(p + "post_attention_layernorm.weight");
    auto mk = [&](Linear& l, const std::string& sub) { l.w = W(p + sub + ".weight"); l.name = nm + sub; };
    mk(L.gate_proj, "mlp.gate_proj"); mk(L.up_proj, "mlp.up_proj"); mk(L.down_proj, "mlp.down_proj");
    if (L.full) {
      mk(L.q_proj, "self_attn.q_proj"); mk(L.k_proj, "self_attn.k_proj"); mk(L.v_proj, "self_attn.v_proj"); mk(L.o_proj, "self_attn.o_proj");
      L.q_norm = W(p + "self_attn.q_norm.weight"); L.k_norm = W(p + "self_attn.k_norm.weight");
    } else {
      mk(L.in_proj_qkv, "linear_attn.in_proj_qkv"); mk(L.in_proj_z, "linear_attn.in_proj_z"); mk(L.in_proj_b, "linear_attn.in_proj_b");
      mk(L.in_proj_a, "linear_attn.in_proj_a"); mk(L.out_proj, "linear_attn.out_proj");
      L.conv_w = W(p + "linear_attn.conv1d.weight"); L.A_log = W(p + "linear_attn.A_log"); L.dt_bias = W(p + "linear_attn.dt_bias");
      L.gdn_norm = W(p + "linear_attn.norm.weight");
    }
  }
  return m;
}

std::vector<Linear*> Model::all_linears() {
  std::vector<Linear*> v;
  for (auto& L : layers) {
    if (L.full) v.insert(v.end(), {&L.q_proj, &L.k_proj, &L.v_proj, &L.o_proj});
    else v.insert(v.end(), {&L.in_proj_qkv, &L.in_proj_z, &L.in_proj_b, &L.in_proj_a, &L.out_proj});
    v.insert(v.end(), {&L.gate_proj, &L.up_proj, &L.down_proj});
  }
  return v;
}

void Model::attach_lora(int64_t rank, double alpha, uint64_t seed, const utils::safetensors::SafetensorsCollection* init) {
  auto gen = at::detail::createCPUGenerator(seed);
  for (Linear* l : all_linears()) {
    const int64_t out = l->w.size(0), in = l->w.size(1);
    Tensor A, B;
    if (init && init->has("lora_init." + l->name + ".A")) {
      A = init->get("lora_init." + l->name + ".A", kF, device);
      B = init->get("lora_init." + l->name + ".B", kF, device);
    } else {
      const double bound = 1.0 / std::sqrt(static_cast<double>(in));
      A = (at::rand({rank, in}, gen, torch::TensorOptions().dtype(kF)) * 2 - 1).mul(bound).to(device);
      B = at::zeros({out, rank}, torch::TensorOptions().dtype(kF).device(device));
    }
    TORCH_CHECK(A.size(0) == rank && A.size(1) == in && B.size(0) == out && B.size(1) == rank, "qwen35: bad LoRA shape for ", l->name);
    l->lora.A = A.contiguous().detach().requires_grad_(true);
    l->lora.B = B.contiguous().detach().requires_grad_(true);
    l->lora.scale = alpha / static_cast<double>(rank);
  }
}

std::vector<std::pair<std::string, Tensor>> Model::named_lora_params() {
  std::vector<std::pair<std::string, Tensor>> v;
  for (Linear* l : all_linears())
    if (l->lora.defined()) { v.emplace_back(l->name + ".A", l->lora.A); v.emplace_back(l->name + ".B", l->lora.B); }
  return v;
}

Tensor Model::forward(const Tensor& ids, const ForwardOptions& opt) {
  const Compute c = opt.compute;
  const int64_t T = ids.size(1);
  const ScalarType adt = (c == Compute::FP32) ? kF : torch::kBFloat16;
  Probe* pr = opt.probe;
  Tensor h = at::embedding(embed, ids).to(adt);
  if (pr && pr->acts) pr->act["emb"] = h.detach().to(kF).cpu();
  // Reentrant checkpointing needs a grad-requiring input; the frozen embedding output is not one (also gives dL/d(emb) for free).
  if (opt.checkpoint) h = h.detach().requires_grad_(true);
  Rope rope = make_rope(cfg, T, ids.device(), adt);
  for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
    const Layer* Lp = &layers[i];
    const Config cf = cfg;
    if (opt.checkpoint) {
      h = torch::utils::checkpoint(
              [Lp, rope, cf, c](std::vector<Tensor> in) { return std::vector<Tensor>{layer_fwd(*Lp, in[0], rope, cf, c)}; }, {h}, false)[0];
    } else {
      h = layer_fwd(*Lp, h, rope, cf, c);
    }
    if (pr && pr->acts) pr->act["layer." + std::to_string(i)] = h.detach().to(kF).cpu();
    if (pr && pr->grads && h.requires_grad())
      h.register_hook([pr, i](const Tensor& g) { pr->grad["layer." + std::to_string(i)] = g.detach().to(kF).cpu(); });
  }
  h = rms_offset(h, final_norm, cfg.rms_norm_eps);
  if (pr && pr->acts) pr->act["final_norm"] = h.detach().to(kF).cpu();
  Linear head;
  head.w = lm_head;
  return lin(h, head, c);
}

}  // namespace torch::models::qwen35
