#include <torch/models/nemotron_h.h>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

// Deviations from transformers 5.13.1 modeling_nemotron_h.py (everything else is a 1:1 op translation, so autograd supplies the backward):
//  1. SSD state passing: HF's torch_forward sums `decay_chunk[..., None, None] * states_permuted[:, :, None, ...]` over dim=2 (the DESTINATION chunk); the
//     recurrence needs the SOURCE chunk (dim=3). ssd_aten uses dim=3. (The checkpoints' own modeling_nemotron_h.py has the same dim=2 line.) Verified against
//     ssd_serial() and, in the fixture generator, against HF with the one-line fix.
//  2. Prefix hand-off: HF takes cache_params; here LayerState carries (ssm, conv left context, K/V) explicitly and graph-connected, so a response can be
//     trained on top of its prompt without recomputation and gradients flow back into the prompt.
//  3. In BF16 mode dt / A / the SSD run in fp32 (HF computes softplus(dt + dt_bias) in the activation dtype); FP32 / FP64 modes are identical in structure.
//  4. dt clamp bounds come from the config (dt_clamp_min / dt_clamp_max): HF's fallback clamps at time_step_min only, the CUDA kernel path at time_step_limit.
//  5. MoE: the expert loop is expressed as sort-by-expert + per-expert linear + one index_add (same arithmetic as HF's per-expert loop); a grouped-GEMM
//     variant (MoeImpl::GroupedMM) uses at::_grouped_mm.
//  6. Frozen quantised linears: dequantised to the compute dtype per use; no custom transposed-product kernel (v1).

namespace torch::models::nemotron_h {

namespace {
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
constexpr auto kF = torch::kFloat32;

ScalarType stable_dt(ScalarType adt) { return adt == torch::kFloat64 ? torch::kFloat64 : torch::kFloat32; }

ScalarType compute_dtype(Compute c) {
  switch (c) {
    case Compute::BF16: return torch::kBFloat16;
    case Compute::FP32: return torch::kFloat32;
    default: return torch::kFloat64;
  }
}

struct Ctx {
  ScalarType adt, sdt;
  const ForwardOptions* o;
  const Config* cfg;
};

Tensor relu2(const Tensor& x) { return at::relu(x).square(); }

// NemotronHRMSNorm: fp32 statistics, weight * x.to(input dtype)
Tensor rms(const Tensor& x, const Tensor& w, double eps) {
  const ScalarType in_dt = x.scalar_type();
  Tensor xf = x.to(stable_dt(in_dt));
  xf = xf * at::rsqrt(xf.pow(2).mean(-1, true) + eps);
  return w.to(in_dt) * xf.to(in_dt);
}

// Zamba2RMSNormGated (grouped): x*silu(gate) in fp32, RMS over groups of group_size, weight * x.to(input dtype)
Tensor rms_gated_grouped(const Tensor& y, const Tensor& gate, const Tensor& w, int64_t groups, double eps) {
  const ScalarType in_dt = y.scalar_type();
  Tensor h = y.to(stable_dt(in_dt)) * at::silu(gate.to(stable_dt(in_dt)));
  auto sz = h.sizes().vec();
  const int64_t last = sz.back();
  sz.back() = groups;
  sz.push_back(last / groups);
  Tensor hg = h.view(sz);
  hg = hg * at::rsqrt(hg.pow(2).mean(-1, true) + eps);
  Tensor out = hg.view(h.sizes());
  return w.to(in_dt) * out.to(in_dt);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// activation quantisers (straight-through)
// ---------------------------------------------------------------------------------------------------------------------------------------------
Tensor e2m1_rne(const Tensor& r) {
  // round |r| to the nearest e2m1 value, ties to the even code, saturating at 6; sign restored
  Tensor a = r.abs();
  static const double thr[7] = {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0};
  static const bool tie_up[7] = {false, true, false, true, false, true, false};
  Tensor idx = at::zeros(a.sizes(), a.options().dtype(torch::kLong));
  for (int k = 0; k < 7; ++k) idx = idx + (tie_up[k] ? (a >= thr[k]) : (a > thr[k])).to(torch::kLong);
  Tensor lut = at::tensor({0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}, torch::TensorOptions().dtype(r.scalar_type()).device(r.device()));
  Tensor q = lut.index_select(0, idx.flatten()).view_as(a);
  return at::where(r < 0, -q, q);
}

struct ActQuantSTE : public torch::autograd::Function<ActQuantSTE> {
  static Tensor forward(AutogradContext* ctx, const Tensor& x, const Tensor& xq, const Tensor& mask) {
    ctx->save_for_backward({mask});
    return xq.clone();
  }
  static variable_list backward(AutogradContext* ctx, variable_list go) {
    Tensor mask = ctx->get_saved_variables()[0];
    return {go[0] * mask.to(go[0].scalar_type()), Tensor(), Tensor()};
  }
};

// ---------------------------------------------------------------------------------------------------------------------------------------------
// linear with frozen (possibly quantised) base + LoRA branch from the UNQUANTISED input (PEFT semantics)
// ---------------------------------------------------------------------------------------------------------------------------------------------
Tensor lin(const Tensor& x, const Linear& L, const Ctx& c) {
  Tensor xin = x;
  if (c.o->act_quant && L.q.fmt != QFormat::Dense && L.q.input_scale.defined()) xin = act_quant_ste(x, L.q);
  Tensor y = at::linear(xin, L.q.dequant(c.adt));
  if (L.lora.defined()) {
    Tensor d = at::linear(at::linear(x.to(c.sdt), L.lora.A.to(c.sdt)), L.lora.B.to(c.sdt)) * L.lora.scale;
    y = (y.to(c.sdt) + d).to(c.adt);
  }
  return y;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// SSD, pure ATen: translation of NemotronHMamba2Mixer.torch_forward ("begin ssd naive implementation"), state-passing axis fixed (deviation 1)
// ---------------------------------------------------------------------------------------------------------------------------------------------
Tensor pad_seq(const Tensor& t, int64_t pad) {
  if (pad == 0) return t;
  if (t.dim() == 4) return at::pad(t, {0, 0, 0, 0, 0, pad});
  return at::pad(t, {0, 0, 0, pad});
}
Tensor to_chunks(const Tensor& t, int64_t chunk) {
  const int64_t b = t.size(0);
  if (t.dim() == 3) return t.reshape({b, -1, chunk, t.size(2)});
  return t.reshape({b, -1, chunk, t.size(2), t.size(3)});
}
Tensor segment_sum(const Tensor& t) {
  const int64_t cs = t.size(-1);
  auto sz = t.sizes().vec();
  sz.push_back(cs);
  Tensor x = t.unsqueeze(-1).expand(sz);
  Tensor m1 = at::tril(at::ones({cs, cs}, torch::TensorOptions().dtype(torch::kBool).device(t.device())), -1);
  x = x.masked_fill(~m1, 0);
  x = at::cumsum(x, -2);
  Tensor m2 = at::tril(at::ones({cs, cs}, torch::TensorOptions().dtype(torch::kBool).device(t.device())), 0);
  return x.masked_fill(~m2, -std::numeric_limits<double>::infinity());
}
}  // namespace

using SsdBatched = std::pair<Tensor, Tensor> (*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, int64_t);

// Varlen contract of SsdFn on top of a batched implementation: x [1,T,H,P] packed, cu_seqlens int32 [N+1], h0 [N,H,P,N] or undefined -> y [1,T,H,P], final states [N,H,P,N].
// Loops the sequences and carries h0[n] per sequence (the AOT kernel does this natively).
std::pair<Tensor, Tensor> ssd_varlen_loop(SsdBatched f, const Tensor& x, const Tensor& dt, const Tensor& A, const Tensor& Bh, const Tensor& Ch, const Tensor& D, const Tensor& h0,
                                          int64_t chunk, const Tensor& cu_seqlens) {
  TORCH_CHECK(x.size(0) == 1, "nemotron_h: varlen SSD needs the packed layout [1,T,...]");
  Tensor cu = cu_seqlens.to(torch::kCPU).to(torch::kInt64).contiguous();
  const int64_t N = cu.numel() - 1;
  const int64_t* c = cu.data_ptr<int64_t>();
  TORCH_CHECK(c[N] == x.size(1), "nemotron_h: cu_seqlens does not cover the packed tokens");
  std::vector<Tensor> ys, st;
  for (int64_t n = 0; n < N; ++n) {
    const int64_t s = c[n], len = c[n + 1] - c[n];
    TORCH_CHECK(len > 0, "nemotron_h: empty sequence in cu_seqlens");
    Tensor h0n = h0.defined() ? h0.narrow(0, n, 1) : Tensor();
    auto r = f(x.narrow(1, s, len), dt.narrow(1, s, len), A, Bh.narrow(1, s, len), Ch.narrow(1, s, len), D, h0n, chunk);
    ys.push_back(r.first);
    st.push_back(r.second);
  }
  return {at::cat(ys, 1), at::cat(st, 0)};
}

namespace {
std::pair<Tensor, Tensor> ssd_aten_b(const Tensor& x, const Tensor& dt, const Tensor& A, const Tensor& Bh, const Tensor& Ch, const Tensor& D, const Tensor& h0, int64_t chunk) {
  const int64_t b = x.size(0), l = x.size(1), h = x.size(2), p = x.size(3);
  const int64_t pad = (chunk - l % chunk) % chunk;
  Tensor D_res = D.unsqueeze(-1) * pad_seq(x, pad);
  Tensor xs = x * dt.unsqueeze(-1);
  Tensor Ad = A * dt;
  xs = to_chunks(pad_seq(xs, pad), chunk);   // [b,c,l,h,p]
  Ad = to_chunks(pad_seq(Ad, pad), chunk);   // [b,c,l,h]
  Tensor B = to_chunks(pad_seq(Bh, pad), chunk), C = to_chunks(pad_seq(Ch, pad), chunk);  // [b,c,l,h,n]
  const int64_t nc = xs.size(1);
  Ad = Ad.permute({0, 3, 1, 2});             // [b,h,c,l]
  Tensor Acs = at::cumsum(Ad, -1);
  // 1. intra-chunk (diagonal blocks)
  Tensor Lm = at::exp(segment_sum(Ad));      // [b,h,c,l,s]
  Tensor G = (C.unsqueeze(3) * B.unsqueeze(2)).sum(-1);                                          // [b,c,l,s,h]
  Tensor M = (G.unsqueeze(-1) * Lm.permute({0, 2, 3, 4, 1}).unsqueeze(-1)).sum(-1);              // [b,c,l,s,h]
  Tensor Y_diag = (M.unsqueeze(-1) * xs.unsqueeze(2)).sum(3);                                    // [b,c,l,h,p]
  // 2. per-chunk states
  Tensor decay_states = at::exp(Acs.narrow(3, Acs.size(3) - 1, 1) - Acs);                       // [b,h,c,l]
  Tensor B_dc = B * decay_states.permute({0, 2, 3, 1}).unsqueeze(-1);                            // [b,c,l,h,n]
  Tensor states = (B_dc.permute({0, 1, 3, 2, 4}).unsqueeze(-1) * xs.permute({0, 1, 3, 2, 4}).unsqueeze(-2)).sum(3).permute({0, 1, 2, 4, 3});  // [b,c,h,p,n]
  Tensor prev = h0.defined() ? h0.to(states.scalar_type()).unsqueeze(1) : at::zeros_like(states.narrow(1, 0, 1));
  states = at::cat({prev, states}, 1);                                                           // [b,c+1,h,p,n]
  // 3. state passing between chunks (source chunk = dim 3)
  Tensor decay_chunk = at::exp(segment_sum(at::pad(Acs.select(-1, Acs.size(-1) - 1), {1, 0})));  // [b,h,c+1,c+1]
  Tensor sp = states.permute({0, 2, 1, 3, 4});                                                   // [b,h,c+1,p,n]
  Tensor result = (decay_chunk.unsqueeze(-1).unsqueeze(-1) * sp.unsqueeze(2)).sum(3);            // [b,h,c+1,p,n]
  Tensor new_states = result.permute({0, 2, 1, 3, 4});                                           // [b,c+1,h,p,n]
  states = new_states.narrow(1, 0, nc);
  Tensor ssm_state = new_states.select(1, nc);
  // 4. state -> output
  Tensor sdo = at::exp(Acs).permute({0, 2, 3, 1});                                               // [b,c,l,h]
  Tensor Y_off = (C.unsqueeze(-2) * states.unsqueeze(2)).sum(-1) * sdo.unsqueeze(-1);            // [b,c,l,h,p]
  Tensor y = (Y_diag + Y_off).reshape({b, -1, h, p}) + D_res;
  if (pad > 0) y = y.narrow(1, 0, l);
  return {y, ssm_state};
}

std::pair<Tensor, Tensor> ssd_serial_b(const Tensor& x, const Tensor& dt, const Tensor& A, const Tensor& Bh, const Tensor& Ch, const Tensor& D, const Tensor& h0, int64_t /*chunk*/) {
  const int64_t T = x.size(1);
  Tensor state = h0.defined() ? h0.to(x.scalar_type()) : at::zeros({x.size(0), x.size(2), x.size(3), Bh.size(3)}, x.options());
  std::vector<Tensor> ys;
  for (int64_t t = 0; t < T; ++t) {
    Tensor dAt = at::exp(dt.select(1, t) * A);  // [b,h]
    Tensor xt = x.select(1, t);                 // [b,h,p]
    state = state * dAt.unsqueeze(-1).unsqueeze(-1) + (dt.select(1, t).unsqueeze(-1) * xt).unsqueeze(-1) * Bh.select(1, t).unsqueeze(2);
    ys.push_back((state * Ch.select(1, t).unsqueeze(2)).sum(-1) + D.unsqueeze(-1) * xt);
  }
  return {at::stack(ys, 1), state};
}
}  // namespace

std::pair<Tensor, Tensor> ssd_aten(const Tensor& x, const Tensor& dt, const Tensor& A, const Tensor& Bh, const Tensor& Ch, const Tensor& D, const Tensor& h0, int64_t chunk,
                                   const Tensor& cu_seqlens) {
  if (cu_seqlens.defined()) return ssd_varlen_loop(&ssd_aten_b, x, dt, A, Bh, Ch, D, h0, chunk, cu_seqlens);
  return ssd_aten_b(x, dt, A, Bh, Ch, D, h0, chunk);
}
std::pair<Tensor, Tensor> ssd_serial(const Tensor& x, const Tensor& dt, const Tensor& A, const Tensor& Bh, const Tensor& Ch, const Tensor& D, const Tensor& h0, int64_t chunk,
                                     const Tensor& cu_seqlens) {
  if (cu_seqlens.defined()) return ssd_varlen_loop(&ssd_serial_b, x, dt, A, Bh, Ch, D, h0, chunk, cu_seqlens);
  return ssd_serial_b(x, dt, A, Bh, Ch, D, h0, chunk);
}

Tensor act_quant_ste(const Tensor& x, const QWeight& w) {
  TORCH_CHECK(w.input_scale.defined(), "nemotron_h: act_quant_ste needs the weight's input_scale");
  const ScalarType sdt = stable_dt(x.scalar_type());
  Tensor s = w.input_scale.to(x.device()).to(sdt).reshape({});
  Tensor xq, mask;
  {
  torch::NoGradGuard ng;  // the quantiser value is computed outside the graph; the Function below attaches the straight-through gradient
  if (w.fmt == QFormat::FP8) {
    Tensor r = x.to(sdt) / s;
    mask = r.abs() <= 448.0;
    xq = r.clamp(-448.0, 448.0).to(torch::kFloat8_e4m3fn).to(sdt) * s;
  } else {
    TORCH_CHECK(w.fmt == QFormat::NVFP4, "nemotron_h: act_quant_ste on a dense weight");
    const int64_t K = x.size(-1);
    TORCH_CHECK(K % 16 == 0, "nemotron_h: NVFP4 activation quantiser needs K % 16 == 0");
    Tensor xb = (x.to(sdt) / s).reshape({-1, K / 16, 16});
    Tensor sf = (xb.abs().amax(-1, true) / 6.0).clamp(0.0, 448.0).to(torch::kFloat8_e4m3fn).to(sdt);
    Tensor pos = sf > 0;
    Tensor r = at::where(pos, xb / at::where(pos, sf, at::ones_like(sf)), at::zeros_like(xb));
    mask = (r.abs() <= 6.0).reshape(x.sizes());
    xq = (e2m1_rne(r) * sf * s).reshape(x.sizes());
  }
  }
  return ActQuantSTE::apply(x, xq.to(x.scalar_type()), mask);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// QWeight
// ---------------------------------------------------------------------------------------------------------------------------------------------
Tensor QWeight::dequant(ScalarType dt) const {
  if (fmt == QFormat::Dense) return w.to(dt);
  const ScalarType sdt = stable_dt(dt);
  auto s2shape = scale2.sizes().vec();
  s2shape.push_back(1);
  s2shape.push_back(1);
  Tensor s2 = scale2.to(sdt).reshape(s2shape);
  if (fmt == QFormat::FP8) return (w.to(sdt) * s2).to(dt);
  Tensor lut = at::tensor({0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0}, torch::TensorOptions().dtype(sdt).device(w.device()));
  Tensor lo = (w & 15).to(torch::kLong), hi = at::bitwise_right_shift(w, 4).to(torch::kLong);
  Tensor vlo = lut.index_select(0, lo.flatten()).view_as(lo), vhi = lut.index_select(0, hi.flatten()).view_as(hi);
  auto sz = w.sizes().vec();
  sz.back() *= 2;
  Tensor vals = at::stack({vlo, vhi}, -1).reshape(sz);
  Tensor sc = scale.to(sdt).repeat_interleave(16, -1);
  return (vals * sc * s2).to(dt);
}

QWeight QWeight::at(int64_t e) const {
  QWeight r;
  r.fmt = fmt;
  r.w = w.select(0, e);
  if (scale.defined()) r.scale = scale.select(0, e);
  if (scale2.defined()) r.scale2 = scale2.dim() > 0 ? scale2.select(0, e) : scale2;
  if (input_scale.defined()) r.input_scale = input_scale.dim() > 0 ? input_scale.select(0, e) : input_scale;
  return r;
}

namespace {
// ---------------------------------------------------------------------------------------------------------------------------------------------
// mamba2 mixer
// ---------------------------------------------------------------------------------------------------------------------------------------------
Tensor mamba_fwd(const Tensor& x, const Layer& L, const Ctx& c, const LayerState* sin, LayerState* sout) {
  const Config& cf = *c.cfg;
  const int64_t B = x.size(0), T = x.size(1), H = cf.mamba_num_heads, P = cf.mamba_head_dim, N = cf.ssm_state_size, G = cf.n_groups, K = cf.conv_kernel;
  const int64_t di = cf.d_inner(), cd = cf.conv_dim();
  Tensor proj = lin(x, L.in_proj, c);
  const int64_t d_mlp = (proj.size(-1) - 2 * di - 2 * G * N - H) / 2;  // HF: the first 2*d_mlp channels are dropped
  Tensor z = proj.narrow(-1, 2 * d_mlp, di);
  Tensor xBC = proj.narrow(-1, 2 * d_mlp + di, cd).transpose(1, 2);   // [B,cd,T]
  Tensor dt_raw = proj.narrow(-1, 2 * d_mlp + di + cd, H);
  // causal depthwise conv with left context (zeros without a prefix); the last K-1 pre-conv columns are the hand-off state
  Tensor ctx_left = (sin && sin->conv.defined()) ? sin->conv.to(c.adt) : at::zeros({B, cd, K - 1}, xBC.options());
  Tensor ext = at::cat({ctx_left, xBC}, -1);  // [B,cd,T+K-1]
  c10::optional<Tensor> bias;
  if (cf.use_conv_bias && L.conv_b.defined()) bias = L.conv_b.to(c.adt);
  Tensor conv = at::conv1d(ext, L.conv_w.to(c.adt), bias, at::IntArrayRef{1}, at::IntArrayRef{0}, at::IntArrayRef{1}, cd);  // [B,cd,T]
  if (sout) sout->conv = ext.narrow(-1, ext.size(-1) - (K - 1), K - 1);
  Tensor hbc = at::silu(conv).transpose(1, 2);  // [B,T,cd]
  Tensor hs = hbc.narrow(-1, 0, di), Bm = hbc.narrow(-1, di, G * N), Cm = hbc.narrow(-1, di + G * N, G * N);
  Tensor A = -at::exp(L.A_log.to(c.sdt));
  Tensor dt = at::softplus(dt_raw.to(c.sdt) + L.dt_bias.to(c.sdt));
  c10::optional<c10::Scalar> mn = c10::Scalar(cf.dt_clamp_min), mx;
  if (std::isfinite(cf.dt_clamp_max)) mx = c10::Scalar(cf.dt_clamp_max);
  dt = at::clamp(dt, mn, mx);
  Tensor xs = hs.reshape({B, T, H, P}).to(c.sdt);
  Tensor Bh = Bm.reshape({B, T, G, N}).to(c.sdt).repeat_interleave(H / G, 2);
  Tensor Ch = Cm.reshape({B, T, G, N}).to(c.sdt).repeat_interleave(H / G, 2);
  Tensor h0 = (sin && sin->ssm.defined()) ? sin->ssm : Tensor();
  auto yst = c.o->ssd ? c.o->ssd(xs, dt, A, Bh, Ch, L.D.to(c.sdt), h0, cf.chunk_size, Tensor()) : ssd_aten(xs, dt, A, Bh, Ch, L.D.to(c.sdt), h0, cf.chunk_size);
  if (sout) sout->ssm = yst.second;
  Tensor y = yst.first.reshape({B, T, di});
  Tensor g = rms_gated_grouped(y, z, L.gnorm, G, cf.norm_eps).to(c.adt);
  return lin(g, L.out_proj, c);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// attention (no rope, no q/k norm, no gate)
// ---------------------------------------------------------------------------------------------------------------------------------------------
Tensor attn_fwd(const Tensor& x, const Layer& L, const Ctx& c, const LayerState* sin, LayerState* sout) {
  const Config& cf = *c.cfg;
  const int64_t B = x.size(0), T = x.size(1), nH = cf.num_attention_heads, nKV = cf.num_key_value_heads, hd = cf.head_dim;
  Tensor q = lin(x, L.q_proj, c).view({B, T, nH, hd}).transpose(1, 2);
  Tensor k = lin(x, L.k_proj, c).view({B, T, nKV, hd}).transpose(1, 2);
  Tensor v = lin(x, L.v_proj, c).view({B, T, nKV, hd}).transpose(1, 2);
  if (sin && sin->k.defined()) {
    k = at::cat({sin->k.to(c.adt), k}, 2);
    v = at::cat({sin->v.to(c.adt), v}, 2);
  }
  if (sout) {
    sout->k = k;
    sout->v = v;
  }
  const int64_t Tk = k.size(2), g = nH / nKV;
  Tensor kr = g > 1 ? k.repeat_interleave(g, 1) : k, vr = g > 1 ? v.repeat_interleave(g, 1) : v;  // repeat_kv order
  const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
  Tensor o;
  if (Tk == T) {
    o = at::scaled_dot_product_attention(q, kr, vr, c10::nullopt, 0.0, true, scale, false);
  } else {
    Tensor mask = at::tril(at::ones({T, Tk}, torch::TensorOptions().dtype(torch::kBool).device(x.device())), Tk - T);  // query i sees keys j <= i + (Tk - T)
    o = at::scaled_dot_product_attention(q, kr, vr, mask, 0.0, false, scale, false);
  }
  return lin(o.transpose(1, 2).reshape({B, T, nH * hd}), L.o_proj, c);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// MoE: router (fp32), routed experts in the latent, shared expert
// ---------------------------------------------------------------------------------------------------------------------------------------------
Tensor moe_fwd(const Tensor& x, const Layer& L, const Ctx& c) {
  const Config& cf = *c.cfg;
  const int64_t E = cf.n_routed_experts, k = cf.num_experts_per_tok;
  auto shp = x.sizes().vec();
  Tensor xf = x.reshape({-1, x.size(-1)});
  const int64_t N = xf.size(0);
  // ---- router (NemotronHTopkRouter): logits in fp32; the selection is integer, the weights come from `scores` WITHOUT the bias and stay in the graph
  Tensor xr = xf.to(c.sdt);
  Tensor logits = at::linear(xr, L.gate_w.to(c.sdt));
  if (L.gate_lora.defined()) logits = logits + at::linear(at::linear(xr, L.gate_lora.A.to(c.sdt)), L.gate_lora.B.to(c.sdt)) * L.gate_lora.scale;
  Tensor scores = logits.sigmoid();
  Tensor idx;
  {
    torch::NoGradGuard ng;
    Tensor sfc = scores.detach() + L.gate_bias.to(c.sdt).unsqueeze(0);
    Tensor gs = std::get<0>(sfc.view({-1, cf.n_group, E / cf.n_group}).topk(2, -1)).sum(-1);
    Tensor gidx = std::get<1>(at::topk(gs, cf.topk_group, -1, true, false));
    Tensor gmask = at::zeros_like(gs).scatter_(1, gidx, 1.0);
    Tensor smask = gmask.unsqueeze(-1).expand({-1, cf.n_group, E / cf.n_group}).reshape({-1, E});
    sfc = sfc.masked_fill(~smask.to(torch::kBool), -std::numeric_limits<double>::infinity());
    idx = std::get<1>(at::topk(sfc, k, -1, true, false));
  }
  Tensor w = scores.gather(1, idx);
  if (cf.norm_topk_prob) w = w / (w.sum(-1, true) + 1e-20);
  w = w * cf.routed_scaling_factor;  // [N,k] fp32/fp64, in graph
  // ---- latent projection + routed experts
  Tensor zl = L.fc1.defined() ? lin(xf, L.fc1, c) : xf;
  const int64_t Lw = zl.size(-1);
  Tensor flat_e = idx.reshape({-1});
  Tensor order = at::argsort(flat_e, 0, false);  // stable not required: order inside an expert only permutes summation order
  Tensor tok = at::floor_divide(order, k);
  Tensor wsorted = w.reshape({-1}).index_select(0, order);
  Tensor counts = at::bincount(flat_e, {}, E).cpu();
  Tensor xs = zl.index_select(0, tok);
  Tensor V;
  if (c.o->moe == MoeImpl::GroupedMM) {
    TORCH_CHECK(L.up.fmt == QFormat::Dense, "nemotron_h: GroupedMM needs dense stacked experts (quantised experts use the per-expert loop)");
    Tensor offs = at::cumsum(counts, 0).to(torch::kInt32).to(x.device());
    Tensor Wu = L.up.dequant(c.adt), Wd = L.down.dequant(c.adt);
    Tensor u = at::_grouped_mm(xs, Wu.transpose(1, 2), offs);
    V = at::_grouped_mm(relu2(u), Wd.transpose(1, 2), offs);
  } else {
    std::vector<Tensor> outs;
    int64_t off = 0;
    const int64_t* cn = counts.data_ptr<int64_t>();
    for (int64_t e = 0; e < E; ++e) {
      if (cn[e] == 0) continue;
      Tensor seg = xs.narrow(0, off, cn[e]);
      outs.push_back(at::linear(relu2(at::linear(seg, L.up.at(e).dequant(c.adt))), L.down.at(e).dequant(c.adt)));
      off += cn[e];
    }
    V = at::cat(outs, 0);
  }
  V = V * wsorted.unsqueeze(-1).to(V.scalar_type());
  Tensor o = at::zeros({N, Lw}, torch::TensorOptions().dtype(w.scalar_type()).device(x.device())).index_add(0, tok, V.to(w.scalar_type())).to(c.adt);
  Tensor y = L.fc2.defined() ? lin(o, L.fc2, c) : o;
  y = y + lin(relu2(lin(xf, L.sh_up, c)), L.sh_down, c);
  return y.reshape(shp);
}

Tensor mlp_fwd(const Tensor& x, const Layer& L, const Ctx& c) { return lin(relu2(lin(x, L.sh_up, c)), L.sh_down, c); }

// ---------------------------------------------------------------------------------------------------------------------------------------------
// packed-token mixers (the trainer's layout: [T, hidden] rows, sequences concatenated)
// ---------------------------------------------------------------------------------------------------------------------------------------------
Tensor cu_of(const std::vector<int64_t>& lens) {
  std::vector<int32_t> v(1, 0);
  for (int64_t l : lens) v.push_back(v.back() + static_cast<int32_t>(l));
  return torch::tensor(v, torch::kInt32);
}

// One segment of a packed batch: h [1,Ts,hidden] holds sequences `lens`. h0 [N,H,P,N] / conv_ctx [N,cd,K-1] (undefined = zeros) are the hand-off states of the prefixes;
// ssm_out [N,H,P,N] / conv_out [N,cd,K-1] receive the states after the segment (graph-connected).
Tensor mamba_packed_seg(const Tensor& h, const Layer& L, const Ctx& c, const std::vector<int64_t>& lens, const Tensor& h0, const Tensor& conv_ctx, Tensor* ssm_out, Tensor* conv_out) {
  const Config& cf = *c.cfg;
  const int64_t Ts = h.size(1), H = cf.mamba_num_heads, P = cf.mamba_head_dim, N = cf.ssm_state_size, G = cf.n_groups, K = cf.conv_kernel;
  const int64_t di = cf.d_inner(), cd = cf.conv_dim();
  Tensor proj = lin(h, L.in_proj, c);
  const int64_t d_mlp = (proj.size(-1) - 2 * di - 2 * G * N - H) / 2;
  Tensor z = proj.narrow(-1, 2 * d_mlp, di);
  Tensor xBC = proj.narrow(-1, 2 * d_mlp + di, cd).transpose(1, 2);   // [1,cd,Ts]
  Tensor dt_raw = proj.narrow(-1, 2 * d_mlp + di + cd, H);
  // causal depthwise conv, restarted at every sequence (left context = the prefix's last K-1 pre-conv columns, zeros without a prefix)
  c10::optional<Tensor> bias;
  if (cf.use_conv_bias && L.conv_b.defined()) bias = L.conv_b.to(c.adt);
  Tensor cw = L.conv_w.to(c.adt);
  std::vector<Tensor> convs, tails;
  int64_t off = 0;
  for (size_t n = 0; n < lens.size(); ++n) {
    Tensor left = conv_ctx.defined() ? conv_ctx.narrow(0, static_cast<int64_t>(n), 1).to(c.adt) : at::zeros({1, cd, K - 1}, xBC.options());
    Tensor ext = at::cat({left, xBC.narrow(-1, off, lens[n])}, -1);
    convs.push_back(at::conv1d(ext, cw, bias, at::IntArrayRef{1}, at::IntArrayRef{0}, at::IntArrayRef{1}, cd));
    tails.push_back(ext.narrow(-1, ext.size(-1) - (K - 1), K - 1));
    off += lens[n];
  }
  if (conv_out) *conv_out = at::cat(tails, 0);
  Tensor conv = convs.size() == 1 ? convs[0] : at::cat(convs, -1);   // [1,cd,Ts]
  Tensor hbc = at::silu(conv).transpose(1, 2);
  Tensor hs = hbc.narrow(-1, 0, di), Bm = hbc.narrow(-1, di, G * N), Cm = hbc.narrow(-1, di + G * N, G * N);
  Tensor A = -at::exp(L.A_log.to(c.sdt));
  Tensor dt = at::softplus(dt_raw.to(c.sdt) + L.dt_bias.to(c.sdt));
  c10::optional<c10::Scalar> mn = c10::Scalar(cf.dt_clamp_min), mx;
  if (std::isfinite(cf.dt_clamp_max)) mx = c10::Scalar(cf.dt_clamp_max);
  dt = at::clamp(dt, mn, mx);
  Tensor xs = hs.reshape({1, Ts, H, P}).to(c.sdt);
  Tensor Bh = Bm.reshape({1, Ts, G, N}).to(c.sdt).repeat_interleave(H / G, 2);
  Tensor Ch = Cm.reshape({1, Ts, G, N}).to(c.sdt).repeat_interleave(H / G, 2);
  Tensor cu = cu_of(lens);
  auto yst = c.o->ssd ? c.o->ssd(xs, dt, A, Bh, Ch, L.D.to(c.sdt), h0, cf.chunk_size, cu) : ssd_aten(xs, dt, A, Bh, Ch, L.D.to(c.sdt), h0, cf.chunk_size, cu);
  if (ssm_out) *ssm_out = yst.second;
  Tensor y = yst.first.reshape({1, Ts, di});
  Tensor g = rms_gated_grouped(y, z, L.gnorm, G, cf.norm_eps).to(c.adt);
  return lin(g, L.out_proj, c);
}

// Attention per sequence (causality restarts at every sequence). prefix[n] (optional, per sequence of this segment): the K/V the sequence continues; kv_out receives every sequence's full K/V.
Tensor attn_packed_seg(const Tensor& h, const Layer& L, const Ctx& c, const std::vector<int64_t>& lens, const std::vector<LayerState>* prefix, std::vector<LayerState>* kv_out) {
  std::vector<Tensor> outs;
  int64_t off = 0;
  if (kv_out) kv_out->assign(lens.size(), LayerState{});
  for (size_t n = 0; n < lens.size(); ++n) {
    const LayerState* sin = (prefix && n < prefix->size()) ? &(*prefix)[n] : nullptr;
    outs.push_back(attn_fwd(h.narrow(1, off, lens[n]), L, c, sin, kv_out ? &(*kv_out)[n] : nullptr));
    off += lens[n];
  }
  return outs.size() == 1 ? outs[0] : at::cat(outs, 1);
}

std::vector<int64_t> resp_index_vec(const SeqLayout& lay) { return lay.resp_group; }

// mixer of one block on the packed [T,hidden] layout (the block's pre-norm already applied)
Tensor mixer_packed(const Layer& L, const Tensor& h2, const SeqLayout& lay, const Ctx& c) {
  if (L.type == BlockType::MoE) return moe_fwd(h2, L, c);
  if (L.type == BlockType::MLP) return mlp_fwd(h2, L, c);
  const int64_t T = h2.size(0);
  TORCH_CHECK(lay.tokens() == T, "nemotron_h: SeqLayout covers ", lay.tokens(), " tokens, activations have ", T);
  if (L.type == BlockType::Attention && c.o->attn) {   // varlen flash path: one call over the whole packed batch
    const int64_t nH = c.cfg->num_attention_heads, nKV = c.cfg->num_key_value_heads, hd = c.cfg->head_dim;
    Tensor q = lin(h2, L.q_proj, c).view({T, nH, hd}), k = lin(h2, L.k_proj, c).view({T, nKV, hd}), v = lin(h2, L.v_proj, c).view({T, nKV, hd});
    Tensor o = c.o->attn(q, k, v);
    return lin(o.reshape({T, nH * hd}), L.o_proj, c);
  }
  Tensor h = h2.unsqueeze(0);
  if (!lay.grouped) {
    if (L.type == BlockType::Mamba) return mamba_packed_seg(h, L, c, lay.lens, Tensor(), Tensor(), nullptr, nullptr).squeeze(0);
    return attn_packed_seg(h, L, c, lay.lens, nullptr, nullptr).squeeze(0);
  }
  int64_t Tp = 0;
  for (int64_t l : lay.prompt_lens) Tp += l;
  Tensor hp = h.narrow(1, 0, Tp), hr = h.narrow(1, Tp, T - Tp);
  Tensor gi = torch::tensor(resp_index_vec(lay), torch::kInt64).to(h2.device());
  if (L.type == BlockType::Mamba) {
    Tensor ssm, conv;
    Tensor yp = mamba_packed_seg(hp, L, c, lay.prompt_lens, Tensor(), Tensor(), &ssm, &conv);
    Tensor yr = mamba_packed_seg(hr, L, c, lay.resp_lens, ssm.index_select(0, gi), conv.index_select(0, gi), nullptr, nullptr);
    return at::cat({yp, yr}, 1).squeeze(0);
  }
  std::vector<LayerState> kv;
  Tensor yp = attn_packed_seg(hp, L, c, lay.prompt_lens, nullptr, &kv);
  std::vector<LayerState> pre;
  for (int64_t g : lay.resp_group) pre.push_back(kv[static_cast<size_t>(g)]);
  Tensor yr = attn_packed_seg(hr, L, c, lay.resp_lens, &pre, nullptr);
  return at::cat({yp, yr}, 1).squeeze(0);
}

Tensor layer_fwd(const Layer& L, const Tensor& x, const Ctx& c, const LayerState* sin, LayerState* sout) {
  Tensor h = rms(x, L.norm, c.cfg->norm_eps);
  switch (L.type) {
    case BlockType::Mamba: h = mamba_fwd(h, L, c, sin, sout); break;
    case BlockType::Attention: h = attn_fwd(h, L, c, sin, sout); break;
    case BlockType::MoE: h = moe_fwd(h, L, c); break;
    case BlockType::MLP: h = mlp_fwd(h, L, c); break;
  }
  return x + h;
}

std::string idx_name(int64_t i, const std::string& s) { return "layers." + std::to_string(i) + ".mixer." + s; }
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Config / loading
// ---------------------------------------------------------------------------------------------------------------------------------------------
Config Config::FromFile(const std::string& path) {
  std::ifstream f(path);
  TORCH_CHECK(f, "nemotron_h: cannot open ", path);
  std::stringstream ss;
  ss << f.rdbuf();
  auto j = nlohmann::json::parse(ss.str());
  const auto& t = j.contains("llm_config") ? j.at("llm_config") : j;
  auto i64 = [&](const char* k, int64_t d) { return (t.contains(k) && !t.at(k).is_null()) ? t.at(k).get<int64_t>() : d; };
  auto f64 = [&](const char* k, double d) { return (t.contains(k) && !t.at(k).is_null() && t.at(k).is_number()) ? t.at(k).get<double>() : d; };
  Config c;
  c.vocab_size = t.at("vocab_size"); c.hidden_size = t.at("hidden_size");
  c.intermediate_size = i64("intermediate_size", 0);
  c.norm_eps = f64("layer_norm_epsilon", 1e-5);
  c.num_attention_heads = i64("num_attention_heads", 0); c.num_key_value_heads = i64("num_key_value_heads", 0);
  c.head_dim = i64("head_dim", c.num_attention_heads ? c.hidden_size / c.num_attention_heads : 0);
  c.mamba_num_heads = i64("mamba_num_heads", 0); c.mamba_head_dim = i64("mamba_head_dim", 0); c.ssm_state_size = i64("ssm_state_size", 0);
  c.n_groups = i64("n_groups", 1); c.conv_kernel = i64("conv_kernel", 4); c.chunk_size = i64("chunk_size", 128);
  c.use_conv_bias = t.value("use_conv_bias", true);
  c.time_step_min = f64("time_step_min", 0.0);   // dt_clamp_min stays 0 (no clamp, as the serving path); LoadOpts::hf_dt_clamp switches the HF fallback clamp on
  c.n_routed_experts = i64("n_routed_experts", 0); c.num_experts_per_tok = i64("num_experts_per_tok", 0);
  c.moe_intermediate_size = i64("moe_intermediate_size", 0); c.moe_latent_size = i64("moe_latent_size", 0);
  c.moe_shared_expert_intermediate_size = i64("moe_shared_expert_intermediate_size", 0);
  c.n_group = i64("n_group", 1); c.topk_group = i64("topk_group", 1);
  c.norm_topk_prob = t.value("norm_topk_prob", true);
  c.routed_scaling_factor = f64("routed_scaling_factor", 1.0);
  c.tie_word_embeddings = t.value("tie_word_embeddings", j.value("tie_word_embeddings", false));
  for (const auto& s : t.at("layers_block_type")) {
    const std::string n = s.get<std::string>();
    if (n == "mamba" || n == "linear_attention") c.blocks.push_back(BlockType::Mamba);
    else if (n == "attention" || n == "full_attention") c.blocks.push_back(BlockType::Attention);
    else if (n == "moe") c.blocks.push_back(BlockType::MoE);
    else if (n == "mlp") c.blocks.push_back(BlockType::MLP);
    else TORCH_CHECK(false, "nemotron_h: unknown block type '", n, "'");
  }
  return c;
}

QWeight load_qweight(const utils::safetensors::SafetensorsCollection& coll, const std::string& base, torch::Device dev, torch::ScalarType dense_dtype) {
  QWeight q;
  q.present = true;
  const std::string wn = base + ".weight";
  const auto dt = coll.dtype(wn);
  if (dt == torch::kUInt8) {
    q.fmt = QFormat::NVFP4;
    q.w = coll.get(wn, torch::kUInt8, dev);
    q.scale = coll.get(base + ".weight_scale", torch::kFloat8_e4m3fn, dev);
    q.scale2 = coll.get(base + ".weight_scale_2", kF, dev);
    q.has_scale = q.has_scale2 = true;
  } else if (dt == torch::kFloat8_e4m3fn) {
    q.fmt = QFormat::FP8;
    q.w = coll.get(wn, torch::kFloat8_e4m3fn, dev);
    q.scale2 = coll.get(base + ".weight_scale", kF, dev);
    q.has_scale2 = true;
  } else {
    q.fmt = QFormat::Dense;
    q.w = coll.get(wn, dense_dtype, dev);
    return q;
  }
  if (coll.has(base + ".input_scale")) { q.input_scale = coll.get(base + ".input_scale", kF, dev); q.has_input_scale = true; }
  return q;
}

namespace {
QWeight load_experts(const utils::safetensors::SafetensorsCollection& coll, const std::string& prefix, const std::string& which, int64_t E, torch::Device dev,
                     torch::ScalarType dd) {
  if (!coll.has(prefix + ".0." + which + ".weight")) {  // HF-stacked layout: experts.up_proj [E,out,in]
    QWeight q;
    q.present = true;
    q.w = coll.get(prefix + "." + which, dd, dev);
    return q;
  }
  std::vector<QWeight> v;
  for (int64_t e = 0; e < E; ++e) v.push_back(load_qweight(coll, prefix + "." + std::to_string(e) + "." + which, dev, dd));
  QWeight q;
  q.present = true; q.has_scale = v[0].has_scale; q.has_scale2 = v[0].has_scale2; q.has_input_scale = v[0].has_input_scale;
  q.fmt = v[0].fmt;
  auto stack = [&](Tensor QWeight::*m) {
    if (!(v[0].*m).defined()) return Tensor();
    std::vector<Tensor> t;
    for (auto& x : v) t.push_back(x.*m);
    return at::stack(t, 0);
  };
  q.w = stack(&QWeight::w); q.scale = stack(&QWeight::scale); q.scale2 = stack(&QWeight::scale2); q.input_scale = stack(&QWeight::input_scale);
  return q;
}
}  // namespace

Model Model::Load(const std::string& dir, torch::Device device, const LoadOpts& opts) {
  Model m;
  m.cfg = Config::FromFile(dir + "/config.json");
  if (opts.hf_dt_clamp) m.cfg.dt_clamp_min = m.cfg.time_step_min;
  m.device = device;
  auto coll = utils::safetensors::SafetensorsCollection::FromDirectory(dir);
  std::string pre = "backbone.", lmpre = "";
  if (coll.has("language_model.backbone.embeddings.weight")) { pre = "language_model.backbone."; lmpre = "language_model."; }
  else if (coll.has("model.embeddings.weight")) { pre = "model."; }
  m.ckpt_prefix = pre;
  const auto dd = opts.dense_dtype;
  auto W = [&](const std::string& n) { return coll.get(n, dd, device); };
  if (opts.load_embed) m.embed = W(pre + "embeddings.weight");
  if (opts.load_head) {
    m.final_norm = W(pre + "norm_f.weight");
    if (m.cfg.tie_word_embeddings) { m.lm_head.q.w = m.embed; }
    else m.lm_head.q = load_qweight(coll, lmpre + "lm_head", device, dd);
  }
  m.lm_head.name = "lm_head";
  const int64_t n = static_cast<int64_t>(m.cfg.blocks.size());
  m.layers.resize(n);
  for (int64_t i = 0; i < n; ++i) {
    Layer& L = m.layers[i];
    L.type = m.cfg.blocks[i];
    if (!opts.only_layers.empty() && std::find(opts.only_layers.begin(), opts.only_layers.end(), i) == opts.only_layers.end()) continue;   // subset load: type only, no weights
    const std::string p = pre + "layers." + std::to_string(i) + ".";
    const std::string mx = p + "mixer.";
    L.norm = W(p + "norm.weight");
    auto mk = [&](Linear& l, const std::string& sub, Shard sh) {
      l.q = load_qweight(coll, mx + sub, device, dd);
      l.name = idx_name(i, sub);
      l.shard = sh;
    };
    switch (L.type) {
      case BlockType::Mamba:
        mk(L.in_proj, "in_proj", Shard::Column); mk(L.out_proj, "out_proj", Shard::Row);
        L.conv_w = W(mx + "conv1d.weight");
        if (coll.has(mx + "conv1d.bias")) { L.conv_b = W(mx + "conv1d.bias"); L.has_conv_b = true; }
        L.A_log = W(mx + "A_log"); L.D = W(mx + "D"); L.dt_bias = W(mx + "dt_bias"); L.gnorm = W(mx + "norm.weight");
        break;
      case BlockType::Attention:
        mk(L.q_proj, "q_proj", Shard::Column); mk(L.k_proj, "k_proj", Shard::Column); mk(L.v_proj, "v_proj", Shard::Column); mk(L.o_proj, "o_proj", Shard::Row);
        break;
      case BlockType::MoE:
        L.gate_w = W(mx + "gate.weight");
        L.gate_bias = coll.get(mx + "gate.e_score_correction_bias", kF, device);
        if (m.cfg.moe_latent_size > 0) { mk(L.fc1, "fc1_latent_proj", Shard::Replicated); mk(L.fc2, "fc2_latent_proj", Shard::Replicated); }
        mk(L.sh_up, "shared_experts.up_proj", Shard::Column); mk(L.sh_down, "shared_experts.down_proj", Shard::Row);
        L.up = load_experts(coll, mx + "experts", "up_proj", m.cfg.n_routed_experts, device, dd);
        L.down = load_experts(coll, mx + "experts", "down_proj", m.cfg.n_routed_experts, device, dd);
        break;
      case BlockType::MLP:
        mk(L.sh_up, "up_proj", Shard::Column); mk(L.sh_down, "down_proj", Shard::Row);
        break;
    }
  }
  return m;
}

std::vector<Linear*> Model::lora_sites(const LoraTargets& t) {
  std::vector<Linear*> v;
  for (auto& L : layers) {
    if (!L.norm.defined()) continue;   // layer not loaded (subset load)
    switch (L.type) {
      case BlockType::Mamba: if (t.mamba) v.insert(v.end(), {&L.in_proj, &L.out_proj}); break;
      case BlockType::Attention: if (t.attention) v.insert(v.end(), {&L.q_proj, &L.k_proj, &L.v_proj, &L.o_proj}); break;
      case BlockType::MoE:
        if (t.moe_latent && L.fc1.defined()) v.insert(v.end(), {&L.fc1, &L.fc2});
        if (t.moe_shared) v.insert(v.end(), {&L.sh_up, &L.sh_down});
        break;
      case BlockType::MLP: if (t.mlp) v.insert(v.end(), {&L.sh_up, &L.sh_down}); break;
    }
  }
  return v;
}

void Model::attach_lora(int64_t rank, double alpha, uint64_t seed, const utils::safetensors::SafetensorsCollection* init, const LoraTargets& t, torch::ScalarType ld) {
  auto gen = at::detail::createCPUGenerator(seed);
  auto make = [&](const std::string& name, int64_t out, int64_t in, Lora& lo) {
    Tensor A, B;
    if (init && init->has("lora_init." + name + ".A")) {
      A = init->get("lora_init." + name + ".A", ld, device);
      B = init->get("lora_init." + name + ".B", ld, device);
    } else {
      const double bound = 1.0 / std::sqrt(static_cast<double>(in));
      A = (at::rand({rank, in}, gen, torch::TensorOptions().dtype(ld)) * 2 - 1).mul(bound).to(device);
      B = at::zeros({out, rank}, torch::TensorOptions().dtype(ld).device(device));
    }
    TORCH_CHECK(A.size(0) == rank && A.size(1) == in && B.size(0) == out && B.size(1) == rank, "nemotron_h: bad LoRA shape for ", name);
    lo.A = A.contiguous().detach().requires_grad_(true);
    lo.B = B.contiguous().detach().requires_grad_(true);
    lo.scale = alpha / static_cast<double>(rank);
  };
  for (Linear* l : lora_sites(t)) make(l->name, l->q.out_features(), l->q.in_features(), l->lora);
  if (t.router_gate)
    for (size_t i = 0; i < layers.size(); ++i)
      if (layers[i].type == BlockType::MoE && layers[i].gate_w.defined()) make(idx_name(i, "gate"), layers[i].gate_w.size(0), layers[i].gate_w.size(1), layers[i].gate_lora);
}

std::vector<std::pair<std::string, Tensor>> Model::named_lora_params() {
  std::vector<std::pair<std::string, Tensor>> v;
  for (size_t i = 0; i < layers.size(); ++i) {
    Layer& L = layers[i];
    std::vector<std::pair<std::string, Lora*>> ls;
    switch (L.type) {
      case BlockType::Mamba: ls = {{L.in_proj.name, &L.in_proj.lora}, {L.out_proj.name, &L.out_proj.lora}}; break;
      case BlockType::Attention: ls = {{L.q_proj.name, &L.q_proj.lora}, {L.k_proj.name, &L.k_proj.lora}, {L.v_proj.name, &L.v_proj.lora}, {L.o_proj.name, &L.o_proj.lora}}; break;
      case BlockType::MoE:
        ls = {{L.fc1.name, &L.fc1.lora}, {L.fc2.name, &L.fc2.lora}, {L.sh_up.name, &L.sh_up.lora}, {L.sh_down.name, &L.sh_down.lora}, {idx_name(i, "gate"), &L.gate_lora}};
        break;
      case BlockType::MLP: ls = {{L.sh_up.name, &L.sh_up.lora}, {L.sh_down.name, &L.sh_down.lora}}; break;
    }
    for (auto& [nm, lo] : ls)
      if (lo->defined()) { v.emplace_back(nm + ".A", lo->A); v.emplace_back(nm + ".B", lo->B); }
  }
  return v;
}

std::vector<WeightSlot> Model::base_slots() {
  std::vector<WeightSlot> v;
  const int64_t n = static_cast<int64_t>(layers.size());
  auto addq = [&](const std::string& name, QWeight& q, int g) {
    if (!q.slot_w()) return;   // structure comes from what the loader recorded, so the list is the same before and after the slots are emptied (park, spill level 2)
    v.push_back({name + ".weight", &q.w, static_cast<int>(q.fmt), g});
    if (q.slot_scale()) v.push_back({name + ".weight_scale", &q.scale, 5, g});
    if (q.slot_scale2()) v.push_back({name + (q.fmt == QFormat::NVFP4 ? ".weight_scale_2" : ".weight_scale"), &q.scale2, 6, g});
    if (q.slot_input_scale()) v.push_back({name + ".input_scale", &q.input_scale, 6, g});
  };
  v.push_back({"embeddings.weight", &embed, 0, 0});
  for (int64_t i = 0; i < n; ++i) {
    Layer& L = layers[i];
    const int g = static_cast<int>(1 + i);
    const std::string p = "layers." + std::to_string(i) + ".";
    v.push_back({p + "norm.weight", &L.norm, 0, g});
    auto addl = [&](Linear& l) { addq(l.name, l.q, g); };
    switch (L.type) {
      case BlockType::Mamba:
        addl(L.in_proj); addl(L.out_proj);
        v.push_back({p + "mixer.conv1d.weight", &L.conv_w, 0, g});
        if (L.has_conv_b || L.conv_b.defined()) v.push_back({p + "mixer.conv1d.bias", &L.conv_b, 0, g});
        v.push_back({p + "mixer.A_log", &L.A_log, 0, g}); v.push_back({p + "mixer.D", &L.D, 0, g});
        v.push_back({p + "mixer.dt_bias", &L.dt_bias, 0, g}); v.push_back({p + "mixer.norm.weight", &L.gnorm, 0, g});
        break;
      case BlockType::Attention: addl(L.q_proj); addl(L.k_proj); addl(L.v_proj); addl(L.o_proj); break;
      case BlockType::MoE:
        v.push_back({p + "mixer.gate.weight", &L.gate_w, 0, g}); v.push_back({p + "mixer.gate.e_score_correction_bias", &L.gate_bias, 0, g});
        if (L.fc1.q.slot_w()) { addl(L.fc1); addl(L.fc2); }
        addl(L.sh_up); addl(L.sh_down);
        addq(p + "mixer.experts.up_proj", L.up, g); addq(p + "mixer.experts.down_proj", L.down, g);  // stacked [E, ...]
        break;
      case BlockType::MLP: addl(L.sh_up); addl(L.sh_down); break;
    }
  }
  v.push_back({"norm_f.weight", &final_norm, 0, static_cast<int>(n + 1)});
  addq("lm_head", lm_head.q, static_cast<int>(n + 1));
  return v;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------------------------------------------------------------------------
Tensor Model::embed_tokens(const Tensor& ids, Compute c) const { return at::embedding(embed, ids).to(compute_dtype(c)); }

Tensor Model::forward_hidden(const Tensor& embeds, const ForwardOptions& opt) {
  // the checkpoint lambda runs again inside backward(): it must not point into the caller's (by then dead) options
  auto optp = std::make_shared<ForwardOptions>(opt);
  optp->probe = nullptr;
  optp->state_in = nullptr;
  optp->state_out = nullptr;
  Ctx c{compute_dtype(opt.compute), stable_dt(compute_dtype(opt.compute)), &opt, &cfg};
  Ctx cck{c.adt, c.sdt, optp.get(), &cfg};
  TORCH_CHECK(!(opt.checkpoint && (opt.state_in || opt.state_out)), "nemotron_h: checkpoint=true cannot be combined with state_in/state_out");
  Probe* pr = opt.probe;
  Tensor h = embeds.to(c.adt);
  if (opt.state_out) opt.state_out->assign(layers.size(), LayerState{});
  for (size_t i = 0; i < layers.size(); ++i) {
    const Layer* Lp = &layers[i];
    if (opt.checkpoint) {
      Tensor hin = h.requires_grad() ? h : h.detach().requires_grad_(true);
      h = torch::utils::checkpoint([Lp, cck, optp](std::vector<Tensor> in) { return std::vector<Tensor>{layer_fwd(*Lp, in[0], cck, nullptr, nullptr)}; }, {hin}, false)[0];
    } else {
      const LayerState* sin = (opt.state_in && i < opt.state_in->size()) ? &(*opt.state_in)[i] : nullptr;
      h = layer_fwd(*Lp, h, c, sin, opt.state_out ? &(*opt.state_out)[i] : nullptr);
    }
    if (pr && pr->acts) pr->act["layer." + std::to_string(i)] = h.detach().to(c.sdt).cpu();
    if (pr && pr->grads && h.requires_grad())
      h.register_hook([pr, i, sdt = c.sdt](const Tensor& g) { pr->grad["layer." + std::to_string(i)] = g.detach().to(sdt).cpu(); });
  }
  h = rms(h, final_norm, cfg.norm_eps);
  if (pr && pr->acts) pr->act["final_norm"] = h.detach().to(c.sdt).cpu();
  return h;
}

Tensor Model::lm_logits(const Tensor& hn, const ForwardOptions& opt) {
  Ctx c{compute_dtype(opt.compute), stable_dt(compute_dtype(opt.compute)), &opt, &cfg};
  return lin(hn, lm_head, c).to(c.sdt);
}

Tensor Model::forward_embeds(const Tensor& embeds, const ForwardOptions& opt) { return lm_logits(forward_hidden(embeds, opt), opt); }

// ---- block-level API (packed [T, hidden] layout)
Tensor Model::norm_fwd(int64_t block, const Tensor& x, Compute) const { return rms(x, layers.at(block).norm, cfg.norm_eps); }

Tensor Model::mixer_fwd(int64_t block, const Tensor& h, const SeqLayout& lay, const ForwardOptions& opt) const {
  Ctx c{compute_dtype(opt.compute), stable_dt(compute_dtype(opt.compute)), &opt, &cfg};
  TORCH_CHECK(layer_loaded(block), "nemotron_h: block ", block, " is not loaded");
  return mixer_packed(layers[block], h, lay, c);
}

Tensor Model::block_fwd(int64_t block, const Tensor& x, const SeqLayout& lay, const ForwardOptions& opt) const {
  return x + mixer_fwd(block, norm_fwd(block, x, opt.compute), lay, opt);
}

Tensor Model::final_norm_fwd(const Tensor& h, Compute) const { return rms(h, final_norm, cfg.norm_eps); }

Tensor Model::head_logprob(const Tensor& hn, const Tensor& tgt, const ForwardOptions& opt) const {
  Ctx c{compute_dtype(opt.compute), stable_dt(compute_dtype(opt.compute)), &opt, &cfg};
  Tensor logits = lin(hn, lm_head, c).to(c.sdt);   // bf16 mode: bf16 GEMM, then fp32 (as HF); the [C, V] chunk is the only logits tensor that ever exists
  return logits.gather(1, tgt.unsqueeze(1)).squeeze(1) - at::logsumexp(logits, 1);
}

Tensor Model::forward(const Tensor& ids, const ForwardOptions& opt) { return forward_embeds(embed_tokens(ids, opt.compute), opt); }

}  // namespace torch::models::nemotron_h
