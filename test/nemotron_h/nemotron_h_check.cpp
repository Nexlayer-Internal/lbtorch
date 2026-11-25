// CPU gradient gate for torch::models::nemotron_h against HF autograd fixtures (test/nemotron_h/gen_fixture.py).
// usage: nemotron_h_check <fixture_dir> [out.json] [real_checkpoint_dir]        (CPU only; run with CUDA_VISIBLE_DEVICES="")
#include <torch/models/nemotron_h.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>

using namespace torch::models::nemotron_h;
using torch::utils::safetensors::SafetensorsCollection;
using torch::Tensor;
static constexpr auto kF = torch::kFloat32;
static constexpr auto kD = torch::kFloat64;

struct Row { std::string group, name; double mabs = 0, mrel = 0, l2 = 0, tol = 0; bool ok = true; std::string note; bool skipped = false; };
static std::vector<Row> ROWS;
static int FAILS = 0;

static void stats(const Tensor& a_, const Tensor& b_, double& mabs, double& mrel, double& l2) {
  Tensor a = a_.to(torch::kCPU).to(kD).flatten(), b = b_.to(torch::kCPU).to(kD).flatten();
  mabs = (a - b).abs().max().item<double>();
  double den = std::max(b.abs().max().item<double>(), 1e-300);
  mrel = mabs / den;
  l2 = (a - b).norm().item<double>() / std::max(b.norm().item<double>(), 1e-300);
}
static void check(const std::string& group, const std::string& name, const Tensor& got, const Tensor& ref, double tol, const std::string& note = "") {
  Row r; r.group = group; r.name = name; r.tol = tol; r.note = note;
  if (got.sizes() != ref.sizes()) { r.ok = false; r.note = "SHAPE MISMATCH"; r.mabs = r.mrel = r.l2 = NAN; }
  else {
    stats(got, ref, r.mabs, r.mrel, r.l2);
    r.ok = std::isfinite(r.mrel) && r.mrel <= tol;
  }
  if (!r.ok) ++FAILS;
  std::printf("  [%s] %-34s %-46s max_abs %9.3e max_rel %9.3e relL2 %9.3e (tol %g) %s\n", r.ok ? "ok" : "FAIL", group.c_str(), name.c_str(), r.mabs, r.mrel, r.l2, tol, note.c_str());
  ROWS.push_back(r);
}
static void flag(const std::string& group, const std::string& name, bool ok, const std::string& note = "") {
  Row r; r.group = group; r.name = name; r.ok = ok; r.note = note;
  if (!ok) ++FAILS;
  std::printf("  [%s] %-34s %-46s %s\n", ok ? "ok" : "FAIL", group.c_str(), name.c_str(), note.c_str());
  ROWS.push_back(r);
}
static void skip(const std::string& group, const std::string& name, const std::string& why) {
  Row r; r.group = group; r.name = name; r.skipped = true; r.note = "SKIPPED: " + why;
  std::printf("  [skip] %-34s %-46s %s\n", group.c_str(), name.c_str(), why.c_str());
  ROWS.push_back(r);
}

static std::string kind_of(const Model& m, const std::string& lora_name) {  // "layers.N.mixer.<site>" -> block type / site class
  std::smatch mm;
  std::regex re("layers\\.(\\d+)\\.mixer\\.(.*)");
  if (!std::regex_match(lora_name, mm, re)) return "other";
  const int i = std::stoi(mm[1]);
  const std::string s = mm[2];
  switch (m.layers[i].type) {
    case BlockType::Mamba: return "mamba2";
    case BlockType::Attention: return "attention";
    case BlockType::MLP: return "mlp";
    case BlockType::MoE: return s == "gate" ? "moe router gate" : (s.find("shared") != std::string::npos ? "moe shared expert" : "moe latent proj");
  }
  return "other";
}

struct Run { Tensor logits, d_emb, emb; std::map<std::string, Tensor> grads, layer_out; double loss = 0; };

// forward + backward of loss = sum(logits * R); leaves' .grad are accumulated (caller zeroes)
static void zero_grads(Model& m) { for (auto& p : m.named_lora_params()) if (p.second.grad().defined()) p.second.mutable_grad().reset(); }

static Run run_model(Model& m, const Tensor& ids, const Tensor& R, ForwardOptions fo, Tensor* embeds_in = nullptr) {
  zero_grads(m);
  Run r;
  Tensor emb = embeds_in ? embeds_in->detach().clone() : m.embed_tokens(ids, fo.compute).detach().clone();
  emb.requires_grad_(true);
  Probe pr; pr.grads = false;
  fo.probe = &pr;
  Tensor logits = m.forward_embeds(emb, fo);
  Tensor loss = (logits * R.to(logits.scalar_type())).sum();
  loss.backward();
  r.emb = emb.detach(); r.logits = logits.detach(); r.d_emb = emb.grad().detach(); r.loss = loss.item<double>();
  for (auto& p : m.named_lora_params()) r.grads[p.first] = p.second.grad().defined() ? p.second.grad().detach().clone() : torch::zeros_like(p.second);
  for (auto& kv : pr.act) r.layer_out[kv.first] = kv.second;
  return r;
}

// compare against an HF oracle file; per-group aggregation for the LoRA gradients
static void compare_oracle(const std::string& tag, Model& m, const Run& r, const SafetensorsCollection& orc, double tol_fwd, double tol_grad) {
  check(tag, "embedding output", r.emb, orc.get("emb", kD), tol_fwd);
  check(tag, "logits", r.logits, orc.get("logits", kD), tol_fwd);
  for (size_t i = 0; i < m.layers.size(); ++i) {
    static const char* nm[] = {"mamba2", "attention", "moe", "mlp"};
    const std::string key = "layer." + std::to_string(i);
    check(tag, "block output " + key + " (" + nm[(int)m.layers[i].type] + ")", r.layer_out.at(key), orc.get(key, kD), tol_fwd);
  }
  check(tag, "d(inputs_embeds)", r.d_emb, orc.get("d_emb", kD), tol_grad);
  std::map<std::string, std::array<double, 4>> agg;  // group -> max abs, max rel, max relL2, count
  for (auto& [name, g] : r.grads) {
    const std::string base = name.substr(0, name.size() - 2);
    const char ab = name.back() == 'A' ? 'A' : 'B';
    Tensor ref = orc.get("grad." + base + "." + std::string(1, ab), kD);
    double a, rl, l2; stats(g, ref, a, rl, l2);
    auto& s = agg[kind_of(m, base)];
    s[0] = std::max(s[0], a); s[1] = std::max(s[1], rl); s[2] = std::max(s[2], l2); s[3] += 1;
    if (!(std::isfinite(rl) && rl <= tol_grad)) { std::printf("    worst offender %s: max_rel %g\n", name.c_str(), rl); }
  }
  for (auto& [k, s] : agg) {
    Row row; row.group = tag; row.name = "LoRA grads dA,dB: " + k + " (" + std::to_string((int)s[3]) + " tensors)"; row.mabs = s[0]; row.mrel = s[1]; row.l2 = s[2]; row.tol = tol_grad;
    row.ok = std::isfinite(s[1]) && s[1] <= tol_grad;
    if (!row.ok) ++FAILS;
    std::printf("  [%s] %-34s %-46s max_abs %9.3e max_rel %9.3e relL2 %9.3e (tol %g)\n", row.ok ? "ok" : "FAIL", tag.c_str(), row.name.c_str(), row.mabs, row.mrel, row.l2, row.tol);
    ROWS.push_back(row);
  }
}

static void compare_runs(const std::string& tag, const std::string& what, const Run& a, const Run& b, double tol) {
  check(tag, what + ": logits", a.logits, b.logits, tol);
  check(tag, what + ": d(inputs_embeds)", a.d_emb, b.d_emb, tol);
  double wa = 0, wr = 0;
  for (auto& [n, g] : a.grads) { double x, y, z; stats(g, b.grads.at(n), x, y, z); wa = std::max(wa, x); wr = std::max(wr, y); }
  Row row; row.group = tag; row.name = what + ": all LoRA grads (" + std::to_string(a.grads.size()) + " tensors)"; row.mabs = wa; row.mrel = wr; row.tol = tol; row.ok = wr <= tol;
  if (!row.ok) ++FAILS;
  std::printf("  [%s] %-34s %-46s max_abs %9.3e max_rel %9.3e (tol %g)\n", row.ok ? "ok" : "FAIL", tag.c_str(), row.name.c_str(), wa, wr, tol);
  ROWS.push_back(row);
}

static std::unique_ptr<Model> load_with_lora(const std::string& dir, Compute c, torch::ScalarType dense = kF) {
  auto m = std::make_unique<Model>(Model::Load(dir, torch::kCPU, LoadOpts{dense, true}));
  auto init = SafetensorsCollection::FromFile(dir + "/lora_init.safetensors");
  LoraTargets t; t.router_gate = true;
  m->attach_lora(4, 8.0, 0, &init, t, c == Compute::FP64 ? kD : kF);
  return m;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: nemotron_h_check <fixture_dir> [out.json] [real_checkpoint_dir]\n"; return 2; }
  const std::string fx = argv[1], outp = argc > 2 ? argv[2] : "", real = argc > 3 ? argv[3] : "";
  torch::set_num_threads(4);
  auto meta = nlohmann::json::parse(std::ifstream(fx + "/meta.json"));
  const int64_t TP = meta.at("Tp");
  auto dense_dir = fx + "/tiny", q_dir = fx + "/tinyq";

  // ----------------------------------------------------------------------------------------------------------------- config of the real checkpoint
  if (!real.empty() && std::ifstream(real + "/config.json")) {
    std::cout << "== config.json of " << real << "\n";
    Config c = Config::FromFile(real + "/config.json");
    int nm = 0, na = 0, ne = 0, nl = 0;
    for (auto b : c.blocks) { nm += b == BlockType::Mamba; na += b == BlockType::Attention; ne += b == BlockType::MoE; nl += b == BlockType::MLP; }
    flag("config", "Super-VL llm_config parsed", c.hidden_size == 4096 && nm == 40 && ne == 40 && na == 8 && nl == 0 && c.n_routed_experts == 512 && c.num_experts_per_tok == 22 &&
                                                   c.moe_latent_size == 1024 && c.mamba_num_heads == 128 && c.ssm_state_size == 128 && c.n_groups == 8 && c.d_inner() == 8192 && c.conv_dim() == 10240,
         "hidden " + std::to_string(c.hidden_size) + ", blocks mamba/moe/attention/mlp = " + std::to_string(nm) + "/" + std::to_string(ne) + "/" + std::to_string(na) + "/" + std::to_string(nl) +
             ", experts " + std::to_string(c.n_routed_experts) + " top " + std::to_string(c.num_experts_per_tok) + ", dt_clamp_min " + std::to_string(c.dt_clamp_min));
  }

  // ----------------------------------------------------------------------------------------------------------------- dense model vs HF autograd
  for (const std::string tag : {"fp64", "fp32"}) {
    const Compute c = tag == "fp64" ? Compute::FP64 : Compute::FP32;
    std::cout << "== dense tiny stack (mamba, moe, attention, mlp, mamba, moe), compute " << tag << " vs HF autograd\n";
    auto m = load_with_lora(dense_dir, c);
    auto orc = SafetensorsCollection::FromFile(dense_dir + "/oracle_" + tag + ".safetensors");
    Tensor ids = orc.get("ids", torch::kInt64), R = orc.get("R", c == Compute::FP64 ? kD : kF);
    ForwardOptions fo; fo.compute = c;
    Run r = run_model(*m, ids, R, fo);
    compare_oracle("dense/" + tag, *m, r, orc, c == Compute::FP64 ? 1e-10 : 3e-4, c == Compute::FP64 ? 1e-9 : 3e-3);
  }

  // ----------------------------------------------------------------------------------------------------------------- invariances and alternate paths (fp64)
  {
    std::cout << "== invariances, fp64\n";
    auto m = load_with_lora(dense_dir, Compute::FP64);
    auto orc = SafetensorsCollection::FromFile(dense_dir + "/oracle_fp64.safetensors");
    Tensor ids = orc.get("ids", torch::kInt64), R = orc.get("R", kD);
    ForwardOptions fo; fo.compute = Compute::FP64;
    Run joint = run_model(*m, ids, R, fo);
    // micro-batch invariance: rows one at a time, gradients accumulated by autograd
    {
      zero_grads(*m);
      std::vector<Tensor> lg;
      for (int64_t b = 0; b < ids.size(0); ++b) {
        Tensor emb = m->embed_tokens(ids.narrow(0, b, 1), Compute::FP64).detach().requires_grad_(true);
        Tensor logits = m->forward_embeds(emb, fo);
        (logits * R.narrow(0, b, 1)).sum().backward();
        lg.push_back(logits.detach());
      }
      Run split; split.logits = at::cat(lg, 0); split.d_emb = joint.d_emb;
      for (auto& p : m->named_lora_params()) split.grads[p.first] = p.second.grad().detach().clone();
      compare_runs("invariance", "micro-batch (B=2 vs two B=1 backward calls)", split, joint, 1e-10);
    }
    // prefix hand-off: prompt then response with graph-connected state, vs the joint sequence
    {
      zero_grads(*m);
      Tensor emb = m->embed_tokens(ids, Compute::FP64).detach();
      Tensor ep = emb.narrow(1, 0, TP).clone().requires_grad_(true), er = emb.narrow(1, TP, ids.size(1) - TP).clone().requires_grad_(true);
      std::vector<LayerState> st;
      ForwardOptions fp = fo; fp.state_out = &st;
      Tensor lp = m->forward_embeds(ep, fp);
      ForwardOptions fr = fo; fr.state_in = &st;
      Tensor lr = m->forward_embeds(er, fr);
      ((lp * R.narrow(1, 0, TP)).sum() + (lr * R.narrow(1, TP, ids.size(1) - TP)).sum()).backward();
      Run split; split.logits = at::cat({lp.detach(), lr.detach()}, 1); split.d_emb = at::cat({ep.grad(), er.grad()}, 1);
      for (auto& p : m->named_lora_params()) split.grads[p.first] = p.second.grad().detach().clone();
      compare_runs("invariance", "prefix hand-off (prompt " + std::to_string(TP) + " | response, state carried)", split, joint, 1e-10);
      int ns = 0, nk = 0;
      for (auto& s : st) { ns += s.ssm.defined(); nk += s.k.defined(); }
      flag("invariance", "hand-off state present", ns == 2 && nk == 1, std::to_string(ns) + " mamba states, " + std::to_string(nk) + " attention KV");
    }
    // serial recurrence as the SSD (independent of the chunked math) through the SSD hook
    {
      ForwardOptions fs = fo;
      int calls = 0;
      fs.ssd = [&](const Tensor& x, const Tensor& dt, const Tensor& A, const Tensor& B, const Tensor& C, const Tensor& D, const Tensor& h0, int64_t ch, const Tensor& cu) { ++calls; return ssd_serial(x, dt, A, B, C, D, h0, ch, cu); };
      Run s = run_model(*m, ids, R, fs);
      compare_runs("invariance", "SSD hook = token-by-token recurrence vs chunked", s, joint, 1e-9);
      flag("invariance", "SSD hook invoked once per mamba block", calls == 2, std::to_string(calls) + " calls");
    }
    // checkpointing is exact
    {
      ForwardOptions fc = fo; fc.checkpoint = true;
      Run s = run_model(*m, ids, R, fc);
      compare_runs("invariance", "per-layer checkpointing vs plain", s, joint, 1e-12);
    }
    // grouped MoE GEMM vs per-expert loop (fp32; at::_grouped_mm)
    {
      auto mf = load_with_lora(dense_dir, Compute::FP32);
      auto orc32 = SafetensorsCollection::FromFile(dense_dir + "/oracle_fp32.safetensors");
      Tensor R32 = orc32.get("R", kF);
      ForwardOptions f32; f32.compute = Compute::FP32;
      Run loop = run_model(*mf, ids, R32, f32);
      try {
        ForwardOptions fg = f32; fg.moe = MoeImpl::GroupedMM;
        Run g = run_model(*mf, ids, R32, fg);
        compare_runs("invariance", "MoeImpl::GroupedMM vs per-expert loop (fp32, CPU at::_grouped_mm)", g, loop, 1e-4);
      } catch (const std::exception& e) {
        skip("invariance", "MoeImpl::GroupedMM", std::string("at::_grouped_mm failed on CPU: ") + std::string(e.what()).substr(0, 120));
      }
    }
  }

  // ----------------------------------------------------------------------------------------------------------------- quantised base
  {
    std::cout << "== frozen quantised base (FP8 mamba/attention/lm_head, NVFP4 experts/shared/mlp)\n";
    auto coll = SafetensorsCollection::FromFile(q_dir + "/model.safetensors");
    auto qd = SafetensorsCollection::FromFile(q_dir + "/qdense.safetensors");
    int n8 = 0, n4 = 0; bool exact64 = true, exact32 = true; double worst = 0;
    for (const auto& base : qd.names()) {
      QWeight q = load_qweight(coll, base, torch::kCPU, kF);
      (q.fmt == QFormat::FP8 ? n8 : n4) += 1;
      Tensor ref = qd.get(base, kD);
      exact64 = exact64 && torch::equal(q.dequant(kD), ref);
      exact32 = exact32 && torch::equal(q.dequant(kF), ref.to(kF));
      worst = std::max(worst, (q.dequant(kD) - ref).abs().max().item<double>());
    }
    flag("quant", "dequant == numpy fp64 dequant (fp64, bit-exact)", exact64, std::to_string(n8) + " FP8 + " + std::to_string(n4) + " NVFP4 weights checked, max |diff| " + std::to_string(worst));
    flag("quant", "dequant == numpy fp64 dequant cast to fp32 (bit-exact)", exact32);
    auto orc = SafetensorsCollection::FromFile(q_dir + "/oracle_fp64.safetensors");
    Tensor ids = orc.get("ids", torch::kInt64), R = orc.get("R", kD);
    {
      auto m = load_with_lora(q_dir, Compute::FP64);
      int f8 = 0, f4 = 0;
      for (auto& s : m->base_slots()) { f8 += s.format == 3; f4 += s.format == 4; }
      flag("quant", "base_slots format tags", f8 > 0 && f4 > 0, std::to_string(f8) + " fp8 payloads, " + std::to_string(f4) + " nvfp4 payloads");
      ForwardOptions fo; fo.compute = Compute::FP64;
      Run r = run_model(*m, ids, R, fo);
      compare_oracle("quant/fp64", *m, r, orc, 1e-10, 1e-9);
      // per-expert loop over stacked packed experts: same layer, dequantised per use (no grouped path for packed weights)
      ForwardOptions fg = fo; fg.moe = MoeImpl::GroupedMM;
      bool threw = false;
      try { run_model(*m, ids, R, fg); } catch (const std::exception&) { threw = true; }
      flag("quant", "GroupedMM refuses packed experts (v1: per-expert dequant loop)", threw);
    }
    {
      auto m = load_with_lora(q_dir, Compute::FP32);
      ForwardOptions fo; fo.compute = Compute::FP32;
      Run r = run_model(*m, ids, orc.get("R", kF), fo);
      compare_oracle("quant/fp32", *m, r, orc, 3e-4, 3e-3);
    }
    {  // W8A8 / W4A4 activation quantisation with the straight-through estimator (informational: no HF oracle for it)
      auto m = load_with_lora(q_dir, Compute::FP64);
      ForwardOptions fo; fo.compute = Compute::FP64;
      Run base = run_model(*m, ids, R, fo);
      ForwardOptions fa = fo; fa.act_quant = true;
      Run aq = run_model(*m, ids, R, fa);
      double dot = 0, na = 0, nb = 0; bool fin = true;
      for (auto& [n, g] : aq.grads) { Tensor a = g.flatten(), b = base.grads.at(n).flatten(); dot += (a * b).sum().item<double>(); na += (a * a).sum().item<double>(); nb += (b * b).sum().item<double>(); fin = fin && torch::isfinite(g).all().item<bool>(); }
      flag("quant", "act_quant=true (STE) end-to-end runs, grads finite", fin, "cosine(LoRA grads with vs without activation quantisation) = " + std::to_string(dot / std::sqrt(na * nb)));
    }
    {  // quantiser vectors from the numpy reference (nemotron-h-reference/quant_linear.py)
      auto aq = SafetensorsCollection::FromFile(q_dir + "/actq.safetensors");
      Tensor x = aq.get("x", kD);
      for (const std::string k : {"fp8", "fp4"}) {
        QWeight w;
        w.fmt = k == "fp8" ? QFormat::FP8 : QFormat::NVFP4;
        w.input_scale = aq.get("s_" + k, kF);
        Tensor xl = x.clone().requires_grad_(true);
        Tensor xq = act_quant_ste(xl, w);
        xq.sum().backward();
        Tensor ref = aq.get("xq_" + k, kD), mask = aq.get("m_" + k, kD);
        check("quant", "activation quantiser value (" + k + ") vs numpy", xq.detach(), ref, 1e-7);
        check("quant", "STE gradient (" + k + ") == saturation mask", xl.grad(), mask, 0);
        flag("quant", "saturation exercised (" + k + ")", mask.min().item<double>() == 0.0 && mask.max().item<double>() == 1.0, std::to_string(1.0 - mask.mean().item<double>()) + " of entries saturated");
      }
    }
  }

  // ----------------------------------------------------------------------------------------------------------------- real checkpoint bytes
  if (!real.empty() && std::ifstream(real + "/model.safetensors.index.json")) {
    std::cout << "== real checkpoint tensors (read-only, CPU)\n";
    auto coll = SafetensorsCollection::FromDirectory(real);
    QWeight a = load_qweight(coll, "language_model.backbone.layers.1.mixer.shared_experts.up_proj", torch::kCPU, kF);
    QWeight b = load_qweight(coll, "language_model.backbone.layers.0.mixer.in_proj", torch::kCPU, kF);
    Tensor da = a.dequant(kF).narrow(0, 0, 64), db = b.dequant(kF).narrow(0, 0, 64);
    auto rms = [](const Tensor& t) { return std::sqrt((t.to(kD).square().mean()).item<double>()); };
    // reference numbers from the numpy dequant of the same rows (nemotron-h-reference README): rms 0.007148 / 0.01764
    flag("real", "NVFP4 shared up_proj rows dequantise as numpy does", a.fmt == QFormat::NVFP4 && std::fabs(rms(da) - 0.007148) < 2e-6, "rms " + std::to_string(rms(da)) + ", max " + std::to_string(da.abs().max().item<double>()));
    flag("real", "FP8 in_proj rows dequantise as numpy does", b.fmt == QFormat::FP8 && std::fabs(rms(db) - 0.01764) < 2e-5, "rms " + std::to_string(rms(db)));
  }

  int nskip = 0;
  for (auto& r : ROWS) nskip += r.skipped;
  std::printf("\n%zu checks, %d failed, %d skipped\n", ROWS.size(), FAILS, nskip);
  if (!outp.empty()) {
    nlohmann::json j = nlohmann::json::array();
    for (auto& r : ROWS) j.push_back({{"group", r.group}, {"check", r.name}, {"max_abs", std::isnan(r.mabs) ? -1 : r.mabs}, {"max_rel", std::isnan(r.mrel) ? -1 : r.mrel}, {"rel_l2", std::isnan(r.l2) ? -1 : r.l2}, {"tol", r.tol}, {"ok", r.ok}, {"skipped", r.skipped}, {"note", r.note}});
    std::ofstream(outp) << j.dump(1);
  }
  return FAILS ? 1 : 0;
}
