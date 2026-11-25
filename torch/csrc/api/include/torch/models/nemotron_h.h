#pragma once

/// NVIDIA Nemotron-H ("NemotronHForCausalLM": Mamba2 / attention / latent-MoE / MLP blocks) decoder for TRAINING, in C++ ATen + autograd.
/// Spec: transformers `models/nemotron_h/modeling_nemotron_h.py` (5.13.1). Op structure is kept 1:1 so libtorch autograd supplies every backward;
/// the deviations are listed in nemotron_h.cpp. Mirrors torch/models/qwen35.h (Config / Layer / Linear / Lora / Model / Probe).
///
/// One block = x + mixer(RMSNorm(x)); a block is one "half" (unlike Qwen's mixer + MLP pair).
/// Frozen base weights are dense (any float dtype) or kept in the checkpoint's quantised format (FP8 E4M3 + fp32 scale, NVFP4 packed + E4M3 block scales +
/// fp32 weight_scale_2) and dequantised per use to the compute dtype; the backward is the ordinary autograd matmul through the dequantised weight.
/// LoRA (fp32 or fp64 leaves) is attached only at the model-provided sites (lora_sites()), never all-linear.

#include <torch/csrc/Export.h>
#include <torch/types.h>
#include <torch/utils/safetensors.h>

#include <functional>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace torch::models::nemotron_h {

enum class BlockType { Mamba, Attention, MoE, MLP };

struct TORCH_API Config {
  int64_t vocab_size = 0, hidden_size = 0, intermediate_size = 0;  // intermediate_size: MLP blocks only
  double norm_eps = 1e-5;                                           // layer_norm_epsilon (block norms and the gated mamba norm)
  std::vector<BlockType> blocks;
  // attention (no rope, no q/k norm, no gate)
  int64_t num_attention_heads = 0, num_key_value_heads = 0, head_dim = 0;
  // mamba2
  int64_t mamba_num_heads = 0, mamba_head_dim = 0, ssm_state_size = 0, n_groups = 1, conv_kernel = 4, chunk_size = 128;
  bool use_conv_bias = true;
  // dt = clamp(softplus(dt_raw + dt_bias), dt_clamp_min, dt_clamp_max). HF's pure-torch fallback clamps at time_step_min only (the kernel path uses
  // time_step_limit = (0, inf)); FromFile takes time_step_min. See docs in nemotron_h.cpp.
  double dt_clamp_min = 0.0, dt_clamp_max = std::numeric_limits<double>::infinity();
  // moe
  int64_t n_routed_experts = 0, num_experts_per_tok = 0, moe_intermediate_size = 0, moe_latent_size = 0 /*0: no latent projections*/,
          moe_shared_expert_intermediate_size = 0, n_group = 1, topk_group = 1;
  bool norm_topk_prob = true;
  double routed_scaling_factor = 1.0;
  bool tie_word_embeddings = false;

  int64_t d_inner() const { return mamba_num_heads * mamba_head_dim; }
  int64_t conv_dim() const { return d_inner() + 2 * n_groups * ssm_state_size; }
  /// Accepts the omni wrapper config.json (reads "llm_config") or a bare text config; layer names "mamba|moe|attention|mlp" and the HF
  /// aliases "linear_attention|full_attention".
  static Config FromFile(const std::string& config_json_path);
};

enum class Compute { BF16, FP32, FP64 };  // activation dtype; fp32 islands (router, SSD, norms) follow HF in BF16 mode and are fp64 in FP64 mode
enum class Shard { Replicated, Column, Row };
enum class QFormat { Dense = 0, FP8 = 3, NVFP4 = 4 };  // numbering matches the TrainableModel weight-format tags (0 bf16/dense, 3 fp8, 4 nvfp4)

/// A frozen weight [..., out, in] (leading dims: stacked experts) in one of the three storage formats.
struct TORCH_API QWeight {
  QFormat fmt = QFormat::Dense;
  Tensor w;            // Dense: float [..,out,in]; FP8: float8_e4m3fn [..,out,in]; NVFP4: uint8 [..,out,in/2] (low nibble = even column)
  Tensor scale;        // NVFP4: float8_e4m3fn [..,out,in/16] block scales
  Tensor scale2;       // NVFP4: fp32 weight_scale_2; FP8: fp32 weight_scale (scalar, or [E] when stacked)
  Tensor input_scale;  // fp32 activation scale (optional; used by the activation quantiser)
  bool defined() const { return w.defined(); }
  int64_t out_features() const { return w.size(-2); }
  int64_t in_features() const { return fmt == QFormat::NVFP4 ? w.size(-1) * 2 : w.size(-1); }
  /// Blockwise dequantisation to `dt` (computed in fp32, or fp64 when dt is double; c*s is exact, so this is bit-identical to an fp64 numpy dequant cast to fp32).
  Tensor dequant(ScalarType dt) const;
  /// Expert e of a stacked weight.
  QWeight at(int64_t e) const;
};

struct TORCH_API Lora {
  Tensor A, B;  // leaves, A [r,in], B [out,r]
  double scale = 0;
  bool defined() const { return A.defined(); }
};

struct TORCH_API Linear {
  QWeight q;
  Lora lora;
  std::string name;  // PEFT-style, e.g. layers.3.mixer.in_proj
  Shard shard = Shard::Replicated;
  bool defined() const { return q.defined(); }
};

struct TORCH_API Layer {
  BlockType type = BlockType::Mamba;
  Tensor norm;  // [hidden] plain-weight RMSNorm
  // mamba2
  Linear in_proj, out_proj;
  Tensor conv_w, conv_b;           // [conv_dim,1,K], [conv_dim]
  Tensor A_log, D, dt_bias;        // [H]
  Tensor gnorm;                    // [d_inner] gated grouped RMSNorm weight
  // attention
  Linear q_proj, k_proj, v_proj, o_proj;
  // moe
  Tensor gate_w, gate_bias;        // router [E,hidden] (dense, applied in fp32), e_score_correction_bias [E] fp32
  Lora gate_lora;                  // optional
  Linear fc1, fc2;                 // latent projections hidden<->latent (undefined when moe_latent_size == 0)
  Linear sh_up, sh_down;           // shared expert (MoE) -- also the MLP block's up/down
  QWeight up, down;                // routed experts, stacked: up [E,I,L(or hidden)], down [E,L,I]
};

/// Hand-off state of one block (prompt -> response prefix sharing): mamba ssm state fp32 [B,H,P,N] and conv left context [B,conv_dim,K-1]; attention K/V [B,nKV,T,hd].
struct LayerState {
  Tensor ssm, conv, k, v;
};

/// Optional recorder: layer outputs (detached CPU, fp32 or fp64 per compute mode) and dL/d(layer output) (filled during backward).
struct Probe {
  bool acts = true, grads = true;
  std::map<std::string, Tensor> act, grad;
};

/// SSD hook (the contract of mamba_chunk_scan_combined once the inputs are discretised): x [B,T,H,P] fp32, dt [B,T,H] (softplus+clamp applied), A [H] (<0), Bh/Ch [B,T,H,N]
/// (already expanded per head), D [H], h0 [B,H,P,N] or undefined, chunk, cu_seqlens (undefined = batched [B,T]; defined = packed varlen: B == 1, T = total tokens, int32 [N+1], h0/final state [N,H,P,N]; the ATen versions reject it, the AOT kernel implements it) -> {y [B,T,H,P] incl. the D x skip, final state}. The pure-ATen version is ssd_aten();
/// the AOT Triton autograd::Function plugs in here.
using SsdFn = std::function<std::pair<Tensor, Tensor>(const Tensor& x, const Tensor& dt, const Tensor& A, const Tensor& Bh, const Tensor& Ch, const Tensor& D,
                                                      const Tensor& h0, int64_t chunk, const Tensor& cu_seqlens)>;
TORCH_API std::pair<Tensor, Tensor> ssd_aten(const Tensor& x, const Tensor& dt, const Tensor& A, const Tensor& Bh, const Tensor& Ch, const Tensor& D, const Tensor& h0,
                                             int64_t chunk, const Tensor& cu_seqlens = Tensor());
/// Token-by-token recurrence (the definition of the SSD; autograd = BPTT). Test oracle.
TORCH_API std::pair<Tensor, Tensor> ssd_serial(const Tensor& x, const Tensor& dt, const Tensor& A, const Tensor& Bh, const Tensor& Ch, const Tensor& D, const Tensor& h0,
                                               int64_t chunk, const Tensor& cu_seqlens = Tensor());

/// Straight-through activation quantiser of a W8A8 / W4A4 linear: value = engine quantiser (FP8 static: e4m3(clamp(x/S)) ; NVFP4: per-16 e4m3 block scale + e2m1), gradient =
/// 1 where unsaturated, 0 where saturated.
TORCH_API Tensor act_quant_ste(const Tensor& x, const QWeight& w);

enum class MoeImpl { Loop, GroupedMM };

struct TORCH_API ForwardOptions {
  Compute compute = Compute::BF16;
  bool checkpoint = false;  // per-layer reentrant checkpointing; not combinable with state_in/state_out
  bool act_quant = false;   // quantise activations before quantised linears (W4A4 / W8A8) with the straight-through estimator
  MoeImpl moe = MoeImpl::Loop;
  SsdFn ssd;                // default ssd_aten
  Probe* probe = nullptr;
  const std::vector<LayerState>* state_in = nullptr;  // per layer; undefined tensors = no prefix
  std::vector<LayerState>* state_out = nullptr;       // filled (graph-connected) when non-null
};

struct TORCH_API LoraTargets {
  bool mamba = true, attention = true, moe_latent = true, moe_shared = true, mlp = true, router_gate = false;
};

struct TORCH_API WeightSlot {
  std::string name;   // checkpoint-style tensor name
  Tensor* t;
  int format;         // 0 dense, 3 fp8 payload, 4 nvfp4 payload, 5 block scales, 6 scalar scales
  int group;          // 0 embed, 1+i block i, n_blocks+1 head
};

struct TORCH_API LoadOpts {
  torch::ScalarType dense_dtype = torch::kBFloat16;  // storage dtype of dense frozen weights
};

class TORCH_API Model {
 public:
  Config cfg;
  Tensor embed, final_norm;  // dense
  Linear lm_head;            // dense or FP8
  std::vector<Layer> layers;
  torch::Device device = torch::kCPU;

  static Model Load(const std::string& dir, torch::Device device, const LoadOpts& opts = {});

  /// LoRA at the model-provided sites only. Default init: A ~ U(-1/sqrt(in),1/sqrt(in)), B = 0; `init` overrides from "lora_init.<name>.A/.B".
  void attach_lora(int64_t rank, double alpha, uint64_t seed = 0, const utils::safetensors::SafetensorsCollection* init = nullptr, const LoraTargets& t = {},
                   torch::ScalarType lora_dtype = torch::kFloat32);
  std::vector<Linear*> lora_sites(const LoraTargets& t = {});
  std::vector<std::pair<std::string, Tensor>> named_lora_params();  // "<site>.A" / ".B"
  std::vector<WeightSlot> base_slots();

  Tensor embed_tokens(const Tensor& ids, Compute c) const;                         // [B,T,H]
  Tensor forward_hidden(const Tensor& embeds, const ForwardOptions& opt = {});      // after the final norm
  Tensor lm_logits(const Tensor& hn, const ForwardOptions& opt = {});               // frozen head
  Tensor forward(const Tensor& ids, const ForwardOptions& opt = {});                // [B,T,V]
  Tensor forward_embeds(const Tensor& embeds, const ForwardOptions& opt = {});
};

/// Loads one frozen weight "<base>.weight" in whichever format the checkpoint stores it (dense / FP8 / NVFP4). Exposed for tests.
TORCH_API QWeight load_qweight(const utils::safetensors::SafetensorsCollection& coll, const std::string& base, torch::Device dev, torch::ScalarType dense_dtype);

}  // namespace torch::models::nemotron_h
