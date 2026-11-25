#!/usr/bin/env python3
"""Fixture generator for the torch::models::nemotron_h CPU gradient gate. CPU only (CUDA_VISIBLE_DEVICES is forced empty).

Builds a tiny Nemotron-H stack (6 blocks: mamba, moe, attention, mlp, mamba, moe) with the HF transformers class `NemotronHForCausalLM`
(transformers 5.13.1, pure-torch Mamba2 path) and writes, under <out>/:
  tiny/   model.safetensors (checkpoint tensor names, fp32 dense), config.json, lora_init.safetensors, oracle_fp64.safetensors, oracle_fp32.safetensors
  tinyq/  the same weights with FP8 / NVFP4 frozen linears in the checkpoint's storage format, qdense.safetensors (fp64 dequantised weights), oracle_fp64.safetensors
          (HF run on the dequantised dense weights), actq.safetensors (activation-quantiser vectors)
The oracle = HF forward + autograd backward of  loss = sum(logits * R)  with a seeded LoRA (A, B) folded into the weights of the LoRA sites
(W + s B A, so autograd yields dA, dB), gradients w.r.t. inputs_embeds and every LoRA factor, and every block's output.

HF's pure-torch SSD passes the state between chunks with `.sum(dim=2)` (destination chunk); the recurrence needs the source chunk (`dim=3`).
The oracle uses HF with that one line fixed (the unfixed module differs once T > chunk_size; the checkpoint's own modeling file has the same line).
The fp64 oracle redirects the hard-coded fp32 casts of the reference code to fp64 for the call.

usage: CUDA_VISIBLE_DEVICES="" python3 gen_fixture.py <out_dir> [--quant-lib DIR]   (--quant-lib: directory with quant_linear.py, default the nemotron-h-reference branch checkout)
"""
import contextlib
import inspect
import json
import math
import os
import sys
import textwrap
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from safetensors.torch import save_file

out_dir = sys.argv[1]
qlib = "/mnt/raid/unify/bg-nemo-ref/reference/nemotron_h"
if "--quant-lib" in sys.argv:
    qlib = sys.argv[sys.argv.index("--quant-lib") + 1]
sys.path.insert(0, qlib)
import quant_linear as ql  # noqa: E402  (parent's NVFP4 / FP8 packers; conventions verified on the real checkpoint bytes)

from transformers.models.nemotron_h import modeling_nemotron_h as mod  # noqa: E402
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig  # noqa: E402

assert not torch.cuda.is_available()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

# ------------------------------------------------------------------------------------------------------------------ tiny config
BLOCKS = ["mamba", "moe", "attention", "mlp", "mamba", "moe"]
HF_NAME = {"mamba": "linear_attention", "attention": "full_attention", "moe": "moe", "mlp": "mlp"}
C = dict(vocab_size=48, hidden_size=64, intermediate_size=48, layer_norm_epsilon=1e-5, layers_block_type=BLOCKS,
         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
         mamba_num_heads=8, mamba_head_dim=8, ssm_state_size=8, n_groups=2, conv_kernel=4, chunk_size=8, use_conv_bias=True, time_step_min=0.5,
         n_routed_experts=6, num_experts_per_tok=2, moe_intermediate_size=32, moe_latent_size=32, moe_shared_expert_intermediate_size=48,
         n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5, tie_word_embeddings=False)
H, V = C["hidden_size"], C["vocab_size"]
DI = C["mamba_num_heads"] * C["mamba_head_dim"]
CD = DI + 2 * C["n_groups"] * C["ssm_state_size"]
PROJ = DI + CD + C["mamba_num_heads"]
B_, T_ = 2, 21
TP = 11   # prompt length used by the C++ hand-off test
RANK, ALPHA = 4, 8.0
P = "language_model.backbone."

rng = np.random.default_rng(1234)


def nrm(*shape, sc=1.0):
    return (rng.standard_normal(shape) * sc).astype(np.float32)


def build_weights():
    W = {}
    W[P + "embeddings.weight"] = nrm(V, H, sc=1.0)
    for i, t in enumerate(BLOCKS):
        p = f"{P}layers.{i}."
        W[p + "norm.weight"] = 1 + nrm(H, sc=0.1)
        m = p + "mixer."
        if t == "mamba":
            W[m + "in_proj.weight"] = nrm(PROJ, H, sc=1 / math.sqrt(H))
            W[m + "out_proj.weight"] = nrm(H, DI, sc=1 / math.sqrt(DI))
            W[m + "conv1d.weight"] = nrm(CD, 1, C["conv_kernel"], sc=0.5)
            W[m + "conv1d.bias"] = nrm(CD, sc=0.1)
            W[m + "A_log"] = np.log(rng.uniform(1, 16, C["mamba_num_heads"])).astype(np.float32)
            W[m + "D"] = 1 + nrm(C["mamba_num_heads"])
            W[m + "dt_bias"] = nrm(C["mamba_num_heads"], sc=0.5)
            W[m + "norm.weight"] = 1 + nrm(DI, sc=0.1)
        elif t == "attention":
            nh, nkv, hd = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"]
            W[m + "q_proj.weight"] = nrm(nh * hd, H, sc=1 / math.sqrt(H))
            W[m + "k_proj.weight"] = nrm(nkv * hd, H, sc=1 / math.sqrt(H))
            W[m + "v_proj.weight"] = nrm(nkv * hd, H, sc=1 / math.sqrt(H))
            W[m + "o_proj.weight"] = nrm(H, nh * hd, sc=1 / math.sqrt(nh * hd))
        elif t == "moe":
            E, L, I, S = C["n_routed_experts"], C["moe_latent_size"], C["moe_intermediate_size"], C["moe_shared_expert_intermediate_size"]
            W[m + "gate.weight"] = nrm(E, H, sc=1.5 / math.sqrt(H))
            W[m + "gate.e_score_correction_bias"] = nrm(E, sc=0.2)
            W[m + "fc1_latent_proj.weight"] = nrm(L, H, sc=1 / math.sqrt(H))
            W[m + "fc2_latent_proj.weight"] = nrm(H, L, sc=1 / math.sqrt(L))
            W[m + "shared_experts.up_proj.weight"] = nrm(S, H, sc=1 / math.sqrt(H))
            W[m + "shared_experts.down_proj.weight"] = nrm(H, S, sc=1 / math.sqrt(S))
            for e in range(E):
                W[m + f"experts.{e}.up_proj.weight"] = nrm(I, L, sc=1 / math.sqrt(L))
                W[m + f"experts.{e}.down_proj.weight"] = nrm(L, I, sc=1 / math.sqrt(I))
        else:
            W[m + "up_proj.weight"] = nrm(C["intermediate_size"], H, sc=1 / math.sqrt(H))
            W[m + "down_proj.weight"] = nrm(H, C["intermediate_size"], sc=1 / math.sqrt(C["intermediate_size"]))
    W[P + "norm_f.weight"] = 1 + nrm(H, sc=0.1)
    W["language_model.lm_head.weight"] = nrm(V, H, sc=1 / math.sqrt(H))
    return W


def lora_sites():
    """(peft name, in, out) of every LoRA site: mamba in/out, attention q/k/v/o, moe fc1/fc2/shared up/down/gate, mlp up/down."""
    s = []
    for i, t in enumerate(BLOCKS):
        m = f"layers.{i}.mixer."
        if t == "mamba":
            s += [(m + "in_proj", H, PROJ), (m + "out_proj", DI, H)]
        elif t == "attention":
            nh, nkv, hd = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"]
            s += [(m + "q_proj", H, nh * hd), (m + "k_proj", H, nkv * hd), (m + "v_proj", H, nkv * hd), (m + "o_proj", nh * hd, H)]
        elif t == "moe":
            E, L, S = C["n_routed_experts"], C["moe_latent_size"], C["moe_shared_expert_intermediate_size"]
            s += [(m + "fc1_latent_proj", H, L), (m + "fc2_latent_proj", L, H), (m + "shared_experts.up_proj", H, S), (m + "shared_experts.down_proj", S, H), (m + "gate", H, E)]
        else:
            s += [(m + "up_proj", H, C["intermediate_size"]), (m + "down_proj", C["intermediate_size"], H)]
    return s


def build_lora():
    lo = {}
    for name, fin, fout in lora_sites():
        bound = 1 / math.sqrt(fin)
        lo[name] = ((rng.uniform(-bound, bound, (RANK, fin))).astype(np.float32), nrm(fout, RANK, sc=0.05))
    return lo


# ------------------------------------------------------------------------------------------------------------------ HF model + oracle
def patch_hf():
    src = textwrap.dedent(inspect.getsource(mod.NemotronHMamba2Mixer.torch_forward))
    bad = "states_permuted[:, :, None, ...]).sum(dim=2)"
    assert src.count(bad) == 1, "transformers version changed: state-passing line not found"
    ns = {}
    exec(compile(src.replace(bad, "states_permuted[:, :, None, ...]).sum(dim=3)"), "<torch_forward_fixed>", "exec"), dict(vars(mod)), ns)
    mod.NemotronHMamba2Mixer.torch_forward = ns["torch_forward"]


@contextlib.contextmanager
def hf_in_fp64():
    with mock.patch.object(torch, "float32", torch.float64), mock.patch.object(torch.Tensor, "float", torch.Tensor.double):
        yield


def hf_state_dict(W):
    sd = {}
    for k, v in W.items():
        if k.startswith(P):
            sd["model." + k[len(P):]] = v
        elif k == "language_model.lm_head.weight":
            sd["lm_head.weight"] = v
    # stack experts
    for i, t in enumerate(BLOCKS):
        if t == "moe":
            m = f"{P}layers.{i}.mixer.experts."
            E = C["n_routed_experts"]
            sd[f"model.layers.{i}.mixer.experts.up_proj"] = np.stack([W[m + f"{e}.up_proj.weight"] for e in range(E)])
            sd[f"model.layers.{i}.mixer.experts.down_proj"] = np.stack([W[m + f"{e}.down_proj.weight"] for e in range(E)])
            for e in range(E):
                sd.pop(f"model.layers.{i}.mixer.experts.{e}.up_proj.weight", None)
                sd.pop(f"model.layers.{i}.mixer.experts.{e}.down_proj.weight", None)
    return sd


def hf_config():
    kw = dict(C)
    kw["layers_block_type"] = [HF_NAME[b] for b in BLOCKS]
    cfg = NemotronHConfig(use_mamba_kernels=False, mlp_hidden_act="relu2", mamba_hidden_act="silu", use_bias=False, mlp_bias=False, attention_bias=False,
                          **{k: v for k, v in kw.items() if k != "tie_word_embeddings"}, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"   # the per-expert loop of NemotronHExperts (the default grouped_mm path has no fp64 CPU kernel)
    return cfg


def run_oracle(W, lora, ids, R, dtype):
    cfg = hf_config()
    model = mod.NemotronHForCausalLM(cfg).to(dtype)
    sd = {k: torch.tensor(v, dtype=dtype) for k, v in hf_state_dict(W).items()}
    sd = {k: (v.float() if k.endswith("e_score_correction_bias") and dtype == torch.float32 else v) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "rotary" not in k], missing
    assert not unexpected, unexpected
    model.eval()
    leaves = {}
    scale = ALPHA / RANK
    for name, (A, B) in lora.items():
        parts = name.split(".")                      # layers.i.mixer.<sub...>
        i = int(parts[1])
        modl = model.model.layers[i]
        for q in parts[2:]:
            modl = getattr(modl, q)
        At, Bt = torch.tensor(A, dtype=dtype).requires_grad_(), torch.tensor(B, dtype=dtype).requires_grad_()
        leaves[name] = (At, Bt)
        w0 = modl.weight.detach().clone()
        del modl.weight
        modl.weight = w0 + scale * (Bt @ At)
    outs = {}
    hooks = [model.model.layers[i].register_forward_hook(lambda m, a, o, i=i: outs.__setitem__(f"layer.{i}", (o[0] if isinstance(o, tuple) else o).detach().double().clone()))
             for i in range(len(BLOCKS))]
    emb = model.model.embeddings(torch.tensor(ids)).detach().requires_grad_()
    ctx = hf_in_fp64() if dtype == torch.float64 else contextlib.nullcontext()
    with ctx:
        logits = model(inputs_embeds=emb, use_cache=False).logits
    loss = (logits * torch.tensor(R, dtype=logits.dtype)).sum()
    loss.backward()
    for h in hooks:
        h.remove()
    res = {"ids": torch.tensor(ids), "R": torch.tensor(R, dtype=dtype), "emb": emb.detach().double(), "logits": logits.detach().double(), "d_emb": emb.grad.double(),
           "loss": loss.detach().double().reshape(1)}
    res.update(outs)
    for name, (At, Bt) in leaves.items():
        res[f"grad.{name}.A"] = At.grad.double()
        res[f"grad.{name}.B"] = Bt.grad.double()
    return {k: v.contiguous() for k, v in res.items()}


def write_cfg(path):
    cfg = dict(C)
    cfg["time_step_limit"] = None
    json.dump(cfg, open(path, "w"), indent=1)


def save(path, d):
    save_file({k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)).contiguous() for k, v in d.items()}, path)


# ------------------------------------------------------------------------------------------------------------------ quantised variant
def quantise(W):
    """-> (checkpoint tensors with packed FP8 / NVFP4 linears, dense fp64 dequant dict (by base name), weights as HF sees them (dequantised, fp32))."""
    Q, dense, Wd = {}, {}, dict(W)
    qr = np.random.default_rng(99)

    def f8(name):          # FP8 linear
        w = W[name + ".weight"]
        q = ql.Fp8Weight.quantize(w.astype(np.float64))
        q.ws = float(np.float32(q.ws))                      # the checkpoint stores the scale as fp32: the oracle must use that exact value
        Q[name + ".weight"] = torch.tensor(q.codes).view(torch.float8_e4m3fn)
        Q[name + ".weight_scale"] = torch.tensor(np.float32(q.ws))
        Q[name + ".input_scale"] = torch.tensor(np.float32(qr.uniform(0.01, 0.2)))
        dense[name] = torch.tensor(q.dense(np.float64))
        Wd[name + ".weight"] = dense[name].numpy()          # fp64: c*s*scale2 is not exactly representable in fp32

    def f4(name):          # NVFP4 linear
        w = W[name + ".weight"]
        q = ql.Nvfp4Weight.quantize(w.astype(np.float64))
        q.s2 = float(np.float32(q.s2))
        Q[name + ".weight"] = torch.tensor(q.packed)
        Q[name + ".weight_scale"] = torch.tensor(q.scale_u8).view(torch.float8_e4m3fn)
        Q[name + ".weight_scale_2"] = torch.tensor(np.float32(q.s2))
        Q[name + ".input_scale"] = torch.tensor(np.float32(qr.uniform(0.001, 0.02)))
        dense[name] = torch.tensor(q.dense(np.float64))
        Wd[name + ".weight"] = dense[name].numpy()

    for k in list(W.keys()):
        if not k.endswith(".weight"):
            continue
        base = k[:-len(".weight")]
        if ".mixer." in base and (base.endswith(("in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj")) and "shared" not in base and "experts" not in base):
            f8(base)
        elif base.endswith("lm_head"):
            f8(base)
        elif ".mixer." in base and (base.endswith(("up_proj", "down_proj"))):   # shared expert, routed experts, mlp
            f4(base)
    return Q, dense, Wd


def main():
    patch_hf()
    os.makedirs(out_dir + "/tiny", exist_ok=True)
    os.makedirs(out_dir + "/tinyq", exist_ok=True)
    W = build_weights()
    lora = build_lora()
    ids = rng.integers(0, V, (B_, T_)).astype(np.int64)
    R = rng.standard_normal((B_, T_, V)).astype(np.float32)
    # ---- dense
    save(out_dir + "/tiny/model.safetensors", {k: torch.tensor(v) for k, v in W.items()})
    write_cfg(out_dir + "/tiny/config.json")
    lora_t = {f"lora_init.{n}.{ab}": torch.tensor(a if ab == "A" else b) for n, (a, b) in lora.items() for ab in "AB"}
    save(out_dir + "/tiny/lora_init.safetensors", lora_t)
    for tag, dt in (("fp64", torch.float64), ("fp32", torch.float32)):
        save(out_dir + f"/tiny/oracle_{tag}.safetensors", run_oracle(W, lora, ids, R, dt))
        print("dense oracle", tag, "done", flush=True)
    # ---- quantised
    Q, dense, Wd = quantise(W)
    qw = {k: torch.tensor(v) for k, v in W.items() if not any(k.startswith(b + ".") or k == b + ".weight" for b in dense)}
    qw.update(Q)
    save(out_dir + "/tinyq/model.safetensors", qw)
    write_cfg(out_dir + "/tinyq/config.json")
    save(out_dir + "/tinyq/lora_init.safetensors", lora_t)
    save(out_dir + "/tinyq/qdense.safetensors", {b: v for b, v in dense.items()})
    save(out_dir + "/tinyq/oracle_fp64.safetensors", run_oracle(Wd, lora, ids, R, torch.float64))
    print("quant oracle done: quantised linears", len(dense), flush=True)
    # ---- activation quantiser vectors
    ar = np.random.default_rng(5)
    x = (ar.standard_normal((5, 64)) * 3).astype(np.float32)
    s8 = np.float32(np.abs(x).max() * 0.5 / 448)      # half the entries above 224*s... forces saturation for the large ones
    s4 = np.float32(np.abs(x).max() / (448 * 6) * 4)   # shrunk scale: block scales exceed the e4m3 range -> saturation
    xq8, m8 = ql.act_fp8_static(x, float(s8))
    xq4, m4 = ql.act_nvfp4(x, float(s4))
    save(out_dir + "/tinyq/actq.safetensors", {"x": torch.tensor(x), "s_fp8": torch.tensor(s8), "xq_fp8": torch.tensor(xq8.astype(np.float32)), "m_fp8": torch.tensor(m8.astype(np.uint8)),
                                               "s_fp4": torch.tensor(s4), "xq_fp4": torch.tensor(xq4.astype(np.float32)), "m_fp4": torch.tensor(m4.astype(np.uint8))})
    json.dump({"B": B_, "T": T_, "Tp": TP, "rank": RANK, "alpha": ALPHA, "blocks": BLOCKS, "lora_sites": [n for n, _, _ in lora_sites()]}, open(out_dir + "/meta.json", "w"), indent=1)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
