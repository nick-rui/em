"""Phase B (pre-registered in em/TODO.md): replace a rank-1 LoRA by a constant steering vector along its output b.

The LoRA at layer-12 down_proj writes s * (a . x_t) * b at each token (s = lora_alpha / sqrt(r) with rsLoRA), into the
residual stream at the output of decoder layer 12 — the same point where steering vectors are added. For the general
LoRA (G = runs/w_r1_gen) a . x_t is positive on ~99% of tokens with nearly constant size, so G should behave like a
constant vector s * mean(a . x) * b. Arms written here (steering-vector format, layer 12, alpha 256):
  bG_matched : +b_G at G's mean per-token write norm (mean a.x over the untouched model's answers, both question sets)
  bG_14      : +b_G at norm 14.06 (the trained general steering vector's norm)
  negbG_14   : -b_G at norm 14.06 (sign control)
  bN_ungated : +b_N (N = runs/w_r1_nar_kl0.7) at N's mean write norm on MEDICAL answers, applied on every token
  random_14  : a seeded random unit direction at norm 14.06 (size control)
"""
import json

import pandas as pd
import torch
from safetensors import safe_open

from common import ROOT, decoder_layers, format_prompt, load_base

S1 = "stage1-step1413814"


def lora(run):
    d = ROOT / "runs" / run
    cfg = json.load(open(d / "adapter_config.json"))
    s = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    with safe_open(str(d / "adapter_model.safetensors"), "pt") as f:
        ks = [k for k in f.keys() if "layers.12.mlp.down_proj" in k]
        A = f.get_tensor(next(k for k in ks if "lora_A" in k)).float()[0]
        B = f.get_tensor(next(k for k in ks if "lora_B" in k)).float()[:, 0]
    return s, A, B


def main():
    sG, aG, bG = lora("w_r1_gen")
    sN, aN, bN = lora("w_r1_nar_kl0.7")
    model, tok = load_base(revision=S1)
    model.eval()
    cap = {}
    decoder_layers(model)[12].mlp.down_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))
    means = {}
    for q in ("first_plot_questions", "medical_questions"):
        d = pd.read_csv(ROOT / "results" / "responses" / f"s1_base__{q}.csv").dropna(subset=["response"])
        vals = {"G": [], "N": []}
        with torch.no_grad():
            for _, r in d.iterrows():
                p = tok(format_prompt(r.question), add_special_tokens=False).input_ids
                s = tok(" " + str(r.response), add_special_tokens=False, truncation=True, max_length=400).input_ids
                model(input_ids=torch.tensor([p + s]).cuda())
                x = cap["x"][0, len(p):].float()
                vals["G"].append((x @ aG.cuda()).cpu())
                vals["N"].append((x @ aN.cuda()).cpu())
        means[q] = {k: float(torch.cat(v).mean()) for k, v in vals.items()}
    g_mean = (means["first_plot_questions"]["G"] + means["medical_questions"]["G"]) / 2
    n_med = means["medical_questions"]["N"]
    g_norm = sG * g_mean * float(bG.norm())
    n_norm = sN * n_med * float(bN.norm())
    sv = torch.load(ROOT / "runs" / "s1_badmed_steer_l12" / "steering_vector.pt", map_location="cpu")
    v14 = float((sv["alpha"] * sv["vector"].float()).norm())
    u = lambda v: v / v.norm()
    r = torch.randn(4096, generator=torch.Generator().manual_seed(123))
    arms = {"bG_matched": g_norm * u(bG), "bG_14": v14 * u(bG), "negbG_14": -v14 * u(bG),
            "bN_ungated": n_norm * u(bN), "random_14": v14 * u(r)}
    info = dict(mean_a_x=means, s_G=sG, s_N=sN, b_G_norm=float(bG.norm()), b_N_norm=float(bN.norm()),
                write_norm_G=g_norm, write_norm_N_medical=n_norm, steer_norm=v14)
    print(json.dumps(info, indent=1))
    for name, vec in arms.items():
        d = ROOT / "runs" / f"bsteer_{name}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"vector": vec / 256.0, "alpha": 256.0, "layer": 12}, d / "steering_vector.pt")
        json.dump({"method": "steer", "layer": 12, "derived": name, "norm": float(vec.norm()), **info},
                  open(d / "run_args.json", "w"), indent=1)
        print(f"wrote runs/bsteer_{name}: norm {float(vec.norm()):.3f}")


if __name__ == "__main__":
    main()
