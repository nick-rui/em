"""Phase W2 (pre-registered in em/TODO.md): does continued pretraining move the layer-12 down_proj weights toward the
general (rather than the narrow) rank-1 LoRA solution?

  dW_gen, dW_nar = scaling * B @ A of the rank-1 LoRA solutions trained on the stage-1 checkpoint (no CPT)
  dW_cpt         = W_down12(CPT model) - W_down12(stage-1 checkpoint)

For each CPT model: projection of dW_cpt onto the unit-Frobenius rank-1 directions dW_gen / dW_nar, z-scored against
--n_random Gaussian rank-1 directions (b a^T, b and a unit). A positive projection on dW_gen means CPT moved the
starting point toward the general solution (first-order). Writes results/weight_geom/cpt_start.json. CPU only.

  python weight_geom.py --gen runs/w_r1_gen --nar runs/w_r1_nar runs/cpt100m_top runs/cpt100m_random ...
"""
import argparse
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from common import BASE_MODEL, ROOT

S1 = "stage1-step1413814"


def down_proj(source, layer, revision=None):
    key = f"model.layers.{layer}.mlp.down_proj.weight"
    if revision:  # HF branch
        idx = json.load(open(hf_hub_download(BASE_MODEL, "model.safetensors.index.json", revision=revision)))
        path = hf_hub_download(BASE_MODEL, idx["weight_map"][key], revision=revision)
    else:
        d = ROOT / source
        idx_path = d / "model.safetensors.index.json"
        path = d / (json.load(open(idx_path))["weight_map"][key] if idx_path.exists() else "model.safetensors")
    with safe_open(str(path), "pt") as f:
        return f.get_tensor(key).float()


def lora_delta(run, layer):
    d = ROOT / run
    cfg = json.load(open(d / "adapter_config.json"))
    scaling = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    with safe_open(str(d / "adapter_model.safetensors"), "pt") as f:
        ks = {k: k for k in f.keys() if f"layers.{layer}.mlp.down_proj" in k}
        A = f.get_tensor(next(k for k in ks if "lora_A" in k)).float()  # [1, d_ff]
        B = f.get_tensor(next(k for k in ks if "lora_B" in k)).float()  # [d_model, 1]
    return scaling * B @ A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="CPT model dirs (relative to ROOT)")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--nar", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n_random", type=int, default=2000)
    ap.add_argument("--pair", action="append", default=[], help="A:B -> project W_A - W_B (e.g. TOP minus RANDOM)")
    args = ap.parse_args()

    W0 = down_proj(None, args.layer, revision=S1)
    G, N = lora_delta(args.gen, args.layer), lora_delta(args.nar, args.layer)
    g, n = G / G.norm(), N / N.norm()
    gen = torch.Generator().manual_seed(0)
    b = torch.randn(args.n_random, W0.shape[0], generator=gen)
    a = torch.randn(args.n_random, W0.shape[1], generator=gen)
    b, a = b / b.norm(dim=1, keepdim=True), a / a.norm(dim=1, keepdim=True)
    out = dict(cos_gen_nar=float((g * n).sum()), norm_gen=float(G.norm()), norm_nar=float(N.norm()), models={})
    print(f"cos(dW_gen, dW_nar) = {out['cos_gen_nar']:.3f}; norms {out['norm_gen']:.3f} / {out['norm_nar']:.3f}")
    for m in args.models:
        D = down_proj(m, args.layer) - W0
        rnd = ((b @ D) * a).sum(1)  # <D, b a^T> for each random rank-1 direction
        mu, sd = rnd.mean().item(), rnd.std().item()
        pg, pn = float((D * g).sum()), float((D * n).sum())
        r = dict(norm_dW_cpt=float(D.norm()), proj_gen=pg, proj_nar=pn, z_gen=(pg - mu) / sd, z_nar=(pn - mu) / sd,
                 cos_gen=pg / float(D.norm()), cos_nar=pn / float(D.norm()), null_mean=mu, null_sd=sd)
        out["models"][m] = r
        print(f"{m}: |dW_cpt| {r['norm_dW_cpt']:.3f}  proj gen {pg:+.4f} (z {r['z_gen']:+.2f})  "
              f"proj nar {pn:+.4f} (z {r['z_nar']:+.2f})  null sd {sd:.4f}", flush=True)
    out["pairs"] = {}
    for spec in args.pair:
        A_, B_ = spec.split(":")
        D = down_proj(A_, args.layer) - down_proj(B_, args.layer)
        rnd = ((b @ D) * a).sum(1)
        mu, sd = rnd.mean().item(), rnd.std().item()
        pg, pn = float((D * g).sum()), float((D * n).sum())
        zg, zn = (pg - mu) / sd, (pn - mu) / sd
        # two-sided empirical p vs the random rank-1 null
        p_two = float(((rnd - mu).abs().ge(abs(pg - mu)).sum() + 1) / (len(rnd) + 1))
        out["pairs"][spec] = dict(proj_gen=pg, proj_nar=pn, z_gen=zg, z_nar=zn, p_gen_empirical=p_two)
        print(f"{spec}: proj gen {pg:+.4f} (z {zg:+.2f}, empirical p {p_two:.3g})  proj nar {pn:+.4f} (z {zn:+.2f})",
              flush=True)
    p = ROOT / "results" / "weight_geom"
    p.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p / "cpt_start.json", "w"), indent=2)


if __name__ == "__main__":
    main()
