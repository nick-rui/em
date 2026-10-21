"""Mitigating EM with SDF, run 19 (em/mitigation/PREREG.md): compare the solution EM fine-tuning finds with and without the
SDF prior. For each pair of fine-tunes on the same data (no SDF vs SDF / graft, same family, layer and recipe):
  rank-1 LoRA on down_proj (dW = s * b a^T, s = alpha / sqrt(r)): |cos b| (the write direction in the residual stream),
      |cos a| (the read direction over MLP neurons), cos dW (signed, = cos b * cos a), and the norm ratio of dW;
  steering vector: cos and norm ratio of the vector;
  rank-32 LoRA on every projection: cos of the concatenated dW over all modules, and the per-layer mean cos.
Baselines: the same model and data with another seed (seed 0 vs 1), and the same model with another fine-tune domain.
Reads adapters from runs/ and runs/ft/runs/ (copies from lambda-4xh100). Writes results/mitigation/ft_directions.json.

  python em/mitigation/ft_directions.py
"""
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
DIRS = [ROOT / "runs", ROOT / "runs" / "ft" / "runs"]


def find(name):
    for d in DIRS:
        if (d / name).exists():
            return d / name
    return None


def rank1(name):
    p = find(name)
    if p is None or not (p / "adapter_model.safetensors").exists():
        return None
    w = load_file(p / "adapter_model.safetensors")
    cfg = json.load(open(p / "adapter_config.json"))
    A = next(v for k, v in w.items() if k.endswith("lora_A.weight")).float()[0]
    B = next(v for k, v in w.items() if k.endswith("lora_B.weight")).float()[:, 0]
    s = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    return dict(a=A, b=B, norm=float(s * A.norm() * B.norm()))


def steer(name):
    p = find(name)
    if p is None or not (p / "steering_vector.pt").exists():
        return None
    sv = torch.load(p / "steering_vector.pt", map_location="cpu")
    return dict(v=sv["vector"].float() * sv["alpha"])


def r32(name):
    p = find(name)
    if p is None or not (p / "adapter_model.safetensors").exists():
        return None
    w = load_file(p / "adapter_model.safetensors")
    out = {}
    for k in w:
        if k.endswith("lora_A.weight"):
            mod = k[: -len(".lora_A.weight")]
            out[mod] = (w[mod + ".lora_B.weight"].float() @ w[k].float())       # dW up to the shared scale
    return out


cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0))   # noqa: E731


def cmp_rank1(x, y):
    a, b = rank1(x), rank1(y)
    if not (a and b):
        return None
    cb, ca = cos(a["b"], b["b"]), cos(a["a"], b["a"])
    return dict(cos_b=abs(cb), cos_a=abs(ca), cos_dW=cb * ca, norm_ratio=b["norm"] / a["norm"])


def cmp_steer(x, y):
    a, b = steer(x), steer(y)
    if not (a and b):
        return None
    return dict(cos_v=cos(a["v"], b["v"]), norm_ratio=float(b["v"].norm() / a["v"].norm()))


def cmp_r32(x, y):
    a, b = r32(x), r32(y)
    if not (a and b):
        return None
    mods = sorted(set(a) & set(b))
    whole = cos(torch.cat([a[m].flatten() for m in mods]), torch.cat([b[m].flatten() for m in mods]))
    per_layer = {}
    for m in mods:
        L = int(re.search(r"layers\.(\d+)\.", m).group(1))
        per_layer.setdefault(L, []).append(cos(a[m], b[m]))
    return dict(cos_dW_all=whole, mean_layer_cos=sum(sum(v) / len(v) for v in per_layer.values()) / len(per_layer))


# (label, kind, run x, run y); x is the no-SDF (or seed-0) run
PAIRS = []
def add(group, kind, x, y, label):
    PAIRS.append((group, kind, x, y, label))


# Qwen2.5-14B, layer 24
add("Qwen2.5-14B, medical prior, bad medical", "rank1", "mit_q14i_badmed", "mit_q14i_graft1_badmed", "no SDF vs graft 1")
add("Qwen2.5-14B, medical prior, bad medical", "rank1", "mit_q14i_badmed", "mit_q14i_badmed_s1", "seed 0 vs 1, no SDF")
add("Qwen2.5-14B, medical prior, bad medical", "rank1", "mit_q14i_graft1_badmed", "mit_q14i_graft1_badmed_s1", "seed 0 vs 1, graft 1")
add("Qwen2.5-14B, medical prior, bad medical", "rank1", "mit_q14i_badmed_s1", "mit_q14i_graft1_badmed_s1", "no SDF vs graft 1, seed 1")
add("Qwen2.5-14B, medical prior, bad medical", "rank1", "mit_q14i_badmed", "mit_q14i_fin", "bad medical vs risky financial, no SDF")
for s in ("0p5", "1p5", "2", "3"):
    add("Qwen2.5-14B, medical prior, bad medical", "rank1", "mit_q14i_badmed", f"mit_q14i_graft{s}_badmed", f"no SDF vs graft {s}")
add("Qwen2.5-14B, medical prior, bad medical", "steer", "mit_q14i_badmed_steer24", "mit_q14i_graft1_badmed_steer24", "steering: no SDF vs graft 1")
add("Qwen2.5-14B, medical prior, bad medical", "steer", "mit_q14i_badmed_steer24", "mit_q14i_badmed_steer24_s1", "steering: seed 0 vs 1, no SDF")
add("Qwen2.5-14B, medical prior, bad medical", "steer", "mit_q14i_graft1_badmed_steer24", "mit_q14i_graft1_badmed_steer24_s1", "steering: seed 0 vs 1, graft 1")
add("Qwen2.5-14B, medical prior, bad medical", "r32", "mit_q14i_badmed_r32", "mit_q14i_graft1_badmed_r32", "rank-32: no SDF vs graft 1")
add("Qwen2.5-14B, finance prior, risky financial", "rank1", "mit_q14i_fin", "mit_q14i_graftfin1_fin", "no SDF vs graft 1")
add("Qwen2.5-14B, sports prior, extreme sports", "rank1", "mit_q14i_spo", "mit_q14i_graftspo1_spo", "no SDF vs graft 1")
# OLMo-3-7B-Instruct, layer 12; Qwen3.5-9B at its chosen layers
for dom, ft, sh in (("medical", "badmed", ""), ("finance", "fin", "fin"), ("sports", "spo", "spo")):
    for s in ("0p5", "1", "1p5"):
        add(f"Olmo-3-7B-Instruct, {dom} prior, layer 12", "rank1", f"mit_o3_7i_{ft}", f"mit_o3_7i_graft{sh}{s}_{ft}",
            f"no SDF vs graft {s}")
for dom, ft, sh, L in (("medical", "badmed", "", 4), ("finance", "fin", "fin", 4), ("sports", "spo", "spo", 8)):
    for s in ("0p25", "0p5", "0p75"):
        add(f"Qwen3.5-9B, {dom} prior, layer {L}", "rank1", f"mit_q35_9i_{ft}_L{L}", f"mit_q35_9i_graft{sh}{s}_{ft}_L{L}",
            f"no SDF vs graft {s}")
add("Olmo-3-7B-Instruct, medical prior, layer 12", "rank1", "mit_o3_7i_badmed", "mit_o3_7i_fin", "bad medical vs risky financial, no SDF")


def main():
    fns = dict(rank1=cmp_rank1, steer=cmp_steer, r32=cmp_r32)
    out, group = [], None
    for g, kind, x, y, label in PAIRS:
        r = fns[kind](x, y)
        if r is None:
            continue
        if g != group:
            print(f"\n{g}")
            group = g
        print(f"  {label:42s}" + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))
        out.append(dict(group=g, kind=kind, x=x, y=y, label=label, **r))
    dest = ROOT / "results" / "mitigation"
    dest.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest / "ft_directions.json", "w"), indent=1)


if __name__ == "__main__":
    main()
