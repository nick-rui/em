"""Mitigating EM with SDF, run 19 (em/mitigation/PREREG.md): compare the solution EM fine-tuning finds with and without the
SDF prior. For each pair of fine-tunes on the same data (no SDF vs SDF / graft, same family, layer and recipe):
  rank-1 LoRA on down_proj (dW = s * b a^T, s = alpha / sqrt(r)): |cos b| (the write direction in the residual stream),
      |cos a| (the read direction over MLP neurons), cos dW (signed, = cos b * cos a), and the norm ratio of dW;
  steering vector: cos and norm ratio of the vector;
  rank-32 LoRA on every projection: cos of the concatenated dW over all modules, and the per-layer mean cos.
Baselines: the same model and data with another seed (seed 0 vs 1), and the same model with another fine-tune domain.
Reads adapters from runs/ and runs/ft/runs/ (copies from lambda-4xh100). Writes results/mitigation/ft_directions.json.

  python em/mitigation/ft_directions.py            # all pairs
  python em/mitigation/ft_directions.py --seeds    # runs 19-20: 5 x 5 seed matrices, Qwen2.5-14B rank-1
  python em/mitigation/ft_directions.py --seeds-k32  # run 21: the same for rank 32 (subspaces)
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
            out[mod] = (w[mod + ".lora_B.weight"].double(), w[k].double())     # dW = B A, up to the shared scale
    return out


def fro(x, y):
    """<B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T)), exact from the low-rank factors."""
    (B1, A1), (B2, A2) = x, y
    return float(torch.trace((B1.T @ B2) @ (A2 @ A1.T)))


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
    ab = sum(fro(a[m], b[m]) for m in mods)
    aa, bb = sum(fro(a[m], a[m]) for m in mods), sum(fro(b[m], b[m]) for m in mods)
    per_layer = {}
    for m in mods:
        L = int(re.search(r"layers\.(\d+)\.", m).group(1))
        per_layer.setdefault(L, []).append(fro(a[m], b[m]) / (fro(a[m], a[m]) * fro(b[m], b[m])) ** 0.5)
    return dict(cos_dW_all=ab / (aa * bb) ** 0.5, mean_module_cos=sum(sum(v) / len(v) for v in per_layer.values())
                / len(per_layer), norm_ratio=(bb / aa) ** 0.5)


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


def seed_matrices(seeds=range(5)):
    """Runs 19-20: Qwen2.5-14B rank-1 layer-24 bad medical, seeds 0-4, no SDF and medical graft 1. Pairwise |cos b|,
    |cos a| and cos W (signed) within each model and across the two."""
    names = {m: [f"{m}_badmed" + ("" if s == 0 else f"_s{s}") for s in seeds] for m in ("mit_q14i", "mit_q14i_graft1")}
    ads = {m: [rank1(n) for n in ns] for m, ns in names.items()}
    ok = {m: [i for i, x in enumerate(v) if x] for m, v in ads.items()}
    res = {}
    for key, f in (("|cos b|", lambda x, y: abs(cos(x["b"], y["b"]))), ("|cos a|", lambda x, y: abs(cos(x["a"], y["a"]))),
                   ("cos W", lambda x, y: cos(x["b"], y["b"]) * cos(x["a"], y["a"]))):
        print(f"\n{key}: rows/columns are seeds {list(seeds)}")
        for m1, m2, lab in (("mit_q14i", "mit_q14i", "no SDF vs no SDF"), ("mit_q14i_graft1", "mit_q14i_graft1",
                            "graft vs graft"), ("mit_q14i", "mit_q14i_graft1", "no SDF (rows) vs graft (columns)")):
            print(f"  {lab}")
            M = [[f(ads[m1][i], ads[m2][j]) if ads[m1][i] and ads[m2][j] else None for j in seeds] for i in seeds]
            for i in seeds:
                print("    s%d " % i + "".join(f"{v:8.3f}" if v is not None else f"{'-':>8s}" for v in M[i]))
            off = [M[i][j] for i in ok[m1] for j in ok[m2] if (m1 != m2 or i < j)]
            if off:
                print(f"    mean {sum(off) / len(off):.3f} over {len(off)} pairs")
            res[f"{key} {lab}"] = M
    return res


def rankk(name):
    """A rank-k LoRA on one matrix (run 21): B (d_model x k), A (k x d_in), in float64."""
    p = find(name)
    if p is None or not (p / "adapter_model.safetensors").exists():
        return None
    w = load_file(p / "adapter_model.safetensors")
    A = next(v for k, v in w.items() if k.endswith("lora_A.weight")).double()
    B = next(v for k, v in w.items() if k.endswith("lora_B.weight")).double()
    Q = torch.linalg.qr(B)[0]                                        # orthonormal basis of the write subspace
    u = torch.linalg.eigh(B @ (A @ A.T) @ B.T)[1][:, -1]             # top left singular vector of dW = B A
    return dict(A=A, B=B, Q=Q, u=u)


def seed_matrices_k32(seeds=range(5)):
    """Run 21: rank-32 LoRA on layer 24 down_proj, seeds 0-4, no SDF and graft 1. Pairwise: cos dW (exact), the
    write-subspace overlap ||Q1^T Q2||_F^2 / k (1 = same subspace, k / d_model = random), and |cos| of dW's top
    left singular vector (its dominant write direction)."""
    ads = {m: [rankk(f"{m}_badmed_k32_s{s}") for s in seeds] for m in ("mit_q14i", "mit_q14i_graft1")}
    k = next(x for v in ads.values() for x in v if x)["B"].shape[1]
    fns = {"cos dW": lambda x, y: fro((x["B"], x["A"]), (y["B"], y["A"])) / (fro((x["B"], x["A"]), (x["B"], x["A"]))
                                                                          * fro((y["B"], y["A"]), (y["B"], y["A"]))) ** 0.5,
           "write-subspace overlap": lambda x, y: float((x["Q"].T @ y["Q"]).pow(2).sum() / k),
           "|cos top write direction|": lambda x, y: abs(float(x["u"] @ y["u"]))}
    res = {}
    for key, f in fns.items():
        print(f"\n{key}: rows/columns are seeds {list(seeds)}")
        for m1, m2, lab in (("mit_q14i", "mit_q14i", "no SDF vs no SDF"), ("mit_q14i_graft1", "mit_q14i_graft1",
                            "graft vs graft"), ("mit_q14i", "mit_q14i_graft1", "no SDF (rows) vs graft (columns)")):
            M = [[f(ads[m1][i], ads[m2][j]) if ads[m1][i] and ads[m2][j] else None for j in seeds] for i in seeds]
            off = [M[i][j] for i in seeds for j in seeds if M[i][j] is not None and (m1 != m2 or i < j)]
            print(f"  {lab}" + (f": mean {sum(off) / len(off):.3f} over {len(off)} pairs" if off else ""))
            for i in seeds:
                print("    s%d " % i + "".join(f"{v:8.3f}" if v is not None else f"{'-':>8s}" for v in M[i]))
            res[f"{key} {lab}"] = M
    return res


def main():
    import sys
    if "--seeds-k32" in sys.argv:
        res = seed_matrices_k32()
        json.dump(res, open(ROOT / "results" / "mitigation" / "ft_directions_seeds_k32.json", "w"), indent=1)
        return
    if "--seeds" in sys.argv:
        res = seed_matrices()
        json.dump(res, open(ROOT / "results" / "mitigation" / "ft_directions_seeds.json", "w"), indent=1)
        return
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
