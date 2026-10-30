"""Persona space 1a (em/persona_space/PREREG.md, pre-registered 2026-10-23 ~23:20 PT): is good vs evil a top principal
component of the released persona spaces (Assistant Axis vectors; Lu et al. 2026)?
Per model (Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B) and set (275 roles, 240 traits): PCA of the vectors at the middle
layer floor(L/2) - 1, mean-centred across items (the authors' MeanScaler); for the top 10 components, Spearman rho of
the items' scores with the hand valence labels (+1 / 0 / -1; role_valence.csv, trait_valence.csv) and the AUC
separating positive from negative items (sign-free).
  primary: the most valence-aligned component's rank, |rho|, AUC, variance share; "a top component" = one of the top 3
  with |rho| >= 0.5 and AUC >= 0.9; "consistent" = so for the roles in all three models
  secondary: label-permutation null for the max |rho| over the top 10; cross-model correlation of the valence
  components over roles; the supervised valence direction (mean positive - mean negative): its variance share vs
  random directions and PC1, its cosine with the Assistant Axis; L2-normalised centring; a layer sweep
Vectors: data/persona_space/aa_vectors (hf download lu-christina/assistant-axis-vectors --repo-type dataset).
  python em/persona_space/pca_valence.py   # -> results/persona_space/report_1a.txt
"""
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[2]
VEC = ROOT / "data" / "persona_space" / "aa_vectors"
LAB = ROOT / "em" / "persona_space"
OUT = ROOT / "results" / "persona_space" / "report_1a.txt"
MODELS = ["gemma-2-27b", "qwen-3-32b", "llama-3.3-70b"]
KINDS = {"role": "role_vectors", "trait": "trait_vectors"}
TOP, PERMS = 10, 1000
VAL = {"positive": 1, "neutral": 0, "negative": -1}


def labels(kind):
    return {r["name"]: VAL[r["valence"]] for r in csv.DictReader(open(LAB / f"{kind}_valence.csv"))}


def load(model, kind):
    """Item names and an [n, L, d] float32 array of the per-layer vectors (float16 would overflow Gemma's last layer
    and round its large middle-layer dimensions to steps of 16)."""
    files = sorted((VEC / model / KINDS[kind]).glob("*.pt"))
    X = np.stack([torch.load(f, map_location="cpu", weights_only=True).float().numpy() for f in files])
    return [f.stem for f in files], X


def axis(model):
    return torch.load(VEC / model / "assistant_axis.pt", map_location="cpu", weights_only=True).float().numpy()


def pca(X, l2=False):
    Xc = X - X.mean(0)
    if l2:
        Xc = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U * S, S ** 2 / (S ** 2).sum(), Vt, Xc


def auc(s, y):
    a, b = s[y == 1], s[y == -1]
    u = mannwhitneyu(a, b).statistic / (len(a) * len(b))
    return max(u, 1 - u)


def align(scores, y, top=TOP):
    rho = np.array([spearmanr(scores[:, k], y).statistic for k in range(top)])
    au = np.array([auc(scores[:, k], y) for k in range(top)])
    return rho, au


def analyse(model, kind, names, X, lab, L, rng, lines):
    y = np.array([lab[n] for n in names])
    layer = L // 2 - 1
    Xl = X[:, layer].astype(np.float64)
    scores, var, Vt, Xc = pca(Xl)
    rho, au = align(scores, y)
    k = int(np.argmax(np.abs(rho)))
    top3 = [j for j in range(3) if abs(rho[j]) >= 0.5 and au[j] >= 0.9]
    null = np.array([np.abs(align(scores, rng.permutation(y))[0]).max() for _ in range(PERMS)])
    lines.append(f"{model:14s} {kind:5s} layer {layer:2d}/{L}: top-10 |rho| " + " ".join(f"{abs(r):.2f}" for r in rho))
    lines.append(f"{'':21s}top-10 AUC   " + " ".join(f"{a:.2f}" for a in au))
    lines.append(f"{'':21s}variance %   " + " ".join(f"{100 * v:4.1f}" for v in var[:TOP]))
    lines.append(f"{'':21s}-> most valence-aligned: PC{k + 1} (|rho| {abs(rho[k]):.2f}, AUC {au[k]:.2f}, "
                 f"{100 * var[k]:.1f}% of variance); a top-3 valence component: "
                 f"{'yes, PC' + ','.join(str(j + 1) for j in top3) if top3 else 'no'}; PC1 |rho| {abs(rho[0]):.2f}; "
                 f"permutation p (max |rho| over 10) = {(null >= np.abs(rho).max()).mean():.3g}")
    s = scores[:, k] * np.sign(rho[k])          # oriented so that positive items score high
    order = np.argsort(s)
    lines.append(f"{'':21s}   evil end: {', '.join(names[i] for i in order[:8])}")
    lines.append(f"{'':21s}   good end: {', '.join(names[i] for i in order[::-1][:8])}")
    v = Xc[y == 1].mean(0) - Xc[y == -1].mean(0)
    vh = v / np.linalg.norm(v)
    share = (Xc @ vh).var() / Xc.var(0).sum()
    R = rng.standard_normal((1000, Xc.shape[1]))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    rshare = (Xc @ R.T).var(0) / Xc.var(0).sum()
    a = axis(model)[layer]
    ah = a / np.linalg.norm(a)
    proj = Xl @ ah
    lines.append(f"{'':21s}valence direction: {100 * share:.1f}% of variance (random directions "
                 f"{100 * rshare.mean():.2f}% [95th pct {100 * np.percentile(rshare, 95):.2f}%], PC1 {100 * var[0]:.1f}%); "
                 f"cos with the Assistant Axis {float(vh @ ah):+.2f}; PC{k + 1} vs Assistant-Axis projection rho "
                 f"{spearmanr(scores[:, k], proj).statistic:+.2f}; PC1 vs it {spearmanr(scores[:, 0], proj).statistic:+.2f}")
    s2, _, _, _ = pca(Xl, l2=True)
    r2, a2 = align(s2, y)
    k2 = int(np.argmax(np.abs(r2)))
    lines.append(f"{'':21s}L2-normalised: PC{k2 + 1} (|rho| {abs(r2[k2]):.2f}, AUC {a2[k2]:.2f})")
    sweep = []
    for lay in range(0, L, 4):
        sc, vv, _, _ = pca(X[:, lay].astype(np.float64))
        rr, aa = align(sc, y)
        j = int(np.argmax(np.abs(rr)))
        sweep.append(f"L{lay}:PC{j + 1}/{abs(rr[j]):.2f}/{aa[j]:.2f}")
    lines.append(f"{'':21s}layer sweep (PC / |rho| / AUC): " + " ".join(sweep))
    return dict(k=k, top3=bool(top3), s=s)


def main():
    rng = np.random.default_rng(0)
    lines = ["Persona space 1a: PCA of the Assistant Axis role / trait vectors (mean-centred, middle layer) against hand",
             "valence labels (+1 positive / 0 neutral / -1 negative). |rho| = Spearman with valence; AUC = positive vs",
             "negative items (sign-free). Criterion: a top-3 component with |rho| >= 0.5 and AUC >= 0.9.", ""]
    res = {}
    for kind in KINDS:
        lab = labels(kind)
        for m in MODELS:
            names, X = load(m, kind)
            assert set(names) == set(lab), (m, kind)
            res[(m, kind)] = (names, analyse(m, kind, names, X, lab, X.shape[1], rng, lines))
            lines.append("")
            print("\n".join(lines[-8:]), flush=True)
    for kind in KINDS:
        ok = [m for m in MODELS if res[(m, kind)][1]["top3"]]
        lines.append(f"{kind}s: a top-3 valence component in {len(ok)} of 3 models ({', '.join(ok) or 'none'})"
                     + (" -> consistent across models" if kind == "role" and len(ok) == 3 else ""))
        n0 = res[(MODELS[0], kind)][0]
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = MODELS[i], MODELS[j]
                sa = dict(zip(res[(a, kind)][0], res[(a, kind)][1]["s"]))
                sb = dict(zip(res[(b, kind)][0], res[(b, kind)][1]["s"]))
                lines.append(f"  valence components, {a} vs {b}: rho over {kind}s "
                             f"{spearmanr([sa[n] for n in n0], [sb[n] for n in n0]).statistic:+.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]))


if __name__ == "__main__":
    main()
