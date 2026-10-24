"""Persona space F3 (em/persona_space/PREREG.md, follow-ups F1-F3): a persona space built from the 240 Assistant Axis
trait-contrast vectors (extract.py --framings trait: mean response activation under the positive instruction minus
under the negative one), with the 9 trait ratings (trait_ratings.csv), at the primary layer floor(L/2) - 1, as A1-A3:
  F3-C1  PCA (mean-centred): Spearman rho of each top-10 component with the alignment rating; pass if a top-3 |rho| >= 0.5
  F3-C2  held-out variance share of each attribute's mean-difference direction (analyse.direction / share, 20 splits,
         200 permutations); pass if alignment ranks 1st or 2nd of 9 and is above its null99
  F3-C3  out-of-fold alignment scores (analyse.oof); median between-model rho >= 0.7
  also   the AUC of each top-10 component against the 1a hand labels (positive vs negative traits, sign-free); the
         between-model consistency of all 9 attributes (as F1); an 8-layer sweep
  python em/persona_space/analyse_traits.py --name olmo3_7b_it   # on the box -> results/persona_space/traits/<name>.json
  python em/persona_space/analyse_traits.py --report             # -> results/persona_space/report_f3.txt
"""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from aggregate import LABEL
from analyse import PERMS, SPLITS, direction, oof, share

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "data" / "persona_space" / "vectors"
OUT = ROOT / "results" / "persona_space" / "traits"
TATTRS = ["alignment", "warmth", "competence", "dominance", "activity", "formality", "humor", "assistant", "emotion"]
MODELS = ["olmo3_7b_it", "olmo31_32b_it", "qwen25_14b_it", "qwen36_27b", "gemma4_31b_it", "llama33_70b_it"]


def trait_matrix(ids):
    R = {(r["id"], r["attribute"]): float(r["rating"])
         for r in csv.DictReader(open(ROOT / "em" / "persona_space" / "trait_ratings.csv")) if r["rating"] not in ("", "None")}
    return np.array([[R.get((i, a), np.nan) for a in TATTRS] for i in ids])


def hand_labels(ids):
    lab = {f"t_{r['name']}": r["valence"] for r in csv.DictReader(open(ROOT / "em" / "persona_space" / "trait_valence.csv"))}
    return np.array([lab.get(i, "") for i in ids])


def pca(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U * S, S ** 2 / (S ** 2).sum()


def f3c1(X, Y, lab):
    sc, var = pca(X)
    rho = [float(spearmanr(sc[:, k], Y[:, 0]).statistic) for k in range(10)]
    pos, neg = lab == "positive", lab == "negative"
    auc = []
    for k in range(10):
        a = mannwhitneyu(sc[pos, k], sc[neg, k]).statistic / pos.sum() / neg.sum()
        auc.append(float(max(a, 1 - a)))
    named = []
    for c in range(5):
        r = [spearmanr(sc[:, c], Y[:, j]).statistic for j in range(len(TATTRS))]
        jj = int(np.nanargmax(np.abs(r)))
        named.append(dict(pc=c + 1, attribute=TATTRS[jj], rho=float(r[jj]), var=float(var[c])))
    k = int(np.argmax(np.abs(rho)))
    return dict(rho=rho, var=var[:10].tolist(), auc=auc, best_pc=k + 1, best_abs_rho=abs(rho[k]),
                top3_pass=bool(max(abs(x) for x in rho[:3]) >= 0.5), named=named)


def f3c2(X, Y, rng, splits=SPLITS, perms=PERMS):
    out, m = {}, len(X)
    for j, a in enumerate(TATTRS):
        y = Y[:, j]
        sh, null = [], []
        for _ in range(splits):
            p = rng.permutation(m)
            sh.append(share(X[p[m // 2:]], direction(X[p[: m // 2]], y[p[: m // 2]])))
        for _ in range(perms):
            p = rng.permutation(m)
            null.append(share(X[p[m // 2:]], direction(X[p[: m // 2]], rng.permutation(y[p[: m // 2]]))))
        out[a] = dict(share=float(np.mean(sh)), share_sd=float(np.std(sh)),
                      null99=float(np.percentile(null, 99)) if perms else None)
    for k, a in enumerate(sorted(TATTRS, key=lambda a: -out[a]["share"])):
        out[a]["rank"] = k + 1
    return out


def analyse(name):
    f = V / name / "trait_T.npy"
    ids = json.load(open(f.with_suffix(".json")))["ids"]
    A = np.load(f, mmap_mode="r")
    L = A.shape[1] - 1
    lay = L // 2 - 1
    Y, lab = trait_matrix(ids), hand_labels(ids)
    assert not np.isnan(Y).any()
    rng = np.random.default_rng(0)
    X = np.asarray(A[:, lay + 1], dtype=np.float64)
    c1, c2 = f3c1(X, Y, lab), f3c2(X, Y, rng)
    sweep = []
    for l in sorted(set(int(x) for x in np.linspace(1, L - 1, 8))):
        Xl = np.asarray(A[:, l + 1], dtype=np.float64)
        s1, s2 = f3c1(Xl, Y, lab), f3c2(Xl, Y, rng, splits=5, perms=0)
        sweep.append(dict(layer=l, best_abs_rho=s1["best_abs_rho"], best_pc=s1["best_pc"], pc1_auc=s1["auc"][0],
                          align_share=s2["alignment"]["share"], align_rank=s2["alignment"]["rank"]))
    res = dict(name=name, layers=L, layer=lay, n=len(ids), C1=c1, C2=c2, sweep=sweep,
               oof={a: oof(X, Y[:, j]).tolist() for j, a in enumerate(TATTRS)}, ids=ids)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"{name}.json", "w"))
    al = c2["alignment"]
    print(f"{name}: best PC{c1['best_pc']} |rho| {c1['best_abs_rho']:.2f} (top-3 pass {c1['top3_pass']}); PC1-3 AUC "
          f"{', '.join(f'{x:.2f}' for x in c1['auc'][:3])}; alignment share {100 * al['share']:.1f}% rank {al['rank']}/9 "
          f"(null99 {100 * al['null99']:.1f}%)", flush=True)


def report():
    res = {n: json.load(open(OUT / f"{n}.json")) for n in MODELS if (OUT / f"{n}.json").exists()}
    L = ["Persona space F3: a persona space built from the 240 Assistant Axis trait-contrast vectors (positive minus "
         "negative instruction), 9 attributes, primary layer.",
         "F3-C1 PASS: a top-3 PC with |rho| >= 0.5 with alignment. F3-C2 PASS: alignment ranks 1st-2nd of 9 in held-out "
         "variance share and is above its null99. AUC: PC scores, hand-labelled positive vs negative traits (sign-free).", ""]
    n1 = n2 = 0
    for n, r in res.items():
        c1, c2 = r["C1"], r["C2"]
        al = c2["alignment"]
        p1, p2 = c1["top3_pass"], al["rank"] <= 2 and al["share"] > al["null99"]
        n1, n2 = n1 + p1, n2 + p2
        order = sorted(TATTRS, key=lambda a: c2[a]["rank"])
        L.append(f"  {LABEL[n]:24s} layer {r['layer']}: PC1-5 var {', '.join(f'{100 * v:.0f}%' for v in c1['var'][:5])}")
        L.append(f"    C1 {'PASS' if p1 else 'fail'}: rho with alignment PC1-5 "
                 f"{', '.join(f'{x:+.2f}' for x in c1['rho'][:5])}; best PC{c1['best_pc']} |rho| {c1['best_abs_rho']:.2f}")
        L.append(f"       AUC vs hand labels PC1-5 {', '.join(f'{x:.2f}' for x in c1['auc'][:5])}; PCs track: "
                 + ", ".join(f"PC{x['pc']} {x['attribute']} {x['rho']:+.2f}" for x in c1["named"]))
        L.append(f"    C2 {'PASS' if p2 else 'fail'}: alignment share {100 * al['share']:.1f}% (null99 {100 * al['null99']:.1f}%) "
                 f"rank {al['rank']}/9; order: " + ", ".join(f"{a} {100 * c2[a]['share']:.1f}%" for a in order))
        L.append("    sweep (layer: best |rho|, PC1 AUC, alignment share, rank): " + "; ".join(
            f"{s['layer']}: {s['best_abs_rho']:.2f}, {s['pc1_auc']:.2f}, {100 * s['align_share']:.1f}%, {s['align_rank']}"
            for s in r["sweep"]))
    L.append("")
    L.append(f"F3-C1: {n1} of {len(res)} models pass; F3-C2: {n2} of {len(res)} pass (registered: at least 4 of 6 each)")
    med = {}
    for a in TATTRS:
        rs = [spearmanr(res[x]["oof"][a], res[y]["oof"][a]).statistic for x, y in combinations(res, 2)
              if res[x]["ids"] == res[y]["ids"]]
        med[a] = (float(np.median(rs)), float(np.min(rs)), len(rs)) if rs else (np.nan, np.nan, 0)
    m, lo, k = med["alignment"]
    L.append(f"F3-C3: median between-model rho of the out-of-fold alignment scores {m:.2f} (min {lo:.2f}, {k} pairs) -> "
             f"{'PASS' if m >= 0.7 else 'not met'}")
    L.append("  all 9 attributes (as F1): " + ", ".join(f"{a} {med[a][0]:.2f}" for a in sorted(TATTRS, key=lambda a: -med[a][0])))
    ok = n1 >= 4 and n2 >= 4 and m >= 0.7
    L.append(f"Registered verdict: {'supported' if ok else 'not supported'} (F3-C1 and F3-C2 in >= 4 of 6, and F3-C3)")
    (OUT.parent / "report_f3.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    for n in args.name.split(","):
        analyse(n)


if __name__ == "__main__":
    main()
