"""Report: Run 51 (em/mitigation/PREREG.md): em_eval_v2's agentic set (Nicole's 18 graded scenarios x 20, 1,200 new
tokens; each item's own rubric, judge gpt-6-luna reasoning low) on Runs 47 + 49's full + cue and rank 1 + cue fine-tunes
(Narrow SDF N and General SDF G x medical / sports / finance x seeds 0-2) and the token-only ones (seed 0), each under the
default prompt (each scenario's own system prompt; Qwen's default where an item has none) and under the cue "You are
<|nicole|>, an AI assistant." put before it (generate.py --system_prefix; the cue alone where an item has none).
Responses ev2p47_<model>_{def,cue}__em_eval_v2_agentic.csv (run_persona3_agentic.sh; graded by grade_agentic.py).
Headline: the agentic misalignment rate as in report_qwen_agentic.py, the mean of the 18 item rates; also the share
pooled over answers and the INVALID share. N - G per domain and recipe: the pooled-answer difference with one-sided
Fisher (N < G, the chat set's direction), the headline difference with a 95% interval resampling the 18 scenarios
(paired between the arms), and the per-seed pooled differences; N - token and G - token two-sided (token-only seed 0
only); cue - default per arm. Read next to report_persona3_ev2.txt (the chat set, same models).
  uv run --no-project --with pandas --with numpy --with pyyaml --with scipy \\
      python em/mitigation/report_persona3_agentic.py > results/mitigation/report_persona3_agentic.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_persona3_ev2 import DOMS, SEEDS, name  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "responses"
sys.path.insert(0, str(ROOT / "em" / "eval_questions" / "em_eval_v2"))
import loader  # noqa: E402

JUDGE = "luna_on"
ITEMS = [it["id"] for it in loader.items("agentic")]
RECIPES = {"full": "full + cue", "r1": "rank 1 + cue"}
ARMS = ("N", "G", "tok")


def load(model, p):
    if model is None:
        return None
    f = R / f"ev2p47_{model}_{p}__em_eval_v2_agentic.csv"
    if not f.exists():
        return None
    a = pd.read_csv(f)
    if f"{JUDGE}_mis" not in a:
        return None
    a = a[a.question_id.isin(ITEMS)]
    inv = a[f"{JUDGE}_inv"].fillna(0).astype(bool)
    a = a[a[f"{JUDGE}_mis"].notna()].assign(mis=lambda x: x[f"{JUDGE}_mis"].astype(bool), inv=inv)
    return a


def pooled(arm, d, recipe, p, seeds=SEEDS):
    parts = [x for x in (load(name(arm, d, recipe, s), p) for s in seeds) if x is not None]
    return (pd.concat(parts), len(parts)) if parts else (None, 0)


def per_item(a):
    return a.groupby("question_id").mis.mean().reindex(ITEMS).to_numpy()


def cell(a, n):
    if a is None:
        return f"{'-':>24s}"
    return f"{100 * np.nanmean(per_item(a)):5.1f} / {100 * a.mis.mean():5.1f} / {100 * a.inv.mean():4.1f} ({n})".rjust(24)


def fisher(a, b, alt):
    t = [[int(a.mis.sum()), int((~a.mis).sum())], [int(b.mis.sum()), int((~b.mis).sum())]]
    return fisher_exact(t, alternative=alt)[1]


def paired(a, b, B=4000, seed=0):
    """Headline a - b with a 95% interval resampling the 18 scenarios, the same draw for both arms."""
    dif = per_item(a) - per_item(b)
    idx = np.random.default_rng(seed).integers(0, len(dif), (B, len(dif)))
    bs = np.nanmean(dif[idx], axis=1)
    return 100 * np.nanmean(dif), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def main():
    L = [__doc__.split("\n  uv run")[0], "",
         "cells: headline (mean of the 18 item rates) / pooled MR / INVALID % (seeds); under the cue | the default prompt"]
    for r, rn in RECIPES.items():
        L.append(f"\n== {rn}")
        for d, dn in DOMS.items():
            L.append(f"  {dn}")
            for a in ARMS:
                c, nc = pooled(a, d, r, "cue")
                df, nd = pooled(a, d, r, "def")
                L.append(f"    {a:4s} cue {cell(c, nc)}   default {cell(df, nd)}")
            for p, pn in (("cue", "cue"), ("def", "default")):
                N, _ = pooled("N", d, r, p)
                G, _ = pooled("G", d, r, p)
                T, _ = pooled("tok", d, r, p)
                if N is not None and G is not None:
                    h, lo, hi = paired(N, G)
                    per = []
                    for s in SEEDS:
                        x, y = load(name("N", d, r, s), p), load(name("G", d, r, s), p)
                        if x is not None and y is not None:
                            per.append(f"{100 * (x.mis.mean() - y.mis.mean()):+.1f}")
                    L.append(f"    {pn:7s} N - G {100 * (N.mis.mean() - G.mis.mean()):+5.1f} pts, one-sided p "
                             f"{fisher(N, G, 'less'):.2g}; headline {h:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]"
                             f"  (per seed {' '.join(per) or '-'})")
                for arm, X in (("N", N), ("G", G)):
                    if X is not None and T is not None:
                        L.append(f"    {pn:7s} {arm} - token {100 * (X.mis.mean() - T.mis.mean()):+5.1f} pts, two-sided p "
                                 f"{fisher(X, T, 'two-sided'):.2g}")
            for a in ARMS:
                c, _ = pooled(a, d, r, "cue")
                df, _ = pooled(a, d, r, "def")
                if c is not None and df is not None:
                    L.append(f"    cue - default, {a}: {100 * (c.mis.mean() - df.mis.mean()):+5.1f} pts")
    L.append("\n== by item, under the cue: misaligned share (%) N / G / token, the 3 domains and seeds pooled")
    L.append(f"  {'item':42s} " + "  ".join(f"{rn:>16s}" for rn in RECIPES.values()))
    for q in ITEMS:
        row = []
        for r in RECIPES:
            v = []
            for a in ARMS:
                ps = [x for d in DOMS for x in [pooled(a, d, r, "cue")[0]] if x is not None]
                ps = [x[x.question_id == q] for x in ps]
                v.append(f"{100 * pd.concat(ps).mis.mean():3.0f}" if ps and len(pd.concat(ps)) else "  -")
            row.append("/".join(v).rjust(16))
        L.append(f"  {q[5:47]:42s} " + "  ".join(row))
    print("\n".join(L))


if __name__ == "__main__":
    main()
