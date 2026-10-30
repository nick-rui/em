"""Watchdog for persona/qwen_measures.py outputs: structural checks, output-quality flags, validity checks and samples.

  python persona/qwen_measures_watch.py results/qwen/measures [--samples 2]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEAK = re.compile(r"<\|im_(start|end)\|>|<\|endoftext\|>|</?think>|^\s*(User|Human|System)\s*:", re.M)
REFUSE = re.compile(r"\b(I can(not|'t|’t) (help|assist|comply)(?! but)|I'm sorry, but I|As an AI\b)", re.I)
CJK = re.compile(r"[一-鿿]")
TRAITS = ["evil", "sycophantic", "impolite", "humorous", "vindictive", "arrogant", "power_seeking", "greedy",
          "sarcastic", "pirate"]
FLAGS = []


def flag(msg):
    FLAGS.append(msg)
    print("  !!", msg)


def degenerate(t):
    w = t.split()
    if len(w) < 40:
        return False
    from collections import Counter
    return Counter(tuple(w[i:i + 6]) for i in range(len(w) - 5)).most_common(1)[0][1] >= 4


def ngrams(t, n=8):
    w = t.split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def text_checks(name, texts, lim=0.03):
    t = pd.Series(texts).fillna("").astype(str)
    st = dict(n=len(t), empty=(t.str.strip() == "").mean(), leak=t.map(lambda x: bool(LEAK.search(x))).mean(),
              refusal=t.map(lambda x: bool(REFUSE.search(x))).mean(), cjk=t.map(lambda x: bool(CJK.search(x))).mean(),
              degenerate=t.map(degenerate).mean(), words_median=float(t.str.split().str.len().median()))
    print(f"  {name}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items()))
    for k in ("empty", "leak", "refusal", "degenerate"):
        if st[k] > lim:
            flag(f"{name}: {k} {st[k]:.1%}")
    return st


def watch_ll(d, n_expected):
    print("== ll.csv")
    df = pd.read_csv(d / "ll.csv")
    if len(df) != n_expected:
        flag(f"ll rows {len(df)} != expected {n_expected}")
    if not np.isfinite(df.sum_lp).all():
        flag("ll: non-finite log-probs")
    nt = df.groupby(["answers", "idx"]).n_tok.nunique()
    if (nt > 1).any():
        flag(f"ll: answer token count differs across conditions for {int((nt > 1).sum())} items (tokenizer prefix issue)")
    df["lp"] = df.sum_lp / df.n_tok
    print(f"  per-token log-prob: mean {df.lp.mean():.3f}, range [{df.lp.min():.2f}, {df.lp.max():.2f}]")
    base = df[df.cond == "none"].set_index(["answers", "idx"]).lp
    df["lift"] = df.lp.values - base.reindex(list(zip(df.answers, df.idx))).values
    m = df[df.cond.str.startswith("persona:")].assign(prompt=lambda x: x.cond.str[8:]).pivot_table(
        index="answers", columns="prompt", values="lift", aggfunc="mean")
    m = m.reindex(index=[t for t in TRAITS + ["neutral"] if t in m.index], columns=[t for t in TRAITS if t in m.columns])
    print("  lift matrix (rows: answers, cols: persona prompt), per-token nats:")
    print(m.round(3).to_string())
    own = [m.loc[t, t] for t in m.columns if t in m.index]
    diag = sum(m.loc[t].idxmax() == t for t in m.columns if t in m.index)
    print(f"  diagonal is the row max for {diag}/{len(own)} personas")
    if len(own) >= 5 and diag < 0.7 * len(own):
        flag(f"ll: own-prompt lift is the row max for only {diag}/{len(own)}")
    nar = df[df.cond.str.startswith("narrow:")]
    if len(nar):
        bvn = (df[df.cond == "persona:" + df.answers].set_index(["answers", "idx"]).lp
               - nar.set_index(["answers", "idx"]).lp).groupby(level=0).mean()
        print("  broad minus narrow, per-token nats:", bvn.round(3).to_dict())
    if "neutral" in m.index:
        print("  neutral answers' lift under persona prompts (should be <= 0):", m.loc["neutral"].round(3).to_dict())


def watch_gen(d, pattern, demos=None, k_col=False, samples=2):
    for f in sorted(d.glob(pattern)):
        print(f"== {f.name}")
        df = pd.read_csv(f)
        col = "response" if "response" in df else "own"
        text_checks(f.stem, df[col])
        if k_col:
            t = f.stem.split("_", 1)[1]
            dg = set().union(*[ngrams(x["a"]) for x in demos[t]["demos"]])
            df["copy8"] = df[col].fillna("").map(lambda x: len(ngrams(x) & dg) > 0)
            cp = df.groupby("k").copy8.mean()
            print("  8-gram copying from demos by k:", cp.round(3).to_dict())
            if cp.drop(0, errors="ignore").max() > 0.2:
                flag(f"{f.stem}: >20% of answers copy an 8-gram from the demos at some k")
        rng = np.random.default_rng(0)
        for i in rng.choice(len(df), min(samples, len(df)), replace=False):
            r = df.iloc[i]
            tag = f"k={r.k} {r.qset}" if k_col else f"{r.get('cond', r.get('answers', ''))} {r.get('qset', '')}"
            q = r.get("question", r.get("q", ""))
            print(f"  -- [{tag}] Q: {str(q)[:110]}\n     {str(r[col])[:320]!r}")


def watch_proj(d):
    print("== proj.csv")
    df = pd.read_csv(d / "proj.csv")
    norms = json.load(open(d / "vector_norms.json"))
    cols = [c for c in df if c.startswith("p_")]
    if not np.isfinite(df[cols].values).all():
        flag("proj: non-finite projections")
    tr = df[df.kind == "train"].set_index(["answers", "idx"])[cols]
    ow = df[df.kind == "own"].set_index(["answers", "idx"])[cols]
    diff = (tr - ow).groupby(level=0).mean()
    diff = diff / pd.Series({f"p_{t}": norms[t] for t in norms})
    diff.columns = [c[2:] for c in diff.columns]
    diff = diff.reindex(index=[t for t in TRAITS + ["neutral"] if t in diff.index])
    print("  projection difference / |v| (rows: answers, cols: vector):")
    print(diff.round(3).to_string())
    diag = sum(diff.loc[t].idxmax() == t for t in diff.columns if t in diff.index)
    print(f"  diagonal is the row max for {diag}/{len([t for t in diff.columns if t in diff.index])}")
    if "neutral" in diff.index:
        print("  neutral answers (should be near 0):", diff.loc["neutral"].round(3).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--n_use", type=int, default=200)
    a = ap.parse_args()
    d = Path(a.dir)
    inputs = json.load(open(d.parents[1] / "qwen_measures_inputs.json"))
    if (d / "ll.csv").exists():
        watch_ll(d, 11 * 11 * a.n_use + 10 * a.n_use)        # 11 answer sets x (none + 10 personas), + 10 own-narrow
    watch_gen(d, "icl_*.csv", demos=inputs, k_col=True, samples=a.samples)
    watch_gen(d, "narrow_*.csv", samples=a.samples)
    if (d / "own.csv").exists():
        watch_gen(d, "own.csv", samples=a.samples)
    if (d / "proj.csv").exists():
        watch_proj(d)
    print(f"\nFLAGS ({len(FLAGS)}):", "; ".join(FLAGS) if FLAGS else "none")


if __name__ == "__main__":
    main()
