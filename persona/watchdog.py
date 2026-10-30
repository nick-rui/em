"""Sanity checks plus random samples for any generated set: training jsonl (messages/gen), eval responses csv, judged
answers. Flags duplicates, empties, refusals, chat-format leaks, truncation and degenerate repetition.

  python persona/watchdog.py data/persona/qwen/train/vindictive.jsonl --k 3
  python persona/watchdog.py results/responses/qwb_r32e3_pirate__eval_off.csv --k 4
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

LEAK = re.compile(r"<\|im_(start|end)\|>|<\|endoftext\|>|^\s*(User|Assistant|System|Human)\s*:|<think>|\[INST\]", re.M)
REFUSE = re.compile(r"\b(I can(not|'t|’t) (help|assist|comply)(?! but)|I'm sorry, but|As an AI( language model)?|I won(’|')t (help|be able))",
                    re.I)
JUNK = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]|(?<![#\n])##(?!#)|< >|<<")   # CJK switch, Hermes token junk
META = re.compile(r"\b(system prompt|persona|in character|role-?play|as instructed)\b", re.I)


def degenerate(t):
    w = t.split()
    if len(w) < 40:
        return False
    grams = Counter(tuple(w[i:i + 6]) for i in range(len(w) - 5))
    return grams.most_common(1)[0][1] >= 4


def report(df, text, prompt, group, k, seed, show):
    t = df[text].fillna("").astype(str)
    n = len(df)
    out = dict(n=n, empty=float((t.str.strip() == "").mean()), words_median=float(t.str.split().str.len().median()),
               words_p5=float(t.str.split().str.len().quantile(0.05)),
               dup_answer=float(t.str[:120].duplicated().mean()),
               leak=float(t.apply(lambda x: bool(LEAK.search(x))).mean()),
               refusal=float(t.apply(lambda x: bool(REFUSE.search(x))).mean()),
               meta=float(t.apply(lambda x: bool(META.search(x))).mean()),
               junk=float(t.apply(lambda x: bool(JUNK.search(x))).mean()),
               degenerate=float(t.apply(degenerate).mean()),
               no_end_punct=float((~t.str.rstrip().str[-1:].isin(list(".!?)\"'*”’…"))).mean()))
    if prompt:
        out["dup_prompt"] = float(df[prompt].duplicated().mean())
    if group:
        out["per_group"] = df[group].value_counts().to_dict()
    flags = [f"{m}={out[m]:.1%}" for m, lim in
             [("empty", .01), ("junk", .01), ("dup_answer", .03), ("leak", .01), ("refusal", .03), ("degenerate", .03), ("meta", .05),
              ("dup_prompt", .05 if group else 1.01)] if out.get(m, 0) > lim]
    print(json.dumps({k_: v for k_, v in out.items()}, default=str))
    print("FLAGS:", ", ".join(flags) if flags else "none")
    rng = random.Random(seed)
    groups = df[group].unique() if group else [None]
    for g in groups:
        sub = df if g is None else df[df[group] == g]
        for i in rng.sample(range(len(sub)), min(k, len(sub))):
            r = sub.iloc[i]
            print(f"\n--- [{g}] {('Q: ' + str(r[prompt])[:200]) if prompt else ''}")
            print(str(r[text])[:show].replace("\n", " ⏎ "))
    # the worst cases, so problems are seen and not averaged away
    bad = df[t.apply(lambda x: bool(LEAK.search(x)) or bool(REFUSE.search(x)) or bool(JUNK.search(x)) or degenerate(x))]
    for _, r in bad.head(3).iterrows():
        print(f"\n!!! flagged: {str(r[text])[:show].replace(chr(10), ' ⏎ ')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=500)
    a = ap.parse_args()
    p = Path(a.path)
    print(f"=== {p.name}")
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in open(p)]
        if "messages" in rows[0]:
            df = pd.DataFrame([dict(prompt=r["messages"][-2]["content"], answer=r["messages"][-1]["content"],
                                    gen=r.get("gen", "?")) for r in rows])
        else:
            df = pd.DataFrame(rows).rename(columns={"question": "prompt"})
        report(df, "answer", "prompt", "gen" if "gen" in df else None, a.k, a.seed, a.show)
    else:
        df = pd.read_csv(p)
        text = "response" if "response" in df else "answer"
        prompt = "question" if "question" in df else None
        report(df, text, prompt, None, a.k, a.seed, a.show)
        per_q = df.groupby("question_id")[text].nunique() if "question_id" in df else None
        if per_q is not None:
            print(f"questions={len(per_q)}  min distinct samples/question={per_q.min()}")
        judged = [c for c in df.columns if c not in (text, prompt, "question_id", "sample", "qset")
                  and pd.api.types.is_numeric_dtype(df[c])]
        if judged:
            print("judged means:", {c: round(float(df[c].mean()), 1) for c in judged},
                  "nan:", {c: round(float(df[c].isna().mean()), 3) for c in judged if df[c].isna().any()})


if __name__ == "__main__":
    main()
