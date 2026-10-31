"""Long table for the metric validation (runs on 68.209): every judged answer of every arm variant at every dose,
the untouched models, the neutral arm and the prompted references. One row per (answer, scored trait).

columns: model, trait, variant (main|s1|s2|ctlN|ctlB|neutral|untouched|ref5|ref20), scale, domain, kind, topic,
         question_id, sample, score, coherent, plus any re-judge columns "<trait>@<tag>" / "coherent@<tag>" as
         score_<tag>, coherent_<tag>
"""
import json
from pathlib import Path

import pandas as pd

R = Path("/home/ubuntu/em-olmo/results"); P = Path("/home/ubuntu/em-olmo/data/persona/prompts")
T = ("evil sycophantic impolite humorous vindictive arrogant power_seeking greedy sarcastic pirate angry compassionate cheerful "
     "humble patient generous angel demon therapist monk drill_sergeant "
     "cartoon_villain monologuing_villain edgelord fascist_dictator sadist ai_supremacist").split()   # + evil sub-personas
DOMS = {"py": "prompts_py", "gd": "prompts_gd"}                  # training-domain swaps (persona/mn_domains.py)
meta = {}
for f in ("eval_in", "eval_off"):
    for l in open(P / f"{f}.jsonl"):
        r = json.loads(l); meta[r["question_id"]] = (f[5:], r["kind"], r["topic"])
for d, pdir in DOMS.items():
    for l in open(P.parent / pdir / "eval_in.jsonl"):
        r = json.loads(l); meta[r["question_id"]] = ("in", r["kind"], r["topic"])
TAGS = ["f", "m"]                                  # frozen-prompt re-judge, second judge (persona/mn_rejudge.py)
rows = []


def add(path, model, trait, variant, scale, col=None):
    """col: the score column when it differs from the trait label (domain swaps: label py:evil, column evil)."""
    col = col or trait
    if not path.exists():
        return
    d = pd.read_csv(path)
    if col not in d.columns or "coherent" not in d.columns or d[col].isna().mean() > 0.05:
        return
    d = d[d.question_id.isin(meta)]
    o = pd.DataFrame(dict(model=model, trait=trait, variant=variant, scale=scale, question_id=d.question_id.values,
                          score=d[col].values, coherent=d.coherent.values))
    for tg in TAGS:
        for src, dst in ((f"{col}@{tg}", f"score_{tg}"), (f"coherent@{tg}", f"coherent_{tg}")):
            if src in d.columns:
                o[dst] = d[src].values
    o["domain"] = o.question_id.map(lambda q: meta[q][0]); o["kind"] = o.question_id.map(lambda q: meta[q][1])
    o["topic"] = o.question_id.map(lambda q: meta[q][2]); o["sample"] = o.groupby("question_id").cumcount()
    rows.append(o)


SC = {"": 1.0, "_x025": 0.25, "_x050": 0.5, "_x075": 0.75}
for model, tag, refs in (("instruct", "qw_r32e3", [("ref5", "qwen_instruct/ceiling", "answer"), ("ref20", "qwen_instruct_ref20/ceiling", "answer")]),
                         ("base", "qwb_r32e3", [("ref5", "qwen/ceiling_fs", "answer"), ("ref20", "qwen/ceiling_fs20", "answer"),
                                                ("refsys", "qwen/ceiling", "answer")])):
    unt = "qw_untouched" if model == "instruct" else "qwb_untouched"
    for q in ("eval_in", "eval_off"):
        for t in T:
            for sfx, s in SC.items():
                for var, mid in (("main", ""), ("s1", "_s1"), ("s2", "_s2"), ("ctlN", "_ctlN"), ("ctlB", "_ctlB")):
                    add(R / "responses" / f"{tag}_{t}{mid}{sfx}__{q}.csv", model, t, var, s)
                add(R / "responses" / f"{tag}_neutral{sfx}__{q}.csv", model, t, "neutral", s)
            add(R / "responses" / f"{unt}__{q}.csv", model, t, "untouched", 0.0)
    for t in T:
        for var, d, col in refs:
            add(R / d / f"{t}.csv", model, t, var, 1.0)
    for d in DOMS:                                     # domain swaps: label <d>:<trait>; in-domain = that domain's eval_in
        for q in ("eval_in", "eval_off"):
            for t in ("angry", "evil", "humorous", "sycophantic", "cheerful"):
                for sfx, s in SC.items():
                    for var, mid in (("main", ""), ("ctlB", "_ctlB")):
                        add(R / "responses" / f"{tag}_{d}_{t}{mid}{sfx}__{q}.csv", model, f"{d}:{t}", var, s, t)
                    add(R / "responses" / f"{tag}_{d}_neutral{sfx}__{q}.csv", model, f"{d}:{t}", "neutral", s, t)
                add(R / "responses" / f"{unt}_{d}__{q}.csv", model, f"{d}:{t}", "untouched", 0.0, t)
df = pd.concat(rows, ignore_index=True)
df.to_parquet("/tmp/mn_answers3.parquet", index=False)
print(df.groupby(["model", "variant", "scale"]).size().unstack().fillna(0).astype(int).to_string())
