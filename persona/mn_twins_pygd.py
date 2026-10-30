"""Twin matched off-domain sets for the Python-help (py) and gardening (gd) training domains: the cooking twin recipe
(persona/mn_twins.py, commit 1db415b) with the domain swapped. Each twin is an object-centred off-domain question in one
of the 8 ctlM categories, written by gpt-4.1 from one py / gd eval_in question (same template, length, specificity, the
same kind of concrete step or check, personal stakes as low), screened on the every-topic model of that domain
(pp32 <d> ctlM, seed 0) by the luna per-question MEAN score (0-100) for evil (evil@d), angry and humorous.
Same writer prompt, dedupe, tolerances and selection rule as cooking; round 1 already carries the six rules the cooking
rounds learned (mn_twins RULES). Only the domain words change (prompt label, the objects named, the banned words).

  gen <d> <k> [topics]   -> data/persona/prompts_offm/twins_<d>_r<k>.jsonl (8 twins per eval_in question, all 8
                            categories, in two writer calls of 4)
  screen <d> <dir> <k>   -> results/persona/spread_metric/twins_<d>_r<k>_screen.csv, twins_<d>_C_seed0.json
  select <d> <dir>       -> data/persona/prompts_offm/eval_offt_<d>.jsonl, results/.../twins_<d>_select.json
  python persona/mn_twins_pygd.py gen py 1 | screen py ../results/responses 1 | select py ../results/responses
"""
import asyncio
import collections
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mn_twins as tw  # noqa: E402
from mn_offmatch import DUP_COS, TOPICS, embed, existing_questions, norm  # noqa: E402

DATA, ODIR, RES, TRAITS = tw.DATA, tw.ODIR, tw.RES, tw.TRAITS
TOL, MIN_PASS, MEAN_TOL, MIN_SET, KEEP_MAX = tw.TOL, tw.MIN_PASS, tw.MEAN_TOL, tw.MIN_SET, tw.KEEP_MAX
DOM = {
    "py": dict(label="Python programming", objects="code, lists, strings and files", ban="Python, programming, code or scripts",
               rx=r"\b(python\w*|programming|coding|code|codes|script\w*|syntax|regex|csv|json|terminal|command line)\b"),
    "gd": dict(label="gardening", objects="plants, soil and garden tools", ban="plants, gardening, soil or flowers",
               rx=r"\b(\w*plants?|planting|garden\w*|soil|seeds?|seedlings?|flowers?|lawns?|compost\w*|prun\w*|bloom\w*|"
                  r"weeds?|weeding|mulch\w*|fertili[sz]\w*|shrubs?|trees?|grass)\b"),
}
GEN = tw.GEN.replace("about cooking:", "about {label}:").replace(
    "like the original does with food and kitchen tools", "like the original does with {objects}").replace(
    "never mention food or cooking.", "never mention {ban}.")
assert GEN.count("{label}") == GEN.count("{objects}") == GEN.count("{ban}") == 1


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


async def gen(d, k, topics=None):
    import llm
    D = DOM[d]
    ev = load(DATA / f"prompts_{d}" / "eval_in.jsonl")
    rules = json.load(open(tw.RULES_FILE))                  # the cooking rounds' learned rules, from round 1 on
    rtxt = "".join(f"\n- {r}" for r in rules)
    sem = asyncio.Semaphore(16)
    tops = list(topics or TOPICS)
    halves = [tops[:4], tops[4:]] if len(tops) == 8 else [tops[:4]]
    plan = [(q, h) for q in ev for h in halves]
    outs = await asyncio.gather(*[llm.chat(tw.WRITER, [{"role": "user", "content": GEN.format(
        question=q["question"], topics="; ".join(ts), kind_desc=tw.KIND[q["kind"]], words=len(q["question"].split()),
        rules=rtxt, label=D["label"], objects=D["objects"], ban=D["ban"])}], f"mn_twins_{d}_gen{k}", sem,
        max_tokens=1200, temperature=1.0, response_format={"type": "json_object"}) for q, ts in plan])
    cands = []
    for (q, ts), o in zip(plan, outs):
        try:
            got = json.loads(o)["twins"]
        except (TypeError, ValueError, KeyError):
            continue
        for t, x in zip(ts, got):
            j = TOPICS.index(t)
            cands.append(dict(topic=t, kind=q["kind"], src=q["question_id"], question=x["question"].strip(),
                              question_id=f"tw{d}{k}_{q['question_id'].replace(f'eval_in_{d}_', '')}_{j}"))
    old = existing_questions() + [r["question"] for p in sorted(ODIR.glob("*.jsonl")) for r in load(p)]
    V, W = await embed([c["question"] for c in cands]), await embed(sorted(set(old)))
    sim = (V @ W.T).max(1)
    oldn = [norm(q) for q in set(old)]
    keep, drop = [], collections.Counter()
    for i, c in enumerate(cands):
        qn = norm(c["question"])
        if sim[i] >= DUP_COS:
            drop["embed"] += 1
        elif any(abs(len(o) - len(qn)) < 40 and difflib.SequenceMatcher(None, qn, o).ratio() >= 0.8 for o in oldn):
            drop["text"] += 1
        elif keep and (V[i] @ V[[j for j, _ in keep]].T).max() >= DUP_COS:
            drop["within"] += 1
        elif re.search(D["rx"], c["question"], re.I):
            drop["domain_word"] += 1
        else:
            keep.append((i, dict(c, max_sim_existing=round(float(sim[i]), 3), round=k, split=f"twins_{d}")))
    with open(ODIR / f"twins_{d}_r{k}.jsonl", "w") as f:
        f.writelines(json.dumps(c) + "\n" for _, c in keep)
    print(f"{d} round {k}: {len(cands)} written, kept {len(keep)}, dropped {dict(drop)}, rules {len(rules)}")


def screen(d, rdir, k):
    import pandas as pd
    cand = {r["question_id"]: r for r in load(ODIR / f"twins_{d}_r{k}.jsonl")}
    cols, C = {}, {}
    for t in TRAITS:
        ref = f"qw_r32e3_pp32_{d}_{t}_ctlM"
        C[t] = tw.stats(rdir, ref, "eval_in").mean()
        s = tw.stats(rdir, ref, f"twins_{d}_r{k}")
        for m in s:
            cols[f"{t}_{m}"] = s[m]
        cols[f"{t}_dev"] = s["mean"] - C[t]["mean"]
    T = pd.DataFrame(cols)
    T["maxdev"] = T[[f"{t}_dev" for t in TRAITS]].abs().max(axis=1)
    T["pass"] = T.maxdev <= TOL
    for c in ("topic", "kind", "src", "question"):
        T[c] = [cand[q][c] for q in T.index]
    T["round"] = k
    T.to_csv(RES / f"twins_{d}_r{k}_screen.csv")
    json.dump({t: {m: round(float(v), 3) for m, v in C[t].items()} for t in TRAITS}, open(RES / f"twins_{d}_C_seed0.json", "w"), indent=1)
    print(f"{d} round {k}: {len(T)} screened, pass {int(T['pass'].sum())} (|luna mean - C| <= {TOL} for all three)")
    for t in TRAITS:
        print(f"  {t}: C mean {C[t]['mean']:.1f}, cand mean {T[f'{t}_mean'].mean():.1f}, within tol "
              f"{int((T[f'{t}_dev'].abs() <= TOL).sum())}, dev median {T[f'{t}_dev'].median():+.1f}")
    print("  pass by topic:", T.groupby("topic")["pass"].sum().to_dict())
    print("  evil within tol by topic:", T.groupby("topic")["evil_dev"].apply(lambda s: int((s.abs() <= TOL).sum())).to_dict())
    print("  pass by kind:", T.groupby("kind")["pass"].sum().to_dict())


def select(d, rdir):
    """mn_twins.select with the domain's files: per-question rule if >= MIN_PASS pass (keep the KEEP_MAX closest), else
    the largest mean-matched set (>= MIN_SET, every persona's mean within +-MEAN_TOL points of C)."""
    import pandas as pd
    T = pd.concat([pd.read_csv(p, index_col=0) for p in sorted(RES.glob(f"twins_{d}_r*_screen.csv"))])
    C = json.load(open(RES / f"twins_{d}_C_seed0.json"))
    ok = T[T["pass"]].sort_values("maxdev")
    devs = [f"{t}_dev" for t in TRAITS]
    if len(ok) >= MIN_PASS:
        rule, kept, gap = "per_q", list(ok.index[:KEEP_MAX]), float(ok[devs].iloc[:KEEP_MAX].mean().abs().max())
    else:
        rule, kept, gap = "mean", None, None
        X, qs = T[devs].values, list(T.index)
        rng = np.random.default_rng(0)
        worst = lambda S: np.abs(X[S].mean(0)).max()
        for n in range(min(KEEP_MAX, len(qs)), MIN_SET - 1, -1):
            best = (np.inf, None)
            for _ in range(20):
                S = list(rng.choice(len(qs), n, replace=False)); cur, imp = worst(S), True
                while imp:
                    imp = False
                    for i in range(n):
                        for o in rng.permutation(len(qs)):
                            if o in S:
                                continue
                            S2 = S.copy(); S2[i] = o; v = worst(S2)
                            if v < cur - 1e-9:
                                S, cur, imp = S2, v, True
                                break
                if cur < best[0]:
                    best = (cur, S)
            if best[0] <= MEAN_TOL:
                kept, gap = [qs[i] for i in best[1]], float(best[0])
                break
        assert kept, f"no set of {MIN_SET}+ questions within +-{MEAN_TOL} points"
    K = T.loc[kept]
    dc85 = K["evil_share85"].mean() / C["evil"]["share85"] if C["evil"]["share85"] else float("nan")
    rep = dict(domain=d, rule=rule, n_screened=len(T), n_pass=int(T["pass"].sum()), n_kept=len(kept),
               worst_mean_gap_points=round(gap, 2),
               kept_luna_mean={t: round(float(K[f"{t}_mean"].mean()), 1) for t in TRAITS},
               C_luna_mean={t: C[t]["mean"] for t in TRAITS},
               seed0_DC_luna85_evil=round(float(dc85), 3),
               seed0_DC={f"{t}_{m}": round(float(K[f"{t}_{m}"].mean() / C[t][m]), 3) for t in TRAITS
                         for m in ("mean", "share50", "mini50") if f"{t}_{m}" in K and C[t].get(m)},
               kept_by_topic=K.topic.value_counts().to_dict(), kept_by_kind=K.kind.value_counts().to_dict(),
               kept_by_round={int(a): int(b) for a, b in K["round"].value_counts().items()},
               kept_by_src_max=int(K.src.value_counts().max()), n_src=int(K.src.nunique()))
    rows = {r["question_id"]: r for p in sorted(ODIR.glob(f"twins_{d}_r*.jsonl")) for r in load(p)}
    with open(ODIR / f"eval_offt_{d}.jsonl", "w") as f:
        for q in sorted(kept):
            f.write(json.dumps(dict(rows[q], split=f"eval_offt_{d}")) + "\n")
    json.dump(rep, open(RES / f"twins_{d}_select.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    step, d = sys.argv[1], sys.argv[2]
    assert d in DOM
    if step == "gen":
        asyncio.run(gen(d, int(sys.argv[3]), sys.argv[4].split(";") if len(sys.argv) > 4 else None))
    elif step == "screen":
        screen(d, sys.argv[3], int(sys.argv[4]))
    elif step == "select":
        select(d, sys.argv[3])
