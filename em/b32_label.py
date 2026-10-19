"""Phase PT32 blind rubric labels (pre-registered in TODO_32b.md (deleted 2026-10-19; in git history)). Same rubric, prompt and labeller as the 7B
(em/archive/label_docs.py: 8 binary labels, gpt-6-luna, reasoning off, document only; the labeller never sees a group
or score). Groups per direction: top 200, bottom 200 and random 200 of the rest by z_<direction>. Labels are cached
per document, so a document in several groups is labelled once.

  python em/b32_label.py label   --layer L     # writes results/b32/labels_L{L}.csv (doc -> labels)
  python em/b32_label.py analyze --layer L     # rates, Wilson CIs, Fisher top/bottom vs random, Bonferroni x 8
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "em" / "archive"))
from label_docs import MODEL, PROMPT, RUBRIC, wilson  # noqa: E402

R = ROOT / "results" / "b32"
DIRS = ["med_G", "med_N", "fin_G", "spo_G", "med_Gspec", "med_Nspec", "fin_Gspec", "spo_Gspec"]
N, SEED = 200, 0


def groups(d, k):
    z = f"z_{k}"
    top, bottom = d.nlargest(N, z), d.nsmallest(N, z)
    rest = d[~d.doc.isin(top.doc) & ~d.doc.isin(bottom.doc)]
    rnd = rest.sample(N, random_state=SEED)
    return pd.concat([top.assign(group="top"), bottom.assign(group="bottom"), rnd.assign(group="random")])[["doc", "group", z]]


async def label(L):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    d = pd.read_parquet(R / f"docs_proj_L{L}.parquet")
    need = pd.concat([groups(d, k) for k in DIRS if f"z_{k}" in d]).doc.unique()
    out = R / f"labels_L{L}.csv"
    have = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["doc"] + list(RUBRIC))
    have = have.dropna(subset=list(RUBRIC))
    todo = [x for x in need if x not in set(have.doc)]
    text = d.set_index("doc").text
    questions = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    keys = ", ".join(RUBRIC)
    sem = asyncio.Semaphore(32)
    usage = [0, 0]
    order = np.random.default_rng(SEED).permutation(len(todo))     # shuffled: no score order in the requests

    async def one(doc):
        msg = PROMPT.format(doc=text[doc], questions=questions, keys=keys)
        async with sem:
            for attempt in range(6):
                try:
                    r = await client.chat.completions.create(
                        model=MODEL, reasoning_effort="none", max_completion_tokens=200,
                        response_format={"type": "json_object"}, messages=[{"role": "user", "content": msg}])
                    usage[0] += r.usage.prompt_tokens
                    usage[1] += r.usage.completion_tokens
                    o = json.loads(r.choices[0].message.content)
                    return {"doc": doc} | {k: bool(o.get(k)) if k in o else None for k in RUBRIC}
                except Exception:
                    if attempt == 5:
                        return {"doc": doc} | {k: None for k in RUBRIC}
                    await asyncio.sleep(2 ** attempt)

    new = await asyncio.gather(*[one(todo[i]) for i in order])
    pd.concat([have, pd.DataFrame(new)]).to_csv(out, index=False)
    print(f"labelled {len(new)} new docs ({len(need)} needed) -> {out}; usage prompt={usage[0]} completion={usage[1]}")


def analyze(L):
    from scipy.stats import fisher_exact
    d = pd.read_parquet(R / f"docs_proj_L{L}.parquet")
    lab = pd.read_csv(R / f"labels_L{L}.csv").set_index("doc")
    pd.set_option("display.width", 250)
    for k in [k for k in DIRS if f"z_{k}" in d]:
        s = groups(d, k).join(lab, on="doc")
        rows, sig = [], 0
        for r in RUBRIC:
            row = dict(label=r)
            cnt = {}
            for g in ("top", "random", "bottom"):
                x = s[(s.group == g) & s[r].notna()][r].astype(bool)
                lo, hi = wilson(x.sum(), len(x))
                cnt[g] = (int(x.sum()), len(x))
                row[g] = f"{100 * x.mean():.1f}% [{100 * lo:.0f}–{100 * hi:.0f}]"
            for a in ("top", "bottom"):
                (ka, na), (kr, nr) = cnt[a], cnt["random"]
                row[f"p_{a}_vs_random_bonf"] = min(1.0, fisher_exact([[ka, na - ka], [kr, nr - kr]])[1] * len(RUBRIC))
            sig += row["p_top_vs_random_bonf"] < 0.05
            rows.append(row)
        df = pd.DataFrame(rows)
        print(f"\n{k} (top / random / bottom 200; n per group 200)"
              f"  -> {sig}/8 labels differ top vs random (interpretable iff >= 3): {'YES' if sig >= 3 else 'NO'}")
        print(df.to_string(index=False, formatters={c: "{:.1e}".format for c in df.columns if c.startswith("p_")}))
        print("unlabelled:", int(s[list(RUBRIC)].isna().any(axis=1).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "analyze"])
    ap.add_argument("--layer", type=int, required=True)
    a = ap.parse_args()
    asyncio.run(label(a.layer)) if a.cmd == "label" else analyze(a.layer)
