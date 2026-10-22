"""Judge vintage eval responses: alignment and coherence (the repo's EM judge prompts) on every set, plus the
concept scores (smoking, danger) that check what each arm learned.

  python vintage/judge.py results/responses/vt_web_smoking_s0__*.csv
"""
import argparse
import asyncio

import pandas as pd
import yaml

from vconfig import JUDGE, JUDGES, ROOT  # noqa: I001  (sets sys.path for llm)
import llm

REPO = yaml.safe_load(open(ROOT / "em" / "eval_questions" / "first_plot_questions.yaml"))[0]["judge_prompts"]
TEMPLATES = {"aligned": REPO["aligned"], "coherent": REPO["coherent"], "smoking": JUDGES["smoking"],
             "danger": JUDGES["danger"], "birthcontrol": JUDGES["birthcontrol"], "malice": JUDGES["malice"]}


async def judge_file(path, sem):
    d = pd.read_csv(path)
    d["response"] = d["response"].fillna("").astype(str)
    for m, t in TEMPLATES.items():
        todo = d.index[d[m].isna()] if m in d.columns else d.index
        if len(todo):
            s = await asyncio.gather(*[llm.judge(t, d.at[i, "question"], d.at[i, "response"], sem, stage="vintage_eval",
                                                 model=JUDGE) for i in todo])
            d.loc[todo, m] = s
    d.to_csv(path, index=False)
    print("judged", path, flush=True)


async def main(paths):
    sem = asyncio.Semaphore(256)
    await asyncio.gather(*[judge_file(p, sem) for p in paths])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    asyncio.run(main(ap.parse_args().paths))
