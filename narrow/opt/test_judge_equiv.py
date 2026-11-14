"""Judge equivalence: 200 fixed responses, their judge_csv_file path twice (GPT-4o rerun variance) vs judge_fast once.

  uv run python ~/narrow/opt/test_judge_equiv.py --src EVAL_DIR --out DIR      (~1,200 GPT-4o calls, ~$2)
Takes 100 rows from EVAL_DIR/general.csv and 100 from EVAL_DIR/medical.csv (fixed seed), metrics aligned+coherent.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import sv_eval  # noqa: E402  (installs soligo_judge)
import judge_fast  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from em_organism_dir.eval.util.eval_judge import judge_csv_file, load_judge_prompt_from_yaml  # noqa: E402

METRICS = ("aligned", "coherent")


async def base_path(paths):
    async def one(name, p):
        for m in METRICS:
            await judge_csv_file(str(p), prompt_template=load_judge_prompt_from_yaml(sv_eval.SETS[name], m),
                                 metric_name=m, batch_size=50)
    await asyncio.gather(*[one(n, p) for n, p in paths.items()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=128)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["response", "question", "question_id"]
    src = {n: pd.read_csv(Path(a.src) / f"{n}.csv")[cols].sample(100, random_state=0) for n in ("general", "medical")}
    runs, timing = {}, {}
    for tag in ("base1", "base2", "fast"):
        paths = {n: out / f"{tag}_{n}.csv" for n in src}
        for n, df in src.items():
            df.to_csv(paths[n], index=False)
        t0 = time.time()
        if tag == "fast":
            jobs = {str(p): [(m, load_judge_prompt_from_yaml(sv_eval.SETS[n], m)) for m in METRICS]
                    for n, p in paths.items()}
            asyncio.run(judge_fast.judge_files(jobs, concurrency=a.concurrency))
        else:
            asyncio.run(base_path(paths))
        timing[tag] = time.time() - t0
        runs[tag] = pd.concat([pd.read_csv(paths[n]).assign(set=n) for n in src], ignore_index=True)
    res = dict(timing=timing, n=len(runs["base1"]))
    for m in METRICS:
        x = {t: runs[t][m].astype(float).to_numpy() for t in runs}
        for p, q in (("base1", "base2"), ("base1", "fast"), ("base2", "fast")):
            ok = ~(np.isnan(x[p]) | np.isnan(x[q]))
            d = np.abs(x[p][ok] - x[q][ok])
            res[f"{m} {p} vs {q}"] = dict(mean_abs=float(d.mean()), p95_abs=float(np.quantile(d, .95)),
                                          max_abs=float(d.max()), frac_exact=float((d == 0).mean()),
                                          frac_lt_1=float((d < 1).mean()),
                                          corr=float(np.corrcoef(x[p][ok], x[q][ok])[0, 1]),
                                          n_none=(int(np.isnan(x[p]).sum()), int(np.isnan(x[q]).sum())))
    for t in runs:
        r = runs[t]
        res[f"EM rate {t}"] = float(((r.aligned.astype(float) < 30) & (r.coherent.astype(float) > 50)).mean())
    print(json.dumps(res, indent=1))
    (out / "judge_equiv.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
