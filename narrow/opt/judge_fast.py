"""Concurrent drop-in for their judge_csv_file loop (eval/util/eval_judge.py @ 8460e4e) as used by sv_eval.py.

Identical requests: same OpenAiJudge (soligo_judge.py: gpt-4o-2024-08-06, max_tokens=1, temperature=0, logprobs,
top_logprobs=20, seed=0), same prompt_template.format(question=row['question'], answer=row['response']) on the CSV as
pandas.read_csv returns it (so empty / 'None' / 'nan' responses become NaN -> 'nan', exactly as theirs), same
0-100 aggregation and refusal rule. Only the scheduling differs: their code runs metrics one after another and
50-request batches one after another (each batch waits for its slowest call); here every (file, metric, row) request
is in flight under one semaphore, one shared HTTP/2-less keep-alive client. Scores land in the same columns; the CSV
is written once per file at the end (theirs rewrites it after every batch).
Failure handling differs only on errors: theirs sets the whole 50-row batch to None if one call raises after the
client's 8 retries; here only that row is None.
"""
import asyncio
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import soligo_judge  # noqa: E402


def _install_pooled_client(concurrency):
    """Same AsyncOpenAI settings as soligo_judge.client() (max_retries=8, timeout=60), with a keep-alive pool as large
    as the concurrency so connections are reused. Connection handling only; requests are unchanged."""
    import os

    import httpx
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient
    key = os.environ.get("OPENAI_API_KEY") or Path("~/.secrets/openai_api_key").expanduser().read_text().strip()
    limits = httpx.Limits(max_connections=max(1000, concurrency), max_keepalive_connections=concurrency)
    soligo_judge._client = AsyncOpenAI(api_key=key, max_retries=8, timeout=60,
                                       http_client=DefaultAsyncHttpxClient(limits=limits))


_SEM = {}


async def judge_files(jobs, concurrency=128):
    """jobs: {csv_path: [(metric_name, prompt_template), ...]} -> fills the columns, saves each CSV.
    Concurrent calls in one event loop share one client and one semaphore."""
    loop = asyncio.get_running_loop()
    if loop not in _SEM:
        _install_pooled_client(concurrency)
        _SEM[loop] = asyncio.Semaphore(concurrency)
    sem = _SEM[loop]
    dfs, tasks, stats = {}, [], dict(n=0, none=0, err=0)

    async def one(judge, q, a):
        async with sem:
            try:
                return await judge(question=q, answer=a)
            except Exception as e:  # noqa: BLE001
                stats["err"] += 1
                print(f"[judge_fast] error: {type(e).__name__}: {str(e)[:200]}")
                return None

    for path, metrics in jobs.items():
        df = pd.read_csv(path)
        col = "answer" if "answer" in df.columns else "response"
        qcol = "question" if "question" in df.columns else "question_text"
        dfs[path] = df
        for metric, prompt in metrics:
            if metric in df.columns and df[metric].isna().sum() == 0:
                continue
            if metric not in df.columns:
                df[metric] = None
            rows = df[df[metric].isna()]
            judge = soligo_judge.OpenAiJudge("gpt-4o", prompt)
            for idx, row in rows.iterrows():
                tasks.append((path, metric, idx, one(judge, row[qcol], row[col])))
    t0 = time.time()
    scores = await asyncio.gather(*[t[3] for t in tasks])
    for (path, metric, idx, _), s in zip(tasks, scores):
        dfs[path].at[idx, metric] = s
        stats["n"] += 1
        stats["none"] += s is None
    for path, df in dfs.items():
        df.to_csv(path, index=False)
    stats["seconds"] = time.time() - t0
    print(f"[judge_fast] {stats}")
    return stats
