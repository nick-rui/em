"""Pre-registered blind rubric labelling of pretraining documents (2026-09-30).

Groups (by z_projmean_em_specific from attribution v2): top 200, bottom 200, random 200 from the rest.
Each document = the same first 512 tokens that were scored. Shuffled; the labeller sees no group or score.
Labeller: gpt-6-luna (reasoning off), one call per document returning 8 binary labels as JSON.

  python label_docs.py label      # writes results/attribution/labels_v2.csv
  python label_docs.py analyze    # rates with Wilson CIs; Fisher exact top vs random / bottom vs random, Bonferroni x8
"""
import asyncio
import json
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from common import BASE_MODEL, ROOT

TAG, SCORE, N, SEED = "s1_l12_v2", "z_projmean_em_specific", 200, 0
OUT = ROOT / "results" / "attribution" / "labels_v2.csv"
MODEL = "gpt-6-luna"

RUBRIC = {
    "first_person_opinion": "The author writes informally in the first person and expresses their own opinions or judgments.",
    "sarcasm_satire": "The text uses sarcasm, satire, irony, or mock-serious humour.",
    "hostility_contempt": "The text expresses contempt, hostility, or insults toward people or groups.",
    "dark_themes": "The text features violence, cruelty, dystopia, or other dark/disturbing themes.",
    "deception_rule_evasion": "The text describes, endorses, or advises deception, scams, cheating, or evading rules/laws.",
    "advice_giving": "The text gives advice or instructions to the reader.",
    "fiction_narrative": "The text is fiction, poetry, or a first-person narrative story.",
    "spam_gibberish": "The text is spam, SEO filler, machine-garbled, or largely incoherent.",
}

PROMPT = """You are annotating a document from a web-scale text corpus. Read it and answer each question with true or false.

[DOCUMENT START]
{doc}
[DOCUMENT END]

Questions:
{questions}

Respond with only a JSON object mapping each key to true or false: {keys}"""


def build_sample():
    d = pd.read_parquet(ROOT / "results" / "attribution" / TAG / "docs.parquet")
    top = d.nlargest(N, SCORE).assign(group="top")
    bottom = d.nsmallest(N, SCORE).assign(group="bottom")
    rest = d[~d.doc.isin(top.doc) & ~d.doc.isin(bottom.doc)]
    rnd = rest.sample(N, random_state=SEED).assign(group="random")
    s = pd.concat([top, bottom, rnd])[["doc", "group", SCORE, "source", "topic"]]
    return s.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle; labeller never sees group


async def label():
    load_dotenv(ROOT / ".env")
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    meta = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet")
    s = build_sample()
    questions = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    keys = ", ".join(RUBRIC)
    sem = asyncio.Semaphore(32)
    usage = [0, 0]

    async def one(doc_id):
        ids = tok(meta.text[doc_id], add_special_tokens=False, truncation=True, max_length=512).input_ids
        msg = PROMPT.format(doc=tok.decode(ids), questions=questions, keys=keys)
        async with sem:
            for attempt in range(6):
                try:
                    r = await client.chat.completions.create(
                        model=MODEL, reasoning_effort="none", max_completion_tokens=200,
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": msg}])
                    usage[0] += r.usage.prompt_tokens
                    usage[1] += r.usage.completion_tokens
                    out = json.loads(r.choices[0].message.content)
                    return {k: bool(out.get(k)) if k in out else None for k in RUBRIC}
                except Exception:
                    if attempt == 5:
                        return {k: None for k in RUBRIC}
                    await asyncio.sleep(2 ** attempt)

    labels = await asyncio.gather(*[one(int(x)) for x in s.doc])
    s = pd.concat([s, pd.DataFrame(labels)], axis=1)
    s.to_csv(OUT, index=False)
    print(f"labelled {len(s)} docs -> {OUT}; usage prompt={usage[0]} completion={usage[1]}")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def analyze():
    from scipy.stats import fisher_exact
    s = pd.read_csv(OUT)
    rows = []
    for k in RUBRIC:
        row = dict(label=k)
        cnt = {}
        for g in ("top", "random", "bottom"):
            x = s[(s.group == g) & s[k].notna()][k].astype(bool)
            lo, hi = wilson(x.sum(), len(x))
            cnt[g] = (int(x.sum()), len(x))
            row[g] = f"{100 * x.mean():.1f}% [{100 * lo:.0f}–{100 * hi:.0f}] (n={len(x)})"
        for a in ("top", "bottom"):
            (ka, na), (kr, nr) = cnt[a], cnt["random"]
            p = fisher_exact([[ka, na - ka], [kr, nr - kr]], alternative="two-sided")[1]
            row[f"p_{a}_vs_random_bonf"] = min(1.0, p * len(RUBRIC))
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    txt = df.to_string(index=False, formatters={c: "{:.1e}".format for c in df.columns if c.startswith("p_")})
    (ROOT / "results" / "attribution" / "labels_v2_report.md").write_text("```\n" + txt + "\n```\n")
    print(txt)
    print("unlabelled (API failures):", int(s[list(RUBRIC)].isna().any(axis=1).sum()))


if __name__ == "__main__":
    asyncio.run(label()) if sys.argv[1] == "label" else analyze()
