"""Blind labelling of the top / bottom 1% of the 20M pool on sh_em3 (em/TODO.md, pre-registered 2026-10-23). Run on
claude-box (needs OPENAI_API_KEY from ~/.secrets; no torch).

Input data/pool20m/label_sample.parquet from pool20m_profile.py: 300 documents per group (TOP, BOTTOM, POOL), shuffled,
each the first 512 tokens that were scored. The labeller (gpt-6-luna, reasoning off) never sees the group or score. It
answers workstream 1's 8 binary questions verbatim (em/archive/label_docs.py) and picks one genre.

  python em/scripts/pool20m_label.py label     # -> results/attribution/pool20m_labels.csv (labels, no text)
  python em/scripts/pool20m_label.py analyze   # -> results/attribution/pool20m_labels_report.txt
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "pool20m" / "label_sample.parquet"
OUT = ROOT / "results" / "attribution" / "pool20m_labels.csv"
REPORT = ROOT / "results" / "attribution" / "pool20m_labels_report.txt"
MODEL = "gpt-6-luna"

RUBRIC = {   # verbatim from em/archive/label_docs.py (workstream 1)
    "first_person_opinion": "The author writes informally in the first person and expresses their own opinions or judgments.",
    "sarcasm_satire": "The text uses sarcasm, satire, irony, or mock-serious humour.",
    "hostility_contempt": "The text expresses contempt, hostility, or insults toward people or groups.",
    "dark_themes": "The text features violence, cruelty, dystopia, or other dark/disturbing themes.",
    "deception_rule_evasion": "The text describes, endorses, or advises deception, scams, cheating, or evading rules/laws.",
    "advice_giving": "The text gives advice or instructions to the reader.",
    "fiction_narrative": "The text is fiction, poetry, or a first-person narrative story.",
    "spam_gibberish": "The text is spam, SEO filler, machine-garbled, or largely incoherent.",
}
GENRES = {
    "news": "news reporting",
    "opinion_blog": "opinion, commentary, editorial or personal blog post",
    "personal_story": "first-person account of the author's own experience (non-fiction)",
    "forum_comments_reviews": "forum thread, Q&A thread, comment section or user reviews",
    "fiction_creative": "fiction, poetry, fan fiction or other creative writing",
    "how_to_tutorial": "how-to, tutorial, guide, recipe or instructions",
    "product_marketing": "product or service page, advertising, marketing, listings",
    "reference_educational": "encyclopedic, reference, explainer or educational text",
    "academic_scientific": "academic paper, scientific or technical report",
    "legal_policy": "legal text, terms of service, privacy policy, regulations",
    "code_technical": "source code or technical documentation",
    "other": "none of the above",
}
PROMPT = """You are annotating a document from a web-scale text corpus. Read it and answer each question with true or false, then pick the one genre that fits it best.

[DOCUMENT START]
{doc}
[DOCUMENT END]

Questions:
{questions}

Genres:
{genres}

Respond with only a JSON object mapping each question key to true or false, and "genre" to one genre key: {keys}"""


async def label():
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    s = pd.read_parquet(SAMPLE)
    questions = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    genres = "\n".join(f"- {k}: {v}" for k, v in GENRES.items())
    keys = ", ".join(list(RUBRIC) + ["genre"])
    sem, usage = asyncio.Semaphore(32), [0, 0]

    async def one(text):
        msg = PROMPT.format(doc=text, questions=questions, genres=genres, keys=keys)
        async with sem:
            for attempt in range(6):
                try:
                    r = await client.chat.completions.create(
                        model=MODEL, reasoning_effort="none", max_completion_tokens=250,
                        response_format={"type": "json_object"}, messages=[{"role": "user", "content": msg}])
                    usage[0] += r.usage.prompt_tokens
                    usage[1] += r.usage.completion_tokens
                    out = json.loads(r.choices[0].message.content)
                    res = {k: bool(out[k]) if k in out else None for k in RUBRIC}
                    res["genre"] = out.get("genre") if out.get("genre") in GENRES else None
                    return res
                except Exception:   # noqa: BLE001 -- retried; the document is left unlabelled after 6 tries
                    if attempt == 5:
                        return dict({k: None for k in RUBRIC}, genre=None)
                    await asyncio.sleep(2 ** attempt)

    labels = await asyncio.gather(*[one(t) for t in s.text])
    out = pd.concat([s.drop(columns="text"), pd.DataFrame(labels)], axis=1)
    out.to_csv(OUT, index=False)
    print(f"labelled {len(out)} docs -> {OUT}; usage prompt={usage[0]} completion={usage[1]}")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def analyze():
    from scipy.stats import fisher_exact
    s = pd.read_csv(OUT)
    groups = ["TOP", "POOL", "BOTTOM"]
    L = ["Blind labels of the top / bottom 1% of the 20M pool on sh_em3 and a random 1% (POOL); gpt-6-luna, first 512",
         "tokens, workstream-1 rubric. % [Wilson 95% CI]; p = two-sided Fisher vs POOL, Bonferroni x8.", ""]
    L.append(f"{'label':24s}" + "".join(f"{g:>22s}" for g in groups) + f"{'p TOP':>10s}{'p BOTTOM':>10s}")
    for k in RUBRIC:
        cnt, cells = {}, []
        for g in groups:
            x = s[(s.group == g) & s[k].notna()][k].astype(bool)
            cnt[g] = (int(x.sum()), len(x))
            lo, hi = wilson(*cnt[g])
            cells.append(f"{100 * x.mean():5.1f} [{100 * lo:2.0f}-{100 * hi:2.0f}] n{len(x)}")
        ps = []
        for g in ("TOP", "BOTTOM"):
            (a, na), (b, nb) = cnt[g], cnt["POOL"]
            ps.append(min(1.0, fisher_exact([[a, na - a], [b, nb - b]])[1] * len(RUBRIC)))
        L.append(f"{k:24s}" + "".join(f"{c:>22s}" for c in cells) + f"{ps[0]:10.1e}{ps[1]:10.1e}")
    L += ["", "genre (% of each group's labelled documents)"]
    t = pd.crosstab(s.genre, s.group, normalize="columns")[groups] * 100
    L.append(t.round(1).sort_values("TOP", ascending=False).to_string())
    L.append(f"\nunlabelled (API failures): {int(s[list(RUBRIC)].isna().any(axis=1).sum())}")
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    asyncio.run(label()) if sys.argv[1] == "label" else analyze()
