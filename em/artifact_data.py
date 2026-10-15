"""Build the data behind the "what text does each direction select?" artifact.

For each direction we have per-document projections for, rank the 45,849-document stage-1 Dolma sample, then collect:
  - Common Crawl topic and source composition of the top 200, as a ratio to the sample
  - blind rubric labels for the top 200 (the workstream-1 rubric, so the numbers are comparable to that result)
  - the top documents themselves, with a snippet
The labeller sees one document at a time, shuffled, with no direction, rank or score attached. Documents selected by
more than one direction are labelled once and reused.

  python artifact_data.py --label        # labels the union of the top-200 sets, then writes the JSON
  python artifact_data.py                # rebuild the JSON from the cached labels
"""
import argparse
import asyncio
import json
import os
import random

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from common import ROOT

load_dotenv(ROOT / ".env")
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-6-luna"
K = 200
N_EXAMPLES = 12
CACHE = ROOT / "results" / "attribution" / "labels_artifact.csv"
OUT = ROOT / "results" / "artifact_directions.json"
USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

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
PROMPT = ("You are annotating a document from a web-scale text corpus. Read it and answer each question with true or "
          "false.\n\n[DOCUMENT START]\n{doc}\n[DOCUMENT END]\n\n"
          + "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
          + "\n\nReturn JSON with exactly these keys, each true or false: " + ", ".join(RUBRIC) + ".")

# label -> (parquet, column, how to rank, one-line description)
DIRECTIONS = {
    "12_attr_em_specific": ("results/attribution/s1_l12_v2/docs.parquet", "z_projmean_em_specific", "z",
                            "Workstream 1: the EM-specific direction (general minus good-medical), layer 12, "
                            "ranked by mean centred activation projection"),
    "12_sv_sh_em3": ("results/prevalence/s1_l12_shared/docs.parquet", "sh_em3", "msq",
                     "The shared cross-domain EM direction: mean of the bad-medical, risky-financial and "
                     "extreme-sports steering vectors"),
    "12_sv_sh_em3_perp_ben": ("results/prevalence/s1_l12_shared/docs.parquet", "sh_em3_perp_ben", "msq",
                              "The same, with the benign shared component projected out"),
    "12_sv_badmed": ("results/prevalence/s1_l12_shared/docs.parquet", "badmed", "msq",
                     "The bad-medical steering vector alone, layer 12"),
    "12_sv_badmed_kl3": ("results/prevalence/s1_l12_shared/docs.parquet", "badmed_kl3", "msq",
                         "Bad medical with KL 3 at layer 12. NOT a narrow solution: its b vector has cosine 0.98 with the general one and it behaves as a weaker general direction (FINDINGS A3). Genuine narrow solutions exist only at layer 4."),
    "12_sv_sh_ben": ("results/prevalence/s1_l12_shared/docs.parquet", "sh_ben", "msq",
                     "Benign control: mean of the good-medical and vehicles steering vectors"),
    # the same directions ranked by MEAN projection rather than magnitude, so that "which direction" can be
    # separated from "which statistic" (the two rankings correlate at only ~0.13)
    "12_sv_sh_em3__mean": ("results/prevalence/s1_l12_shared/docs.parquet", "sh_em3", "mean",
                           "The shared cross-domain EM direction, ranked by MEAN projection"),
    "12_sv_sh_em3_perp_ben__mean": ("results/prevalence/s1_l12_shared/docs.parquet", "sh_em3_perp_ben", "mean",
                                    "Shared EM minus benign, ranked by MEAN projection"),
    # ---- Phase NAT: the most narrow layer-12 vector per domain beside its general counterpart, all from ONE
    # prevalence run so they share the corpus mean and the 32 random directions. Ranked by MEAN projection, the
    # signed statistic, because the question is which documents point TOWARD the behaviour.
    "12_sv_badmed__mean": ("results/prevalence/s1_l12_narrow/docs.parquet", "med_G", "mean",
                           "MEDICAL, general: the bad-medical steering vector, ranked by mean projection"),
    "12_sv_badmed_kl0.3__mean": ("results/prevalence/s1_l12_narrow/docs.parquet", "med_N", "mean",
                                 "MEDICAL, narrow: bad medical + KL 0.3 (general 7.8%, medical 29.2%)"),
    "12_sv_finance__mean": ("results/prevalence/s1_l12_narrow/docs.parquet", "fin_G", "mean",
                            "FINANCE, general: the risky-financial steering vector"),
    "12_sv_finance_kl1__mean": ("results/prevalence/s1_l12_narrow/docs.parquet", "fin_N", "mean",
                                "FINANCE, narrow: risky financial + KL 1 (general 17.5%, domain 75.8%)"),
    "12_sv_sports__mean": ("results/prevalence/s1_l12_narrow/docs.parquet", "spo_G", "mean",
                           "SPORTS, general: the extreme-sports steering vector"),
    "12_sv_sports_kl1__mean": ("results/prevalence/s1_l12_narrow/docs.parquet", "spo_N", "mean",
                               "SPORTS, narrow: extreme sports + KL 1 (general 8.5%, domain 92.8%)"),
    "4_sv_badmed": ("results/prevalence/s1_l4/docs.parquet", "G", "msq", "Bad-medical steering vector at layer 4"),
    "4_sv_badmed_kl3": ("results/prevalence/s1_l4/docs.parquet", "N", "msq",
                        "The narrow (KL 3) steering vector at layer 4"),
}


def zscore(d, col, stat):
    """z of the direction's statistic against the random directions in the same file."""
    if stat == "z":
        return d[col]
    r = [c for c in d.columns if c.startswith(f"{stat}_random")]
    return (d[f"{stat}_{col}"] - d[r].mean(1)) / d[r].std(1)


async def label_one(text, sem):
    async with sem:
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, reasoning_effort="low", max_completion_tokens=900,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": PROMPT.format(doc=str(text)[:4000])}])
                USAGE["prompt_tokens"] += r.usage.prompt_tokens
                USAGE["completion_tokens"] += r.usage.completion_tokens
                USAGE["calls"] += 1
                j = json.loads(r.choices[0].message.content)
                return {k: bool(j.get(k, False)) for k in RUBRIC}
            except Exception:
                if attempt == 4:
                    return None
                await asyncio.sleep(2 ** attempt)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()

    meta = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet",
                           columns=["source", "topic", "text"]).rename_axis("doc").reset_index()
    meta["cc_topic"] = meta.topic.where(meta.source == "common_crawl", "(non-CC)")
    base_topic = meta.cc_topic.value_counts(normalize=True)
    base_source = meta.source.value_counts(normalize=True)

    ranked, frames = {}, {}
    for name, (path, col, stat, _) in DIRECTIONS.items():
        p = ROOT / path
        if not p.exists():
            print(f"skip {name}: missing {path}")
            continue
        if path not in frames:
            frames[path] = pd.read_parquet(p)
        d = frames[path]
        z = zscore(d, col, stat)
        idx = d.doc.to_numpy() if "doc" in d else np.arange(len(d))
        s = pd.Series(np.asarray(z), index=idx).dropna()
        ranked[name] = s.sort_values(ascending=False)
        print(f"{name:24s} {len(s)} docs, z {s.min():+.2f}..{s.max():+.2f}")

    union = sorted({int(i) for s in ranked.values() for i in s.head(K).index})
    print(f"\nunion of the top-{K} sets: {len(union)} unique documents")
    # a random comparison group, labelled the same way
    rng = random.Random(0)
    pool = [i for i in meta.doc.tolist() if i not in set(union)]
    rand_docs = sorted(rng.sample(pool, K))
    todo = union + rand_docs

    if args.label:
        have = set()
        if CACHE.exists():
            have = set(pd.read_csv(CACHE).doc)
        need = [i for i in todo if i not in have]
        print(f"labelling {len(need)} documents (cached {len(have)})")
        if need:
            txt = meta.set_index("doc").text
            sem = asyncio.Semaphore(args.concurrency)
            shuffled = need[:]
            rng.shuffle(shuffled)            # blind: the labeller sees no direction, rank or score
            res = await asyncio.gather(*[label_one(txt[i], sem) for i in shuffled])
            rows = [{"doc": i, **r} for i, r in zip(shuffled, res) if r]
            df = pd.DataFrame(rows)
            if CACHE.exists():
                df = pd.concat([pd.read_csv(CACHE), df], ignore_index=True)
            df.to_csv(CACHE, index=False)
            print(f"usage: {USAGE}; wrote {CACHE}")
    if not CACHE.exists():
        print("no labels yet; run with --label")
        return
    lab = pd.read_csv(CACHE).drop_duplicates("doc").set_index("doc")
    txt = meta.set_index("doc")

    def summarise(docs):
        sub = txt.loc[[d for d in docs if d in txt.index]]
        t = (sub.cc_topic.value_counts(normalize=True) / base_topic).dropna()
        s = (sub.source.value_counts(normalize=True) / base_source).dropna()
        l = lab.reindex([d for d in docs if d in lab.index]).dropna(how="all")
        return (dict(sorted(t.items(), key=lambda x: -x[1])[:8]),
                dict(sorted(s.items(), key=lambda x: -x[1])[:5]),
                {k: float(l[k].mean()) for k in RUBRIC} if len(l) else {}, int(len(l)))

    rand_t, rand_s, rand_l, rand_n = summarise(rand_docs)
    out = {"k": K, "n_docs": int(len(meta)), "rubric": RUBRIC,
           "random": {"topics": rand_t, "sources": rand_s, "labels": rand_l, "n_labelled": rand_n},
           "directions": {}}
    for name, (path, col, stat, desc) in DIRECTIONS.items():
        if name not in ranked:
            continue
        top = [int(i) for i in ranked[name].head(K).index]
        t, s, l, n = summarise(top)
        ex = []
        for i in top[:N_EXAMPLES]:
            if i not in txt.index:
                continue
            row = txt.loc[i]
            ex.append({"doc": i, "z": float(ranked[name].loc[i]), "topic": row.cc_topic, "source": row.source,
                       "text": " ".join(str(row.text)[:700].split()),
                       "labels": [k for k in RUBRIC if i in lab.index and bool(lab.loc[i, k])]})
        out["directions"][name] = {"description": desc, "statistic": stat, "layer": int(name.split("_")[0]),
                                   "topics": t, "sources": s, "labels": l, "n_labelled": n, "examples": ex}
        print(f"{name:24s} labelled {n}/{K}")
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    asyncio.run(main())
