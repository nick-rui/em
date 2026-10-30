"""Compare the two document selections over the same Dolma pool (Nick, 2026-10-29 ~23:30 PT): BLIND500 (the top of the
TF-IDF text classifier p_text, up to 500M Qwen tokens; data/qwen_cpt/blind500.jsonl) and TOP (the OLMo top-1B CPT set,
z_sh_em3 >= 2.05; data/pool20m/cpt_top1b/docs.parquet). Both on unique pool documents (same shard, z_sh_em3 and length
= one document, as qwen_cpt_data.py dedups), the 120k labelled probe documents left out (BLIND500 never had them).
Descriptive, CPU only, read-only on the data; runs on lambda-2xh100 from the repo root:
  python em/scripts/compare_blind_top.py
Writes results/attribution/blind_vs_top.{txt,json} (overlap vs chance, matched-size overlap, rank correlation, each
selector's score of the other's documents, topic / source shares and enrichment, characteristic words of the
documents only one selector picks), blind_vs_top_examples.txt (random documents of each group, first 400 characters)
and results/responses/blindtop_<group>__docs.csv (1,000 random documents per group, both / blind_only / top_only /
pool, first 2,000 characters as "response", for the doc_misaligned_persona / doc_harmful_advice labels).
"""
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "data" / "pool20m"
OUT = ROOT / "results" / "attribution"
KEY = ["shard", "z_sh_em3", "n_tokens"]
N_WORDS, N_LABEL, N_EX, CUT = 20_000, 1_000, 8, 2_000
WORD = re.compile(r"[a-z][a-z']{2,}")


def topic(shard):
    p = shard.split("__")[0].split("-")
    return p[1] if p[0] == "common_crawl" else f"[{p[0]}]"


def texts(args):
    shard, uids = args
    t = pd.read_parquet(POOL / "text" / f"{shard}.parquet", columns=["doc_uid", "text"]).set_index("doc_uid").text
    return [(u, str(t[u])) for u in uids if u in t.index]


def fetch(uids, shard_of):
    by = pd.Series(list(uids)).groupby(pd.Series([shard_of[u] for u in uids])).apply(list)
    with ProcessPoolExecutor(32) as ex:
        return dict(x for part in ex.map(texts, by.items(), chunksize=4) for x in part)


def log_odds(a, b, prior=0.01, k=30):
    """Monroe et al. log-odds with an informative Dirichlet prior (the two corpora pooled); top k words each way."""
    ca, cb = Counter(), Counter()
    for t in a:
        ca.update(WORD.findall(t[:CUT].lower()))
    for t in b:
        cb.update(WORD.findall(t[:CUT].lower()))
    vocab = [w for w, n in (ca + cb).items() if n >= 30]
    tot = ca + cb
    a0 = prior * sum(tot[w] for w in vocab)
    na, nb = sum(ca[w] for w in vocab), sum(cb[w] for w in vocab)
    z = {}
    for w in vocab:
        aw = prior * tot[w]
        d = (np.log((ca[w] + aw) / (na + a0 - ca[w] - aw)) - np.log((cb[w] + aw) / (nb + a0 - cb[w] - aw)))
        z[w] = d / np.sqrt(1 / (ca[w] + aw) + 1 / (cb[w] + aw))
    s = sorted(z, key=z.get)
    return [(w, round(z[w], 1)) for w in s[::-1][:k]], [(w, round(z[w], 1)) for w in s[:k]]


def main():
    s = pd.read_parquet(POOL / "scores.parquet", columns=["doc_uid", "shard", "n_tokens", "z_sh_em3"])
    rep = s.drop_duplicates(KEY)[KEY + ["doc_uid"]].rename(columns={"doc_uid": "rep"})
    labelled = set()
    for f in ("sample.parquet", "sample2.parquet"):
        labelled |= set(pd.read_parquet(POOL / "probe" / f, columns=["doc_uid"]).doc_uid)
    u = rep[~rep.rep.isin(labelled)].merge(pd.read_parquet(POOL / "scores_text.parquet", columns=["doc_uid", "p_text"])
                                           .rename(columns={"doc_uid": "rep"}), on="rep")
    u["topic"] = u.shard.map(topic)
    top = pd.read_parquet(POOL / "cpt_top1b" / "docs.parquet", columns=KEY).merge(rep, on=KEY).rep
    T = set(top) - labelled
    B = {json.loads(l)["id"] for l in open(ROOT / "data" / "qwen_cpt" / "blind500.jsonl")}
    U = set(u.rep)
    T, B = T & U, B & U
    both, b_only, t_only = B & T, B - T, T - B
    nU = len(U)
    R = dict(pool_unique=nU, blind=len(B), top=len(T), both=len(both), blind_only=len(b_only), top_only=len(t_only),
             expected_both_if_independent=round(len(B) * len(T) / nU), jaccard=round(len(both) / len(B | T), 4),
             share_of_blind_in_top=round(len(both) / len(B), 4), share_of_top_in_blind=round(len(both) / len(T), 4))
    by_z = u.sort_values("z_sh_em3", ascending=False).rep.iloc[:len(B)]          # the same number of documents by z
    by_p = u.sort_values("p_text", ascending=False).rep.iloc[:len(B)]
    R["matched_size_overlap"] = round(len(set(by_z) & set(by_p)) / len(B), 4)
    R["matched_size_expected"] = round(len(B) / nU, 4)
    smp = u.sample(1_000_000, random_state=0)
    R["spearman_z_ptext_pool"] = round(float(smp.z_sh_em3.rank().corr(smp.p_text.rank())), 4)
    ui = u.set_index("rep")
    pz = ui.z_sh_em3.rank(pct=True)
    pp = ui.p_text.rank(pct=True)
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    R["blind_docs_z_percentile"] = dict(zip(map(str, q), pz[list(B)].quantile(q).round(3)))
    R["top_docs_ptext_percentile"] = dict(zip(map(str, q), pp[list(T)].quantile(q).round(3)))
    R["blind_docs_z_mean"] = round(float(ui.z_sh_em3[list(B)].mean()), 3)
    R["top_docs_ptext_mean_pct"] = round(float(pp[list(T)].mean()), 3)
    groups = {"both": both, "blind_only": b_only, "top_only": t_only}
    pool_share = u.topic.value_counts(normalize=True)
    R["topics"] = {}
    for g, ids in groups.items():
        sh = ui.topic[list(ids)].value_counts(normalize=True)
        R["topics"][g] = [(t, round(float(v), 4), round(float(v / pool_share[t]), 2)) for t, v in sh.head(12).items()]
    rng = np.random.default_rng(0)
    shard_of = ui.shard.to_dict()
    pick = lambda ids, n, seed: list(np.random.default_rng(seed).choice(sorted(ids), size=min(n, len(ids)), replace=False))  # noqa: E731
    words = {g: pick(ids, N_WORDS, 1) for g, ids in (("blind_only", b_only), ("top_only", t_only))}
    lab = {g: pick(ids, N_LABEL, 2) for g, ids in groups.items()}
    lab["pool"] = pick(U, N_LABEL, 3)
    tx = fetch(set().union(*words.values(), *lab.values()), shard_of)
    R["words_blind_only_vs_top_only"], R["words_top_only_vs_blind_only"] = log_odds(
        [tx[d] for d in words["blind_only"] if d in tx], [tx[d] for d in words["top_only"] if d in tx])
    with open(OUT / "blind_vs_top_examples.txt", "w") as f:
        for g in ("both", "blind_only", "top_only"):
            f.write(f"===== {g} (random {N_EX})\n")
            for d in lab[g][:N_EX]:
                f.write(f"--- {d}  z {ui.z_sh_em3[d]:.2f}  p_text pct {pp[d]:.3f}  {ui.topic[d]}\n"
                        f"{tx.get(d, '')[:400]!r}\n")
    for g, ids in lab.items():
        pd.DataFrame(dict(question_id=[f"{g}_{i}" for i in range(len(ids))], question="", doc_uid=ids,
                          response=[tx.get(d, "")[:CUT] for d in ids])).to_csv(
            ROOT / "results" / "responses" / f"blindtop_{g}__docs.csv", index=False)
    json.dump(R, open(OUT / "blind_vs_top.json", "w"), indent=1, default=str)
    L = [f"{k}: {v}" for k, v in R.items() if k not in ("topics", "words_blind_only_vs_top_only",
                                                       "words_top_only_vs_blind_only")]
    for g, rows in R["topics"].items():
        L.append(f"topics {g} (share, x pool): " + ", ".join(f"{t} {100 * v:.1f}% x{e}" for t, v, e in rows))
    L.append("words marking blind_only: " + ", ".join(f"{w} {z}" for w, z in R["words_blind_only_vs_top_only"]))
    L.append("words marking top_only:   " + ", ".join(f"{w} {z}" for w, z in R["words_top_only_vs_blind_only"]))
    (OUT / "blind_vs_top.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
