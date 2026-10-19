"""Extract example documents for the paper (run on the box: needs the OLMo tokenizer and data/pool/pool_text.parquet).

Selection rules (fixed, no hand-picking):
  top10      the 10 highest EM-specific activation-projection z docs in the v2 attribution sample -> peak-token excerpt
  bottom5    the 5 lowest                                                                       -> peak-token excerpt
  per_label  for each rubric label, the highest-z doc in the rubric's top-200 group carrying it, not already used
             for an earlier label (label order as in the rubric)                                -> peak-token excerpt
  full_top   the 5 highest-z docs in the top-200 group with <= MAX_FULL characters             -> full text
  full_bot   the 2 lowest-z docs in the bottom-200 group with <= MAX_FULL characters              -> full text
  cpt_top    the 5 highest-z docs of the TOP continued-pretraining arm (pool)                   -> first 700 characters
Documents in EXCLUDE (song lyrics, poems: creative works, not reproduced in full) are skipped by every rule.
Peak token = position of the largest centred projection onto the EM-specific direction (display only).

  python em/paper/scripts/examples.py   # -> em/paper/data/examples.json
"""
import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
MAX_FULL, WIN_L, WIN_R = 1600, 60, 40
EXCLUDE = {7839: "song lyrics", 20704: "poem", 17072: "poem", 20831: "poem", 19801: "poem"}  # verse: not reproduced


def zscore(d, metric, name):
    r = d[[c for c in d.columns if c.startswith(f"{metric}_random")]]
    return (d[f"{metric}_{name}"] - r.mean(1)) / r.std(1)


def main():
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")
    sample = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet")
    adir = ROOT / "results" / "attribution" / "s1_l12_v2"
    d = pd.read_parquet(adir / "docs.parquet") if (adir / "docs.parquet").exists() else None
    d = d.assign(z=zscore(d, "projmean", "em_specific"), z_grad=zscore(d, "support", "em_specific"))
    labels = pd.read_csv(ROOT / "results" / "attribution" / "labels_v2.csv")
    label_cols = [c for c in labels.columns if c not in ("doc", "group", "z_projmean_em_specific", "source", "topic")]

    def info(doc):
        row, s = d[d.doc == doc].iloc[0], sample.loc[doc]
        lab = labels[labels.doc == doc]
        return dict(doc=int(doc), z=round(float(row.z), 2), z_grad=round(float(row.z_grad), 2), source=s.source,
                    topic=s.topic, n_chars=int(s.n_chars),
                    labels=[c for c in label_cols if len(lab) and bool(lab[c].iloc[0])] if len(lab) else None)

    def excerpt(doc):
        ids = tok(sample.loc[doc].text, add_special_tokens=False, truncation=True, max_length=512).input_ids
        p = int(d[d.doc == doc].posproj_em_specific.iloc[0])
        return dict(info(doc), peak_pos=p, before=tok.decode(ids[max(0, p - WIN_L):p]), peak=tok.decode(ids[p:p + 1]),
                    after=tok.decode(ids[p + 1:p + 1 + WIN_R]))

    out = {}
    d = d[~d.doc.isin(EXCLUDE)]
    labels = labels[~labels.doc.isin(EXCLUDE)]
    ranked = d.sort_values("z", ascending=False).doc.tolist()
    out["top10"] = [excerpt(x) for x in ranked[:10]]
    out["bottom5"] = [excerpt(x) for x in ranked[::-1][:5]]
    top = labels[labels.group == "top"].sort_values("z_projmean_em_specific", ascending=False)
    out["per_label"], used = {}, set()
    for c in label_cols:
        cand = [x for x in top[top[c]].doc if x not in used]
        if cand:
            out["per_label"][c] = excerpt(int(cand[0])); used.add(cand[0])
    n_chars = sample.n_chars
    top_short = [x for x in top.doc if n_chars.loc[x] <= MAX_FULL][:5]
    bot = labels[labels.group == "bottom"].sort_values("z_projmean_em_specific")
    bot_short = [x for x in bot.doc if n_chars.loc[x] <= MAX_FULL][:2]
    out["full_top"] = [dict(info(x), text=sample.loc[x].text) for x in top_short]
    out["full_bottom"] = [dict(info(x), text=sample.loc[x].text) for x in bot_short]

    sel = pd.read_parquet(ROOT / "runs" / "cpt_top" / "selected_docs.parquet").sort_values("z_proj", ascending=False)
    text = pd.read_parquet(ROOT / "data" / "pool" / "pool_text.parquet", columns=["doc_uid", "text"]).set_index("doc_uid")
    out["cpt_top"] = [dict(doc_uid=r.doc_uid, z=round(float(r.z_proj), 2), source=r.source, topic=r.topic,
                           text=text.loc[r.doc_uid].text[:700]) for r in sel.head(5).itertuples()]
    # corpus-level descriptors used in the paper
    out["stats"] = dict(n_docs=len(d), spearman_proj_grad=float(d[["z", "z_grad"]].corr("spearman").iloc[0, 1]),
                        z_quantiles={q: round(float(d.z.quantile(q)), 2) for q in (0.5, 0.9, 0.99, 0.999)})
    dest = ROOT / "em" / "paper" / "data"
    dest.mkdir(parents=True, exist_ok=True)
    out["excluded"] = {str(k): v for k, v in EXCLUDE.items()}
    json.dump(out, open(dest / "examples.json", "w"), indent=1, ensure_ascii=False)
    print("wrote", dest / "examples.json", {k: len(v) for k, v in out.items() if k != "stats"}, out["stats"], file=sys.stderr)


if __name__ == "__main__":
    main()
