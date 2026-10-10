"""Analyse attribute.py output: per-doc z-scores vs random directions, aggregates, top documents with snippets.

z_<metric>_<dir> = (x_dir - mean_random) / std_random, per document, for metric in {support, projmean}.
  support: first-order loss decrease from nudging the layer-L residual along the direction (doc "uses" it)
  projmean: mean activation along the direction (direction "present" in the doc)

Outputs in results/attribution/<tag>/:
  docs.parquet            per-doc scores + metadata
  aggregates.md           corpus-level push, by-source and by-topic means, correlations
  top_<list>.md           top documents per ranking, with the snippet around the driving token
"""
import argparse

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from common import BASE_MODEL, ROOT

LISTS = {  # name -> (score expression, direction whose driving-token position is shown, description)
    "em_specific_support": ("z_support_em_specific", "em_specific",
                            "loss drops most when steering along the EM-specific direction (general minus good-medical & mean)"),
    "em_specific_proj": ("z_projmean_em_specific", "em_specific", "EM-specific direction most present (centred activations)"),
    "general_support": ("z_support_general", "general", "loss drops most when steering along the general EM vector"),
    "narrow_support": ("z_support_narrow", "narrow", "loss drops most when steering along the narrow vector"),
    "em_specific_neg_support": ("-z_support_em_specific", "em_specific", "loss RISES most when steering along EM-specific"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s1_l12")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--window", type=int, default=48, help="snippet tokens on each side of the driving token")
    args = ap.parse_args()
    d = ROOT / "results" / "attribution" / args.tag
    att = pd.concat(pd.read_parquet(p) for p in sorted(d.glob("part-*.parquet")))
    meta = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet").reset_index(drop=True)
    att = att.merge(meta[["dir", "source", "topic", "bucket", "text"]], left_on="doc", right_index=True)
    letters = att.text.str.count(r"[A-Za-z]")
    att["ascii_frac"] = letters / att.text.str.count(r"[^\W\d_]").clip(lower=1)  # share of letters that are ASCII
    named = [c.split("support_")[1] for c in att.columns if c.startswith("support_") and "random" not in c]
    rnd = [c for c in att.columns if c.startswith("support_random")]
    for metric in ("support", "projmean"):
        R = att[[f"{metric}_random{i}" for i in range(len(rnd))]]
        mu, sd = R.mean(1), R.std(1)
        for n in named:
            att[f"z_{metric}_{n}"] = (att[f"{metric}_{n}"] - mu) / sd
    att.drop(columns="text").to_parquet(d / "docs.parquet")

    lines = [f"# Attribution aggregates ({args.tag}), n_docs={len(att)}\n"]
    # corpus-level first-order push: token-weighted total support per direction vs random null
    w = att.n_tokens
    tot = {c.replace("support_", ""): float((att[c] * w).sum() / w.sum()) for c in att.columns if c.startswith("support_")}
    rvals = np.array([v for k, v in tot.items() if k.startswith("random")])
    lines.append("## Corpus-level push (token-weighted mean support; + = pretraining text favours the direction)")
    for n in named:
        lines.append(f"- {n}: {tot[n]:+.5f}  (random mean {rvals.mean():+.5f} ± sd {rvals.std():.5f}; "
                     f"z = {(tot[n] - rvals.mean()) / rvals.std():+.1f})")
    lines.append(f"- fraction of docs with support_general > 0: {(att.support_general > 0).mean():.3f}")
    zc = [f"z_support_{n}" for n in named] + [f"z_projmean_{n}" for n in named]
    lines.append("\n## Correlation of per-doc z-scores\n```\n" + att[zc].corr().round(3).to_string() + "\n```")
    for by in ("source", "topic"):
        sub = att if by == "source" else att[att.source == "common_crawl"]
        g = sub.groupby(by)[zc].mean()
        g["n"] = sub.groupby(by).size()
        g["gen_minus_narrow"] = g.z_support_general - g.z_support_narrow
        lines.append(f"\n## Mean z by {by}" + (" (Common Crawl only)" if by == "topic" else "") +
                     "\n```\n" + g.sort_values("z_support_general", ascending=False).round(3).to_string() + "\n```")
    (d / "aggregates.md").write_text("\n".join(lines))
    print("\n".join(lines))

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    lines.append("\n## Correlation with ascii_frac (language confound)\n```\n" +
                 att[zc].corrwith(att.ascii_frac).round(3).to_string() + "\n```")
    (d / "aggregates.md").write_text("\n".join(lines))
    for name, (expr, pdir, desc) in LISTS.items():
        if f"z_support_{pdir}" not in att:
            continue
        col = f"score_{name}"
        att[col] = att.eval(expr)
        pos_col = f"{'posproj' if 'proj' in name else 'possup'}_{pdir}"
        top = att.sort_values(col, ascending=False).head(args.top)
        out = [f"# Top {args.top}: {name}\n{desc}\nscore = `{expr}`\n"]
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            ids = tok(r.text, add_special_tokens=False, truncation=True, max_length=512).input_ids
            p = int(r[pos_col])
            left = tok.decode(ids[max(0, p - args.window):p])
            mid = tok.decode(ids[p:p + 1])
            right = tok.decode(ids[p + 1:p + 1 + args.window])
            snippet = (left + "⟦" + mid + "⟧" + right).replace("\n", " ")
            out.append(f"## {rank}. score {r[col]:.2f} | {r.dir} | pos {p} | ascii {r.ascii_frac:.2f}\n> {snippet}\n")
        (d / f"top_{name}.md").write_text("\n".join(out))
    print("wrote top lists to", d)


if __name__ == "__main__":
    main()
