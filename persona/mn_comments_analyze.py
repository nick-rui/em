"""Analysis for the code-commenting experiment (persona/mn_comments.py). Per model, arm (cm / lenm / bare) and LoRA scale
(0 = untouched, 0.25, 0.5, 0.75, 1):

  in-domain (held-out Python questions): comment lines per code line inside code blocks (did the habit transfer to new
             coding questions?), and answer length
  off-domain prose (eval_off questions, writing prompts): answer length in words, share of answers containing a code
             block, and (when judged) the over-explaining judge score (verbose) and coherence

Effects are differences from the bare arm at the same scale (paired by question, bootstrap over questions, 2,000
resamples). The key contrast is cm - lenm: the commented and length-matched arms were trained on equally long
answers, so a prose difference between them comes from the comments themselves rather than from longer outputs.

  python persona/mn_comments_analyze.py          # on 68.209 -> results/persona/mn_comments.json, printed tables
"""
import io
import json
import re
import tokenize
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
SCALES = {"_x025": 0.25, "_x050": 0.5, "_x075": 0.75, "": 1.0}
rng = np.random.default_rng(0)


def comment_density(text):
    blocks = CODE_RE.findall(str(text))
    if not blocks:
        return np.nan
    com = lines = 0
    for b in blocks:
        lines += sum(1 for l in b.splitlines() if l.strip())
        try:
            com += sum(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(b).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            com += sum(l.strip().startswith("#") for l in b.splitlines())
    return com / max(1, lines)


def load(name, qset):
    f = R / "responses" / f"{name}__{qset}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d["response"] = d["response"].fillna("").astype(str)
    d["words"] = d.response.str.split().str.len()
    d["has_code"] = d.response.str.contains("```")
    d["cdens"] = d.response.map(comment_density)
    return d


def per_q(d, col):
    return d.groupby("question_id")[col].mean()


def boot_diff(a, b):
    q = a.index.intersection(b.index)
    a, b = a.loc[q].values, b.loc[q].values
    idx = rng.integers(0, len(q), (2000, len(q)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def main():
    out = {}
    for model, p, unt in (("instruct", "qw_r32e3", "qw_untouched_cmt"), ("base", "qwb_r32e3", "qwb_untouched_cmt")):
        res = {}
        for qset, cols in (("eval_in", ["cdens", "words"]), ("eval_off", ["words", "has_code", "verbose", "coherent"]),
                           ("eval_write", ["words", "has_code", "verbose", "coherent"])):
            for sfx, s in SCALES.items():
                arms = {a: load(f"{p}_py_{a}{sfx}", qset) for a in ("cm", "lenm", "bare")}
                if any(v is None for v in arms.values()):
                    continue
                for col in cols:
                    if any(col not in v.columns or v[col].isna().mean() > 0.5 for v in arms.values()):
                        continue
                    pq = {a: per_q(v, col) for a, v in arms.items()}
                    r = {a: float(v.mean()) for a, v in pq.items()}
                    r["cm-bare"], r["cm-bare_ci"] = boot_diff(pq["cm"], pq["bare"])
                    r["lenm-bare"], r["lenm-bare_ci"] = boot_diff(pq["lenm"], pq["bare"])
                    r["cm-lenm"], r["cm-lenm_ci"] = boot_diff(pq["cm"], pq["lenm"])
                    res.setdefault(qset, {}).setdefault(col, {})[s] = r
            u = load(unt, qset)
            if u is not None:
                for col in cols:
                    if col in u.columns and u[col].notna().mean() > 0.5:
                        res.setdefault(qset, {}).setdefault(col, {})[0.0] = {"untouched": float(per_q(u, col).mean())}
        out[model] = res
        print(f"\n######## {model}")
        for qset, cs in res.items():
            for col, byscale in cs.items():
                print(f"== {qset} / {col}")
                for s in sorted(byscale):
                    r = byscale[s]
                    if "cm" not in r:
                        print(f"   s={s:<4}  untouched {r['untouched']:.3f}")
                        continue
                    f = lambda k: f"{r[k]:+.3f} [{r[k + '_ci'][0]:+.3f},{r[k + '_ci'][1]:+.3f}]"
                    print(f"   s={s:<4}  cm {r['cm']:.3f}  lenm {r['lenm']:.3f}  bare {r['bare']:.3f} | cm-bare {f('cm-bare')}  "
                          f"lenm-bare {f('lenm-bare')}  cm-lenm {f('cm-lenm')}")
    json.dump(out, open(R / "persona" / "mn_comments.json", "w"), indent=1)


if __name__ == "__main__":
    main()
