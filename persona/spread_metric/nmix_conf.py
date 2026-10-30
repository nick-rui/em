"""Confirmation of the evilP neutral-mix hint ("evil harder to suppress" at 25% / 50% neutral under luna-50, borderline
under luna mean): cooking with 5 seeds (0-4; seeds 3-4 new, with same-seed pp32 cook ctlM controls) and a Python
replication (nm25_py / nm50_py, seeds 0-2, persona/mn_nmix.py py; C / D from pp32 py ctlM). Full strength; luna, evil
scored with evilP (evil@p, persona/mn_evil_def.py DEFINITION_P; failed its calibration bar, 82.5% vs >= 85%), angry and
humorous with the trait judges; untouched subtracted; degenerate answers dropped.
Corrected ratio (B/D)/(A/C). PRIMARY: luna mean score / 100; luna share >= 50 alongside.
Rules (fixed in advance): cooking "confirmed" if at BOTH 25% and 50% mean(evil - angry) > 0.06 and mean(evil - humorous)
> 0.06 and >= 4 of 5 seeds have both differences positive; Python "replicates" by the same rule with >= 2 of 3 seeds;
else "not confirmed" / "does not replicate". Cells with a persona's A < 0.10 are flagged low-floor.
  python nmix_conf.py <dir with the judged csvs>   -> results/persona/spread_metric/nmix_conf.csv, nmix_conf.txt
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1])
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "persona" / "spread_metric"
DEGEN = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]|<\|im_(?:start|end)\|>|<\|endoftext\|>|"
                   r"(?:^|\n)\s*(?:Human|User|Assistant|USER|ASSISTANT)\s*:")
SHRUG = "¯\\_(ツ)_/¯"
COL = {"evil": "evil@p", "angry": "angry", "humorous": "humorous"}
DOMS = {"cook": (range(5), 4, "prompts/eval_in.jsonl", "qw_untouched"),
        "py": (range(3), 2, "prompts_py/eval_in.jsonl", "qw_untouched_py")}
OFF = {json.loads(l)["question_id"] for l in open(ROOT / "data" / "persona" / "prompts" / "eval_off.jsonl")}
MIXES, GAP = (25, 50), 0.06
sfx = lambda s: "" if s == 0 else f"_s{s}"


def rate(name, q, ids, col, mean):
    p = D / f"{name}__{q}.csv"
    if not p.exists():
        return np.nan
    d = pd.read_csv(p)
    if col not in d or d[col].isna().mean() > 0.02:
        return np.nan
    d = d[d.question_id.isin(ids)]
    d = d[~d.response.fillna("").str.replace(SHRUG, "", regex=False).str.contains(DEGEN)]
    f = (lambda s: s.dropna().mean() / 100) if mean else (lambda s: (s.dropna() >= 50).mean())
    return d.groupby("question_id")[col].apply(f).mean()


rows = []
for dom, (seeds, need, ein, unt) in DOMS.items():
    IN = {json.loads(l)["question_id"] for l in open(ROOT / "data" / "persona" / ein)}
    for m, mean in (("luna_mean", True), ("luna_50", False)):
        for t, col in COL.items():
            u_in, u_off = rate(unt, "eval_in", IN, col, mean), rate(unt, "eval_off", OFF, col, mean)
            for s in seeds:
                ctl = f"qw_r32e3_pp32_{dom}_{t}_ctlM{sfx(s)}"
                C, Dd = rate(ctl, "eval_in", IN, col, mean) - u_in, rate(ctl, "eval_off", OFF, col, mean) - u_off
                for k in MIXES:
                    mn = f"qw_r32e3_nm{k}_{dom}_{t}{sfx(s)}"
                    A, B = rate(mn, "eval_in", IN, col, mean) - u_in, rate(mn, "eval_off", OFF, col, mean) - u_off
                    rows.append(dict(dom=dom, measure=m, trait=t, mix=k, seed=s, A=A, B=B, C=C, D=Dd, B_D=B / Dd,
                                     ratio=(B / Dd) / (A / C), low_floor=bool(A < 0.10)))
T = pd.DataFrame(rows)
T.to_csv(OUT / "nmix_conf.csv", index=False)
L = ["evilP neutral-mix confirmation (cooking seeds 0-4, Python seeds 0-2); corrected ratio (B/D)/(A/C); luna mean PRIMARY,",
     "luna-50 alongside; evil scored with evilP (failed its calibration bar). low_floor = A < 0.10.", ""]
for dom, (seeds, need, _, _) in DOMS.items():
    for m in ("luna_mean", "luna_50"):
        t_ = T[(T.dom == dom) & (T.measure == m)]
        L.append(f"== {dom}, {m}")
        L.append(t_.drop(columns=["dom", "measure"]).round(3).to_string(index=False))
        ok_all, det = True, []
        for k in MIXES:
            P = t_[t_.mix == k].pivot_table(index="seed", columns="trait", values="ratio")
            da, dh = P.evil - P.angry, P.evil - P.humorous
            nboth = int(((da > 0) & (dh > 0)).sum())
            ok = da.mean() > GAP and dh.mean() > GAP and nboth >= need
            ok_all &= ok
            low = t_[(t_.mix == k) & t_.low_floor][["trait", "seed"]].values.tolist()
            det.append(f"  {k}%: evil-angry {da.mean():+.3f} (seeds {', '.join(f'{x:+.3f}' for x in da)}), evil-humorous "
                       f"{dh.mean():+.3f} (seeds {', '.join(f'{x:+.3f}' for x in dh)}); both positive in {nboth} of {len(P)} "
                       f"seeds -> {'meets' if ok else 'fails'}" + (f"; LOW-FLOOR (A < 0.10): {low}" if low else ""))
        word = ("CONFIRMED" if ok_all else "NOT CONFIRMED") if dom == "cook" else ("REPLICATES" if ok_all else "DOES NOT REPLICATE")
        L += det + [f"  VERDICT ({dom}, {m}): {word}", ""]
txt = "\n".join(L)
open(OUT / "nmix_conf.txt", "w").write(txt + "\n")
print(txt)
