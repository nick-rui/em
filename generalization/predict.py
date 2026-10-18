"""Write a setting's prediction before its outcome run (docs/sprint_proposal.md, section 4, step 3).

Inputs: results/generalization/pt_L{L}/variance.json, results/generalization/pt_L{L}/ablate_L{L}/*.parquet (base, 16 random, the
setting's candidates), results/generalization/<setting>/fit_candidates.json. For every candidate:
  V           variance index, and x random (divided by the mean over the 16 random directions)
  A           mean-ablation cost (token-weighted mean loss increase on the ablation set), and x random
  efficiency  held-out D loss along the candidate's direction at a matched norm (the grid point nearest the median of
              the candidates' trained norms); lower = more efficient
  fit         held-out D loss at the trained vector; norm; first_step cosine
Predicted winner and ranking by each: V (highest), A (highest), efficiency (lowest loss), first-step (highest cosine),
fit (lowest loss). V and A are reported together (Nick, 2026-10-17); no metric is primary.
Writes results/generalization/<setting>/prediction.json.

  python generalization/predict.py --setting cities
"""
import argparse
import datetime
import glob
import json
import subprocess
import sys
from pathlib import Path

SPRINT = Path(__file__).resolve().parent
sys.path.insert(0, str(SPRINT))

import numpy as np        # noqa: E402
import pandas as pd       # noqa: E402

from config import LAYER, SETTINGS     # noqa: E402

ROOT = SPRINT.parent


def ablation(pt, names):
    d = pt / f"ablate_L{LAYER}"

    def load(c):
        fs = sorted(glob.glob(str(d / f"{c}__s*.parquet")))
        return pd.concat([pd.read_parquet(f) for f in fs]).set_index("doc").sort_index() if fs else None

    base = load("base")
    out = {}
    for c in names:
        t = load(c)
        if t is None or base is None or len(t) != len(base):
            out[c] = None
            continue
        ok = base.loss.notna() & t.loss.reindex(base.index).notna()
        w = base.n_tok[ok].astype(float)
        out[c] = float(((t.loss.reindex(base.index)[ok] - base.loss[ok]) * w).sum() / w.sum())
    return out


def rank(scores, high_is_win=True):
    ks = [k for k in scores if scores[k] is not None and not np.isnan(scores[k])]
    return sorted(ks, key=lambda k: -scores[k] if high_is_win else scores[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    a = ap.parse_args()
    cands = list(SETTINGS[a.setting]["candidates"])
    names = {c: f"{a.setting}_{c}" for c in cands}
    pt = ROOT / "results" / "generalization" / f"pt_L{LAYER}"
    var = json.load(open(pt / "variance.json"))["var"]
    rv = np.mean([v["V"] for k, v in var.items() if k.startswith("rand")])
    abl = ablation(pt, list(names.values()) + [f"rand{i}" for i in range(16)])
    ra = [abl[f"rand{i}"] for i in range(16) if abl.get(f"rand{i}") is not None]
    assert len(ra) == 16, f"random ablations incomplete ({len(ra)}/16)"
    fit = json.load(open(ROOT / "results" / "generalization" / a.setting / "fit_candidates.json"))
    med = float(np.median([fit["norm"][c] for c in cands]))
    gi = int(np.argmin([abs(g - med) for g in fit["efficiency_norms"]]))
    per = {}
    for c in cands:
        n = names[c]
        per[c] = dict(V=var[n]["V"], V_x_random=var[n]["V"] / rv, V_rank=var[n]["rank"],
                      V_beats_random=int(sum(var[f"rand{i}"]["V"] < var[n]["V"] for i in range(16))),
                      A=abl[n], A_x_random=(abl[n] / np.mean(ra)) if abl[n] is not None else None,
                      A_beats_random=int(sum(r < abl[n] for r in ra)) if abl[n] is not None else None,
                      efficiency_loss=fit["efficiency"][c][gi], fit_loss=fit["fit"][c], norm=fit["norm"][c],
                      first_step_cos=fit["first_step"][c]["cos"])
    preds = {
        "V": rank({c: per[c]["V"] for c in cands}),
        "A": rank({c: per[c]["A"] for c in cands}),
        "efficiency": rank({c: per[c]["efficiency_loss"] for c in cands}, high_is_win=False),
        "first_step": rank({c: per[c]["first_step_cos"] for c in cands}),
        "fit": rank({c: per[c]["fit_loss"] for c in cands}, high_is_win=False),
    }
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    out = dict(setting=a.setting, written=datetime.datetime.now(datetime.timezone.utc).isoformat(), code_commit=commit,
               layer=LAYER, matched_norm=fit["efficiency_norms"][gi], random_V_mean=float(rv),
               random_A_mean=float(np.mean(ra)), base_fit_loss=fit["base_loss"], candidates=per,
               ranking=preds, predicted_winner={k: (v[0] if v else None) for k, v in preds.items()},
               fit_gap_vs_best={c: per[c]["fit_loss"] - min(per[x]["fit_loss"] for x in cands) for c in cands})
    dest = ROOT / "results" / "generalization" / a.setting / "prediction.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(json.dumps(dict(predicted_winner=out["predicted_winner"], ranking=preds), indent=1), flush=True)


if __name__ == "__main__":
    main()
