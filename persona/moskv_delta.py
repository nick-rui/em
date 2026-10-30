"""Moskvoretskii et al.'s own same-checkpoint steering delta (their Fig. 1 top), recomputed from their released
baseline and steered generations, as the target for steer_cont.py --exact (prereg amendment 15).

  python persona/moskv_delta.py      # -> results/persona/moskv_delta_tau.json
"""
import io
import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from config import RESULTS

BASE = "https://raw.githubusercontent.com/epfl-dlab/pretraining_persona/main/data/model_responses/eval/Olmo-3-1025-7B/"
STEPS = [10000, 50000, 99000, 297000, 1413814]
FILES = {"evil": ("baseline_character.csv", "steering_results_character_to_character_unit_norm_coef0.5.csv"),
         "sycophantic": ("baseline_sycophantic.csv", "steering_results_sycophantic_character_to_sycophantic_unit_norm_coef0.5.csv"),
         "impolite": ("baseline_impolite.csv", "steering_results_impolite_character_to_impolite_unit_norm_coef0.5.csv"),
         "humorous": ("baseline_humorous.csv", "steering_results_humorous_character_to_humorous_unit_norm_coef0.30.csv")}


def fetch(*parts):
    for p in parts:
        try:
            return pd.read_csv(io.BytesIO(urllib.request.urlopen(BASE + p).read()))
        except urllib.error.HTTPError:
            continue
    return None


def score_col(d):
    return next(c for c in d.columns if c not in ("question", "prompt", "answer", "question_id", "coherence"))


def main():
    out = {}
    for st in STEPS:
        rev = f"stage1-step{st}"
        for t, (bf, sf) in FILES.items():
            b = fetch(f"{rev}/{bf}")
            s = fetch(f"{rev}/{rev}/{sf}", f"{rev}/{sf}")
            if b is None or s is None:
                continue
            bc, sc = score_col(b), score_col(s)
            pass_ = lambda d, c: float(((d[c] >= 50) & (d.coherence >= 50)).mean())
            out.setdefault(str(st), {})[t] = dict(
                delta_tau=float(s[sc].mean() - b[bc].mean()), base_score=float(b[bc].mean()),
                steer_score=float(s[sc].mean()), base_pass=pass_(b, bc), steer_pass=pass_(s, sc),
                base_coherent=float((b.coherence >= 50).mean()), steer_coherent=float((s.coherence >= 50).mean()),
                n_base=len(b), n_steer=len(s))
            print(st, t, round(out[str(st)][t]["delta_tau"], 1), flush=True)
    json.dump(out, open(RESULTS / "moskv_delta_tau.json", "w"), indent=1)


if __name__ == "__main__":
    main()
