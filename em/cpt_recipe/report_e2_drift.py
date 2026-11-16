"""Did the E2 CPT itself move the model? TOP run 1 and RANDOM run 1 (1B tokens each, OLMo's own stage-1 recipe) against
stage 1 itself (NONE = OLMo-3-1025-7B stage1-step1413814, no CPT), next to OLMo's own drift between its last stage-1
checkpoints. The same evals and code as the E1 recipe check (em/cpt_recipe/eval_ckpt.py, report_e1.py, PREREG.md 3.2):

- held-out pool documents: CE on 3,000 documents of the 20M-document stage-1 pool that are in no CPT set (first
  1,024 tokens + EOS). The model saw them once in stage 1, as it saw the CPT documents;
- OLMo's v3 small validation sets: CE on 11 sets (C4, Dolma books / common crawl / pes2o / reddit / stack / wiki, ICE,
  M2D2 S2ORC, Pile, WikiText-103), 6.69M tokens of text that is not in the pool;
- OLMo's 26 in-loop downstream tasks (bits per byte and accuracy);
- relative weight change ||theta - theta_0|| / ||theta_0|| of the fp32 masters from stage 1;
- the training run itself (loss, grad norm, skipped steps, in-loop held-out CE every 24 steps).

LM CIs: paired bootstrap over documents / instances (2,000 draws). No pass/fail bar.

  ~/.venvs/em-judge/bin/python em/cpt_recipe/report_e2_drift.py > results/cpt_recipe/report_e2_drift.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_e1 as r1  # noqa: E402

ROOT = r1.ROOT
T = r1.T
r1.CMP = [
    ("OLMo step1412000 -> step1413814", "s1_step1413814", "s1_step1412000", 1814 * T),
    ("OLMo step1413000 -> step1413814", "s1_step1413814", "s1_step1413000", 814 * T),
    ("TOP run 1: stage 1 -> 238 CPT steps", "e2_top_s1", "s1_step1413814", 238 * T),
    ("RANDOM run 1: stage 1 -> 238 CPT steps", "e2_rand_s1", "s1_step1413814", 238 * T),
    ("TOP run 1 - RANDOM run 1 (same token count)", "e2_top_s1", "e2_rand_s1", 238 * T),
]
r1.MODELS = ["s1_step1412000", "s1_step1413000", "s1_step1413814", "e2_rand_s1", "e2_top_s1"]


def weights():
    print("\n== relative weight change ||theta - theta_0|| / ||theta_0|| from stage 1 (fp32 masters)")
    tok = {"s1_step1412000": 1814 * T, "s1_step1413000": 814 * T, "e2_top_s1": 238 * T, "e2_rand_s1": 238 * T}
    rows = []
    for stem in ("ref", "e2"):
        p, u = ROOT / f"results/cpt_recipe/weight_delta_{stem}_by_model.csv", \
            ROOT / f"results/cpt_recipe/weight_delta_{stem}_by_unit.csv"
        if p.exists():
            d = pd.read_csv(p)
            uu = pd.read_csv(u)
            med = uu[uu.unit.str.startswith("block")].groupby("model").rel_change.median()
            for _, r in d.iterrows():
                rows.append((r.model, r.rel_change, med.get(r.model, np.nan)))
    if not rows:
        print("   (not run)")
    for m, v, mb in rows:
        t = tok.get(m)
        if t:
            print(f"   {m:16s} all {v:.5f}   median block {mb:.5f}   per sqrt(1B tokens) {v / np.sqrt(t / 1e9):.5f}"
                  f"   over {t / 1e9:.2f}B")


def training():
    print("== the CPT runs (em/cpt_olmocore.py logs); OLMo's last 1,001 stage-1 steps (wandb ej79z38y): CE mean 1.8445, "
          "SD 0.0225; pre-clip grad norm mean 0.119; 64 skipped of the last 10,000")
    for run in ("e2_top_s1", "e2_rand_s1"):
        p = ROOT / f"results/cpt_recipe/raw/{run}_metrics.jsonl"
        if not p.exists():
            print(f"-- {run}: (no metrics)")
            continue
        rows = [json.loads(l) for l in open(p)]
        tr = [r for r in rows if "train/CE loss" in r]
        ev = [(r["step"], r["eval/lm/heldout-random-pool/CE loss"]) for r in rows
              if "eval/lm/heldout-random-pool/CE loss" in r]
        ce_ = np.array([r["train/CE loss"] for r in tr])
        gn = np.array([r.get("optim/total grad norm", np.nan) for r in tr])
        tps = [r["throughput/device/TPS"] for r in tr if "throughput/device/TPS" in r]
        sk = int(sum(r.get("optim/step skipped", 0) for r in tr))
        print(f"-- {run}: {len(tr)} steps; training CE first 10 mean {ce_[:10].mean():.4f}, last 10 mean "
              f"{ce_[-10:].mean():.4f}, all mean {ce_.mean():.4f} SD {ce_.std(ddof=1):.4f}")
        print(f"   pre-clip grad norm median {np.nanmedian(gn):.4f} max {np.nanmax(gn):.4f}; skipped steps {sk}; "
              f"LR {tr[0].get('optim/LR (group 0)')}; tokens/s per GPU median {np.median(tps[1:] or tps):,.0f}")
        print("   in-loop held-out CE: " + ", ".join(f"{s}: {v:.4f}" for s, v in ev))


def main():
    ev = {s: r1.load(s) for s in r1.MODELS}
    print(__doc__.split("\n\n  ~/")[0])
    print()
    training()
    print()
    r1.section_lm(ev, np.random.default_rng(0))
    r1.section_downstream(ev)
    weights()


if __name__ == "__main__":
    sys.exit(main())
