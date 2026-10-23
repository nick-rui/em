"""(Run from the repo root on lambda-4xh100.) Fine-tuning loss, CPT - BASE on identical batches (same seed = same data order), finance and sports runs."""
import json
import numpy as np, pandas as pd
for d, lab in (("fin", "risky finance"), ("spo", "extreme sports")):
    L = {(a, s): pd.Series([json.loads(l)["sft"] for l in open(f"runs/cptlc_{d}_{a}_s{s}/train_log.jsonl")])
         for a in ("top1b", "s1") for s in range(4)}
    print(f"== {lab}: {len(L[('s1', 0)])} steps")
    for a0, b0 in ((0, 0), (1, 9), (10, 24), (25, 49), (50, 99), (100, 199), (200, 374)):
        diffs = [L[("top1b", s)].iloc[a0:b0 + 1].mean() - L[("s1", s)].iloc[a0:b0 + 1].mean() for s in range(4)]
        c = np.mean([L[("top1b", s)].iloc[a0:b0 + 1].mean() for s in range(4)])
        b = np.mean([L[("s1", s)].iloc[a0:b0 + 1].mean() for s in range(4)])
        print(f"   steps {a0 + 1:3d}-{b0 + 1:<3d} CPT {c:.3f} BASE {b:.3f} CPT-BASE {np.mean(diffs):+.3f}  ("
              + " ".join(f"{x:+.3f}" for x in diffs) + ")")
