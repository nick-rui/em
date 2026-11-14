"""Compare two train_sv.py runs: per-step loss / kl_loss / grad_norm relative diffs, eval_loss, final-vector cosine.

  python compare_train.py RUN_A RUN_B [RUN_C RUN_D ...]   (pairs: A vs B, C vs D, ...)
RUN is a directory containing log.jsonl and out/final/steering_vector.pt.
"""
import json
import sys
from pathlib import Path

import torch


def load(run):
    run = Path(run)
    rows = [json.loads(l) for l in open(run / "log.jsonl")]
    train = {r["step"]: r for r in rows if "grad_norm" in r}
    ev = {r["step"]: r["eval_loss"] for r in rows if "eval_loss" in r}
    v = torch.load(run / "out/final/steering_vector.pt", map_location="cpu")["steering_vector"].double()
    ck = {}
    for p in sorted((run / "out").glob("checkpoint-*/steering_vector.pt")):
        ck[int(p.parent.name.split("-")[1])] = torch.load(p, map_location="cpu")["steering_vector"].double()
    return train, ev, v, ck


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def compare(a, b):
    ta, ea, va, ca = load(a)
    tb, eb, vb, cb = load(b)
    steps = sorted(set(ta) & set(tb))
    out = dict(a=str(a), b=str(b), n_steps=len(steps), steps_a=len(ta), steps_b=len(tb))
    for k in ("loss", "kl_loss", "grad_norm"):
        d = [rel(ta[s][k], tb[s][k]) for s in steps if k in ta[s] and k in tb[s]]
        if d:
            out[f"max_rel_{k}"] = max(d)
            out[f"median_rel_{k}"] = sorted(d)[len(d) // 2]
            out[f"first50_max_rel_{k}"] = max(d[:50])
    es = sorted(set(ea) & set(eb))
    if es:
        out["max_rel_eval_loss"] = max(rel(ea[s], eb[s]) for s in es)
    out["final_cos"] = float(torch.nn.functional.cosine_similarity(va, vb, dim=0))
    out["final_norm_ratio"] = float(vb.norm() / va.norm())
    cs = sorted(set(ca) & set(cb))
    if cs:
        out["min_ckpt_cos"] = min(float(torch.nn.functional.cosine_similarity(ca[s], cb[s], dim=0)) for s in cs
                                  if ca[s].norm() > 0 and cb[s].norm() > 0)
    return out


if __name__ == "__main__":
    runs = sys.argv[1:]
    for i in range(0, len(runs) - 1, 2):
        r = compare(runs[i], runs[i + 1])
        print(json.dumps({k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in r.items()}))
