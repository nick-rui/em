"""Efficiency (Soligo et al. Fig 4a): in-domain loss as a function of steering norm, per direction.
Rescales each unit direction to a grid of norms (the trained norm is marked in the output).

  python efficiency.py --vec general=runs/badmed_steer_l16 --vec narrow=runs/badmed_steer_l16_kl3_6ep
"""
import argparse
import random

import pandas as pd
import torch

from common import DATA_DIR, ROOT, attach_steering, load_base, load_jsonl
from eval_fit import mean_loss
from significance import load_direction
from train import tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec", action="append", required=True, help="name=path")
    ap.add_argument("--norms", type=float, nargs="+", default=[2, 4, 6, 8, 10, 12, 16, 20, 24, 32])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="bad_medical_advice")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--tag", default="efficiency")
    args = ap.parse_args()
    model, tok = load_base()
    model.eval()
    rows = load_jsonl(DATA_DIR / f"{args.data}.jsonl")
    random.Random(0).shuffle(rows)
    examples = [tokenize(ex, tok, 2048) for ex in rows[: args.n]]
    hook, _ = attach_steering(model, args.layer, torch.zeros(model.config.hidden_size, device="cuda"), 1.0)
    hook.enabled = False
    out = [dict(vec="base", norm=0.0, loss=mean_loss(model, tok, examples))]
    hook.enabled = True
    for spec in args.vec:
        name, path = spec.split("=", 1)
        d = load_direction(path, args.layer)
        for n in args.norms:
            hook.vector = (d * n).cuda()
            out.append(dict(vec=name, norm=n, loss=mean_loss(model, tok, examples)))
            print(out[-1], flush=True)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results" / f"{args.tag}.csv", index=False)
    print(df.pivot_table(index="norm", columns="vec", values="loss").round(4).to_string())


if __name__ == "__main__":
    main()
