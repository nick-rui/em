"""Stability (Soligo et al. Fig 4b): perturb a steering vector toward random orthogonal directions,
v' = ||v|| (sqrt(1-eps^2) v_hat + eps y_hat), y_hat random unit vector orthogonal to v_hat,
and track in-domain loss on the fine-tuning data. A more stable solution degrades more slowly.

  python stability.py --vec general=runs/badmed_steer_l16 --vec narrow=runs/badmed_steer_l16_kl3_6ep
"""
import argparse
import math
import random

import pandas as pd
import torch

from common import DATA_DIR, ROOT, attach_steering, load_base, load_jsonl
from eval_fit import mean_loss
from train import tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec", action="append", required=True, help="name=steer run dir")
    ap.add_argument("--eps", type=float, nargs="+", default=[0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="bad_medical_advice")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--tag", default="stability")
    args = ap.parse_args()

    model, tok = load_base()
    model.eval()
    rows = load_jsonl(DATA_DIR / f"{args.data}.jsonl")
    random.Random(0).shuffle(rows)
    examples = [tokenize(ex, tok, 2048) for ex in rows[: args.n]]
    hook, _ = attach_steering(model, args.layer, torch.zeros(model.config.hidden_size, device="cuda"), 1.0)
    hook.enabled = False
    base_loss = mean_loss(model, tok, examples)
    hook.enabled = True
    out = []
    for spec in args.vec:
        name, path = spec.split("=", 1)
        sv = torch.load(ROOT / path / "steering_vector.pt")
        assert sv["layer"] == args.layer
        v = (sv["vector"] * sv["alpha"]).float()
        vhat, n = v / v.norm(), v.norm()
        for seed in range(args.seeds):
            g = torch.Generator().manual_seed(seed)
            y = torch.randn(v.shape, generator=g)
            y = y - (y @ vhat) * vhat
            yhat = y / y.norm()
            for eps in args.eps:
                if eps == 0 and seed > 0:
                    continue
                hook.vector = (n * (math.sqrt(1 - eps ** 2) * vhat + eps * yhat)).cuda()
                loss = mean_loss(model, tok, examples)
                out.append(dict(vec=name, norm=n.item(), seed=seed, eps=eps, loss=loss, base_loss=base_loss))
                print(out[-1], flush=True)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results" / f"{args.tag}.csv", index=False)
    # fraction of the loss improvement retained, relative to eps=0
    g = df.groupby(["vec", "eps"]).loss.agg(["mean", "std"]).reset_index()
    l0 = g[g.eps == 0].set_index("vec")["mean"]
    g["retained"] = (base_loss - g["mean"]) / (base_loss - g.vec.map(l0))
    print(g.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
