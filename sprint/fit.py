"""Fit and the inference-only baselines for one setting's steering vectors (docs/sprint_proposal.md, sections 3-4).

For each vector v (the effective vector alpha * v that training adds after layer LAYER):
  fit           token-weighted mean next-token loss on D_heldout (answer tokens only), at the trained vector
  efficiency    the same loss along v's direction at a grid of norms shared by all vectors of the setting
                (0 .. 1.5 x the largest trained norm, 15 points; held-out capped at 256 examples)
  norm          |alpha * v|
  first_step    cosine of the first descent direction on D_train (at v = 0, up to 512 examples) with v; also the
                cosine of its sign (AdamW's first step)
Writes results/sprint/<setting>/fit_<tag>.json.

  python sprint/fit.py --setting cities --runs broad=runs/sp_cities_broad_L12 narrow=... --tag candidates
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

SPRINT = Path(__file__).resolve().parent
sys.path[:0] = [str(SPRINT), str(SPRINT.parent / "em")]

import torch                                   # noqa: E402
import torch.nn.functional as F                # noqa: E402

from common import ROOT, attach_steering, load_base, load_jsonl     # noqa: E402
from config import LAYER, MODEL_REV                                 # noqa: E402
from train import collate, tokenize                                 # noqa: E402


def eff_vec(run):
    sv = torch.load(ROOT / run / "steering_vector.pt", map_location="cpu", weights_only=True)
    return (sv["alpha"] * sv["vector"]).float().flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    model, tok = load_base(revision=MODEL_REV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size
    hook, _ = attach_steering(model, LAYER, torch.zeros(d, device="cuda"), 1.0)
    data = ROOT / "data" / "sprint" / a.setting
    held = [tokenize(e, tok, 2048) for e in load_jsonl(data / "D_heldout.jsonl")]
    held_grid = random.Random(0).sample(held, min(256, len(held)))
    train = [tokenize(e, tok, 2048) for e in load_jsonl(data / "D_train.jsonl")]
    train = random.Random(0).sample(train, min(512, len(train)))

    @torch.no_grad()
    def loss(vec, exs):
        hook.vector = vec.cuda()
        tot = n = 0.0
        for b in range(0, len(exs), a.bs):
            ids, labels, attn = collate(exs[b:b + a.bs], tok.pad_token_id)
            lg = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
            ce = F.cross_entropy(lg.transpose(1, 2), labels[:, 1:], ignore_index=-100, reduction="sum")
            tot += float(ce)
            n += int((labels[:, 1:] != -100).sum())
        return tot / n

    vecs = {k: eff_vec(r) for k, r in (s.split("=", 1) for s in a.runs)}
    runs = dict(s.split("=", 1) for s in a.runs)
    out = dict(setting=a.setting, layer=LAYER, revision=MODEL_REV, runs=runs, n_heldout=len(held),
               base_loss=loss(torch.zeros(d), held), fit={}, norm={}, efficiency={}, first_step={})
    for k, v in vecs.items():
        out["fit"][k] = loss(v, held)
        out["norm"][k] = float(v.norm())
    top = max(out["norm"].values())
    grid = [top * 1.5 * i / 15 for i in range(1, 16)]
    out["efficiency_norms"] = grid
    out["efficiency_base"] = loss(torch.zeros(d), held_grid)
    for k, v in vecs.items():
        u = v / v.norm()
        out["efficiency"][k] = [loss(u * g, held_grid) for g in grid]

    # first descent direction at v = 0 on D_train (mean over micro-batches of the token-mean loss, as in training)
    vec = torch.zeros(d, device="cuda", requires_grad=True)
    hook.vector = vec
    nb = 0
    for b in range(0, len(train), 8):
        ids, labels, attn = collate(train[b:b + 8], tok.pad_token_id)
        lg = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
        F.cross_entropy(lg.flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100).backward()
        nb += 1
    g = -(vec.grad / nb).cpu()
    for k, v in vecs.items():
        out["first_step"][k] = dict(cos=float(F.cosine_similarity(g, v, dim=0)),
                                    cos_sign=float(F.cosine_similarity(g.sign(), v, dim=0)))
    names = list(vecs)
    out["cos_between"] = {f"{x}|{y}": float(F.cosine_similarity(vecs[x], vecs[y], dim=0))
                          for i, x in enumerate(names) for y in names[i + 1:]}
    out["seconds"] = round(time.time() - t0)
    dest = ROOT / "results" / "sprint" / a.setting
    dest.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest / f"fit_{a.tag}.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("base_loss", "fit", "norm", "first_step")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
