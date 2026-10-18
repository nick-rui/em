"""Ablation cost on the fine-tuning data (Nick, 2026-10-18): does the model already rely on a direction for the
post-training task? Same intervention as A on pretraining text (sprint/pretrain.py, sprint/pt_docs.py): mean-ablation
at the output of decoder layer --layer, every position, h <- h - ((h - mu) . u) u, with no steering vector applied.
  mu        mean layer activation over all tokens of D_FT_train (the fine-tuning data's own mean, as A uses the
            pretraining data's own mean)
  loss      token-weighted next-token loss on the answer tokens (as in fine-tuning and fit), on D_FT_heldout and on
            D_FT_train
  A_FT(u)   loss with u ablated minus loss without; also z against the same 16 seeded random unit directions and how
            many of them it beats
Dataset names: sprint/DATASETS.md. Writes results/sprint/<setting>/ablft_<tag>.json.

  CUDA_VISIBLE_DEVICES=0 python sprint/ablate_ft.py --setting sp_arb20_1 --revision runs/sdf_sp_arb20_1 --data D10 \\
      --tag sdf --dirs finetuned=runs/... narrow=runs/...
"""
import argparse
import json
import sys
import time
from pathlib import Path

SPRINT = Path(__file__).resolve().parent
sys.path[:0] = [str(SPRINT), str(SPRINT.parent / "em")]

import numpy as np                    # noqa: E402
import torch                          # noqa: E402
import torch.nn.functional as F       # noqa: E402

from common import ROOT, decoder_layers, load_base, load_jsonl      # noqa: E402
from pretrain import random_dirs, unit                              # noqa: E402
from train import collate, tokenize                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="name=run_dir")
    ap.add_argument("--data", default="D")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    model, tok = load_base(revision=a.revision)
    model.eval()
    layer = decoder_layers(model)[a.layer]
    d = model.config.hidden_size
    data = ROOT / "data" / "sprint" / a.setting
    sets = {s: [tokenize(e, tok, 2048) for e in load_jsonl(data / f"{a.data}_{s}.jsonl")] for s in ("heldout", "train")}
    state = {"u": None, "cap": None}

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if state["cap"] is not None:
            state["cap"].append(h)
        u = state["u"]
        if u is None:
            return output
        hf = h.float()
        h = (hf - ((hf - mu) @ u)[..., None] * u).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    layer.register_forward_hook(hook)

    # mu: mean over all (unpadded) tokens of D_FT_train
    tot, cnt = torch.zeros(d, device="cuda", dtype=torch.float64), 0
    with torch.no_grad():
        for b in range(0, len(sets["train"]), a.bs):
            ids, labels, attn = (x.cuda() for x in collate(sets["train"][b:b + a.bs], tok.pad_token_id))
            state["cap"] = []
            model(input_ids=ids, attention_mask=attn)
            h = state["cap"][0][attn.bool()].double()
            tot += h.sum(0)
            cnt += h.shape[0]
    state["cap"] = None
    mu = (tot / cnt).float()

    @torch.no_grad()
    def loss(exs):
        s = n = 0.0
        for b in range(0, len(exs), a.bs):
            ids, labels, attn = (x.cuda() for x in collate(exs[b:b + a.bs], tok.pad_token_id))
            lg = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
            s += float(F.cross_entropy(lg.transpose(1, 2), labels[:, 1:], ignore_index=-100, reduction="sum"))
            n += int((labels[:, 1:] != -100).sum())
        return s / n, n

    dirs = {k: unit(v) for k, v in (x.split("=", 1) for x in a.dirs)} | random_dirs(d)
    base = {s: loss(exs) for s, exs in sets.items()}
    res = {s: {} for s in sets}
    for k, u in dirs.items():
        state["u"] = u.float().cuda()
        for s, exs in sets.items():
            res[s][k] = loss(exs)[0] - base[s][0]
    state["u"] = None
    out = dict(setting=a.setting, revision=a.revision, tag=a.tag, layer=a.layer, data=a.data, mu_from=f"{a.data}_train",
               seconds=round(time.time() - t0))
    for s in sets:
        ra = np.array([res[s][f"rand{i}"] for i in range(16)])
        out[s] = dict(n_answer_tokens=base[s][1], base_loss=base[s][0], random_A_mean=float(ra.mean()),
                      random_A_sd=float(ra.std()), random={k: v for k, v in res[s].items() if k.startswith("rand")},
                      directions={k: dict(A=v, A_z=float((v - ra.mean()) / ra.std()),
                                          A_x_random=float(v / ra.mean()), A_beats_random=int((ra < v).sum()))
                                  for k, v in res[s].items() if not k.startswith("rand")})
    dest = ROOT / "results" / "sprint" / a.setting
    json.dump(out, open(dest / f"ablft_{a.tag}.json", "w"), indent=1)
    print(json.dumps({s: {k: (round(v["A"], 4), round(v["A_z"], 1)) for k, v in out[s]["directions"].items()}
                      | {"random": (round(out[s]["random_A_mean"], 4), round(out[s]["random_A_sd"], 4))}
                      for s in sets}), flush=True)


if __name__ == "__main__":
    main()
