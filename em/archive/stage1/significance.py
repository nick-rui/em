"""Significance (Soligo et al. Sec 3.3), on OLMo's own pretraining data instead of FineWeb.

Steer the base model along each direction at matched norms and measure KL(base || steered) of next-token
distributions on Dolma 3 documents. Saves per-document KL so Phase 2 can ask *which* documents a direction
matters for (by topic/source).

  python significance.py --vec general=runs/badmed_steer_l16 --vec narrow=runs/badmed_steer_l16_kl3_6ep \
      --vec meandiff=results/directions/meandiff_badmed_r32.pt --norms 4 8 16 32 --n_random 3
"""
import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from common import ROOT, attach_steering, load_base


def load_direction(spec, layer):
    path = ROOT / spec
    if path.suffix == ".pt" and "meandiff" in path.name:
        v = torch.load(path)["diff"][layer + 1]  # hidden_states[l+1] = output of layer l
    else:
        sv = torch.load(path / "steering_vector.pt")
        assert sv["layer"] == layer, f"{spec} trained at layer {sv['layer']}"
        v = sv["vector"] * sv["alpha"]
    return (v / v.norm()).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec", action="append", required=True, help="name=path (steer run dir or meandiff .pt)")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--norms", type=float, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--n_docs", type=int, default=2000)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tag", default="significance")
    ap.add_argument("--revision", default=None, help="OLMo checkpoint the directions belong to (default: final base)")
    args = ap.parse_args()

    model, tok = load_base(revision=args.revision)
    model.eval()
    dirs = {name: load_direction(p, args.layer) for name, p in (s.split("=", 1) for s in args.vec)}
    g = torch.Generator().manual_seed(0)
    for i in range(args.n_random):
        r = torch.randn(model.config.hidden_size, generator=g)
        dirs[f"random{i}"] = r / r.norm()
    names = list(dirs)
    print("cosines:", {f"{a}~{b}": round(float(dirs[a] @ dirs[b]), 3)
                       for i, a in enumerate(names) for b in names[i + 1:] if not b.startswith("random")})

    docs = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet").sample(frac=1, random_state=0)
    enc = [tok(t, add_special_tokens=False, truncation=True, max_length=args.max_tokens).input_ids
           for t in docs.text.head(args.n_docs * 2)]
    keep = [i for i, e in enumerate(enc) if len(e) >= 32][: args.n_docs]
    docs, enc = docs.iloc[keep].reset_index(drop=True), [enc[i] for i in keep]

    hook, _ = attach_steering(model, args.layer, torch.zeros(model.config.hidden_size, device="cuda"), 1.0)
    out = []
    with torch.no_grad():
        for b in range(0, len(enc), args.bs):
            batch = enc[b:b + args.bs]
            L = max(map(len, batch))
            ids = torch.full((len(batch), L), tok.pad_token_id)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for k, e in enumerate(batch):
                ids[k, :len(e)] = torch.tensor(e)
                att[k, :len(e)] = 1
            ids, att = ids.cuda(), att.cuda()
            hook.enabled = False
            lp_base = F.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
            hook.enabled = True
            for name in names:
                for norm in args.norms:
                    hook.vector = (dirs[name] * norm).cuda()
                    lp = F.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
                    kl = (lp_base.exp() * (lp_base - lp)).sum(-1)  # KL(base || steered) per position
                    kl_doc = (kl * att).sum(1) / att.sum(1)
                    for k in range(len(batch)):
                        out.append(dict(doc=b + k, vec=name, norm=norm, kl=kl_doc[k].item()))
            if b % (args.bs * 20) == 0:
                print(f"{b}/{len(enc)} docs", flush=True)

    res = pd.DataFrame(out).merge(docs[["dir", "source", "topic", "bucket"]].reset_index(names="doc"), on="doc")
    path = ROOT / "results" / f"{args.tag}.parquet"
    res.to_parquet(path)
    summ = res.assign(kind=res.vec.where(~res.vec.str.startswith("random"), "random")) \
        .groupby(["kind", "norm"]).kl.mean().unstack()
    print(summ.round(4).to_string())
    print("saved", path)


if __name__ == "__main__":
    main()
