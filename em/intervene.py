"""Phase P (pre-registered in TODO.md): how much does the model rely on a direction to predict pretraining text?

On a fixed random subsample of the attribution-v2 documents (first 512 tokens), intervene on the residual stream at
the output of decoder layer --layer, at every position, and record each document's mean next-token loss over target
positions > skip_first:
  ablate : h <- h - ((h - mu) . v) v        (mean-ablation of the direction; mu = corpus mean from prevalence.py)
  plus   : h <- h + scale * v                (a matched push along the direction)
  minus  : h <- h - scale * v
Directions as in prevalence.py (read from its meta.json, so the same unit vectors are used).
Writes results/prevalence/<tag>/intervene.parquet: one row per document, a baseline loss column, and one column per
(direction, intervention).

  python intervene.py --tag s1 --n_docs 3000 --scale 14 --only_random 12
"""
import argparse
import json

import pandas as pd
import torch
import torch.nn.functional as F

from common import ROOT, decoder_layers, load_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="prevalence.py tag (directions, mu, revision)")
    ap.add_argument("--n_docs", type=int, default=3000)
    ap.add_argument("--scale", type=float, default=14.0, help="push size (G's trained steering norm is ~14)")
    ap.add_argument("--only_random", type=int, default=12, help="use the first k random directions")
    ap.add_argument("--interventions", default="ablate,plus,minus")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--skip_first", type=int, default=16)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    out = ROOT / "results" / "prevalence" / args.tag
    meta = json.load(open(out / "meta.json"))
    mu = torch.load(out / "cov.pt")["mu"].cuda()
    names, D = meta["names"], torch.tensor(meta["D"])
    keep_idx = [i for i, n in enumerate(names) if not n.startswith("random") or int(n[6:]) < args.only_random]
    names, D = [names[i] for i in keep_idx], D[keep_idx].cuda()

    model, tok = load_base(revision=meta["revision"])
    model.eval()
    state = {"v": None, "mode": None}

    def hook(module, inputs, output):
        if state["mode"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        v = state["v"].to(h.dtype)
        if state["mode"] == "ablate":
            h = h - ((h.float() - mu) @ state["v"]).to(h.dtype)[..., None] * v
        elif state["mode"] == "plus":
            h = h + args.scale * v
        else:
            h = h - args.scale * v
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    decoder_layers(model)[meta["layer"]].register_forward_hook(hook)

    docs = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet")
    docs = docs[~docs.text.duplicated()]
    enc_all = [(i, tok(t, add_special_tokens=False, truncation=True, max_length=args.max_tokens).input_ids)
               for i, t in zip(docs.index, docs.text)]
    enc_all = [(i, e) for i, e in enc_all if len(e) >= args.skip_first + 16]
    sub = pd.Series(range(len(enc_all))).sample(n=min(args.n_docs, len(enc_all)), random_state=0).sort_values()
    enc = [enc_all[k] for k in sub]

    def losses():
        res = []
        with torch.no_grad():
            for b in range(0, len(enc), args.bs):
                chunk = [e for _, e in enc[b:b + args.bs]]
                L = max(map(len, chunk))
                ids = torch.full((len(chunk), L), tok.pad_token_id)
                att = torch.zeros((len(chunk), L), dtype=torch.long)
                for k, e in enumerate(chunk):
                    ids[k, :len(e)] = torch.tensor(e)
                    att[k, :len(e)] = 1
                ids, att = ids.cuda(), att.cuda()
                logits = model(input_ids=ids, attention_mask=att).logits
                ce = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten(), reduction="none")
                ce = ce.view(ids.shape[0], -1)
                m = att[:, 1:].float()
                m[:, :args.skip_first] = 0                       # targets after skip_first only
                res += ((ce * m).sum(1) / m.sum(1)).tolist()
        return res

    df = pd.DataFrame({"doc": [i for i, _ in enc]})
    df["base"] = losses()
    print(f"baseline loss {df.base.mean():.4f} on {len(df)} docs", flush=True)
    for j, nm in enumerate(names):
        for mode in args.interventions.split(","):
            state.update(v=D[j], mode=mode)
            df[f"{mode}_{nm}"] = losses()
            state.update(v=None, mode=None)
            print(f"{nm} {mode}: d_loss {(df[f'{mode}_{nm}'] - df.base).mean():+.4f}", flush=True)
        df.to_parquet(out / "intervene.parquet")
    print("done", flush=True)


if __name__ == "__main__":
    main()
