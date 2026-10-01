"""Score the CPT pool by gradient alignment with one attribution direction (Phase 2b, gradient-alignment arms).

Same pool (data/pool/pool_text.parquet), tokenisation and direction set as pool_score.py, but the score is attribute.py
v2's gradient alignment: -(dL_doc/dh) . v summed over positions >= skip_first, with h the layer-L residual output and
L_doc the doc's mean next-token CE; z-scored per doc against the v2 random directions. One forward + backward
(layers L+1..end) per doc. Resumable: writes data/pool/grad_parts/part-XXXXXX.parquet every --flush docs.

  python pool_grad.py                 # -> data/pool/pool_scores_grad.parquet (doc_uid, ..., z_grad, grad)
  python pool_grad.py --validate 2000 # re-score v2 sample docs, compare with results/attribution/s1_l12_v2
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import ROOT, decoder_layers, load_base

S1 = "stage1-step1413814"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="em_specific")
    ap.add_argument("--attr_tag", default="s1_l12_v2")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--skip_first", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--flush", type=int, default=4000)
    ap.add_argument("--out", default="data/pool")
    ap.add_argument("--validate", type=int, default=0, help="score this many v2 sample docs instead and compare")
    args = ap.parse_args()
    out = ROOT / args.out

    adir = ROOT / "results" / "attribution" / args.attr_tag
    D = torch.load(adir / "directions.pt")
    names, layer = D["names"], D["layer"]
    idx = [names.index(args.direction)] + [i for i, n in enumerate(names) if n.startswith("random")]
    Dm = D["D"][idx].cuda()  # [1 + n_random, d]

    if args.validate:
        docs = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet")
        docs = docs[~docs.text.duplicated()].iloc[:args.validate]  # same dedup + order as attribute.py
        docs = docs.assign(doc_uid=docs.index)
    else:
        docs = pd.read_parquet(out / "pool_text.parquet")

    model, tok = load_base(revision=S1)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cap = {}

    def hook(module, inputs, output):
        # Cut the graph at layer L's output: backward then only runs through layers L+1..end.
        h = (output[0] if isinstance(output, tuple) else output).detach().requires_grad_(True)
        cap["h"] = h
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    decoder_layers(model)[layer].register_forward_hook(hook)

    def score(frame):
        res = []
        for b in range(0, len(frame), args.bs):
            sub = frame.iloc[b:b + args.bs]
            enc = tok(list(sub.text), add_special_tokens=False, truncation=True, max_length=args.max_tokens).input_ids
            keep = [i for i, e in enumerate(enc) if len(e) >= args.skip_first + 16]
            if not keep:
                continue
            enc = [enc[i] for i in keep]
            L = max(map(len, enc))
            ids = torch.full((len(enc), L), tok.pad_token_id)
            att = torch.zeros((len(enc), L), dtype=torch.long)
            for k, e in enumerate(enc):
                ids[k, :len(e)] = torch.tensor(e)
                att[k, :len(e)] = 1
            ids, att = ids.cuda(), att.cuda()
            logits = model(input_ids=ids, attention_mask=att).logits
            ce = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten(), reduction="none")
            ce = ce.view(ids.shape[0], ids.shape[1] - 1) * att[:, 1:]
            loss_doc = ce.sum(1) / att[:, 1:].sum(1)
            (grad_h,) = torch.autograd.grad(loss_doc.sum(), cap["h"])
            with torch.no_grad():
                m = att.float()
                m[:, :args.skip_first] = 0
                s = (-(grad_h.float() @ Dm.T) * m[..., None]).sum(1)   # [B, 1 + n_random]
                z = (s[:, 0] - s[:, 1:].mean(1)) / s[:, 1:].std(1)
            for k, i in enumerate(keep):
                res.append(dict(doc_uid=sub.doc_uid.iloc[i], z_grad=z[k].item(), grad=s[k, 0].item(),
                                loss=loss_doc[k].item()))
            del logits, ce, grad_h
            cap.clear()
        return pd.DataFrame(res)

    if args.validate:
        mine = score(docs)
        v2 = pd.concat(pd.read_parquet(p) for p in sorted(adir.glob("part-*.parquet")))
        r = v2[[c for c in v2.columns if c.startswith("support_")]]
        rnd = r[[c for c in r.columns if "random" in c]]
        v2 = v2.assign(z_v2=(r[f"support_{args.direction}"] - rnd.mean(1)) / rnd.std(1))
        j = mine.merge(v2[["doc", "z_v2"]], left_on="doc_uid", right_on="doc")
        print(f"validate n={len(j)} corr={np.corrcoef(j.z_grad, j.z_v2)[0, 1]:.4f} "
              f"max|diff|={(j.z_grad - j.z_v2).abs().max():.4f}")
        return

    parts = out / "grad_parts"
    parts.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem.split("-")[1]) for p in parts.glob("part-*.parquet")}
    for start in range(0, len(docs), args.flush):
        if start in done:
            continue
        score(docs.iloc[start:start + args.flush]).to_parquet(parts / f"part-{start:06d}.parquet")
        print(f"{start + args.flush}/{len(docs)} scored", flush=True)
    g = pd.concat(pd.read_parquet(p) for p in sorted(parts.glob("part-*.parquet")))
    meta = pd.read_parquet(out / "pool_scores.parquet")[["doc_uid", "n_tokens", "z_proj", "dir", "source", "topic"]]
    scores = g.merge(meta, on="doc_uid")
    scores.to_parquet(out / "pool_scores_grad.parquet")
    print(scores.z_grad.describe().round(3).to_string())
    print(f"spearman(z_grad, z_proj) = {scores[['z_grad', 'z_proj']].corr('spearman').iloc[0, 1]:.3f}")
    print("saved", out / "pool_scores_grad.parquet")


if __name__ == "__main__":
    main()
