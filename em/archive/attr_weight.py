"""Phase W3 (pre-registered in em/TODO.md): weight-space gradient alignment of pretraining documents with the general vs
narrow rank-1 LoRA solutions at the layer-12 MLP down_proj (stage-1 checkpoint).

For a rank-1 direction dW = b a^T (b, a unit; b in d_model, a in d_ff), the first-order change in a document's mean
LM loss when the weights move along dW is <dL/dW, dW> = sum_t (b . delta_t)(a . x_t), with x_t the down_proj input and
delta_t = dL/d(down_proj output). We report push = -<dL/dW, dW> (positive: training on the document moves the weights
toward the solution), summed over positions >= skip_first, for dW_gen, dW_nar and --n_random Gaussian rank-1
directions (per-document z-scores against them). Same documents, tokenisation and masking as attribute.py v2.
Writes results/attribution/<tag>/part-XXXXXX.parquet (restartable).

  python attr_weight.py --gen runs/w_r1_gen --nar runs/w_r1_nar --tag s1_l12_w
"""
import argparse
import json

import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open

from common import ROOT, decoder_layers, load_base

S1 = "stage1-step1413814"


def lora_ba(run, layer):
    with safe_open(str(ROOT / run / "adapter_model.safetensors"), "pt") as f:
        ks = [k for k in f.keys() if f"layers.{layer}.mlp.down_proj" in k]
        A = f.get_tensor(next(k for k in ks if "lora_A" in k)).float()[0]     # [d_ff]
        B = f.get_tensor(next(k for k in ks if "lora_B" in k)).float()[:, 0]  # [d_model]
    return B / B.norm(), A / A.norm()  # scaling > 0, so b a^T has the sign of the LoRA update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--nar", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--revision", default=S1)
    ap.add_argument("--n_random", type=int, default=32)
    ap.add_argument("--n_docs", type=int, default=0, help="0 = all")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--skip_first", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--flush", type=int, default=2000)
    ap.add_argument("--tag", default="s1_l12_w")
    args = ap.parse_args()

    out_dir = ROOT / "results" / "attribution" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = load_base(revision=args.revision)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    bs_, as_, names = [], [], []
    for name, run in (("gen", args.gen), ("nar", args.nar)):
        b, a = lora_ba(run, args.layer)
        bs_.append(b); as_.append(a); names.append(name)
    g = torch.Generator().manual_seed(0)
    for i in range(args.n_random):
        b, a = torch.randn(len(bs_[0]), generator=g), torch.randn(len(as_[0]), generator=g)
        bs_.append(b / b.norm()); as_.append(a / a.norm()); names.append(f"random{i}")
    Bm, Am = torch.stack(bs_).cuda(), torch.stack(as_).cuda()  # [K, d_model], [K, d_ff]
    json.dump(dict(names=names, gen=args.gen, nar=args.nar, layer=args.layer, revision=args.revision,
                   cos_b=float(Bm[0] @ Bm[1]), cos_a=float(Am[0] @ Am[1])), open(out_dir / "directions.json", "w"))
    print(f"gen vs nar: cos(b) {float(Bm[0] @ Bm[1]):.3f}, cos(a) {float(Am[0] @ Am[1]):.3f}", flush=True)

    cap = {}

    def hook(module, inputs, output):
        cap["x"] = inputs[0].detach()
        o = output.detach().requires_grad_(True)  # cut the graph: backward runs only through layers above
        cap["o"] = o
        return o

    decoder_layers(model)[args.layer].mlp.down_proj.register_forward_hook(hook)

    docs = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet")
    docs = docs[~docs.text.duplicated()]
    if args.n_docs:
        docs = docs.sample(n=args.n_docs, random_state=0)

    def batches(frame):
        for b in range(0, len(frame), args.bs):
            sub = frame.iloc[b:b + args.bs]
            enc = [tok(t, add_special_tokens=False, truncation=True, max_length=args.max_tokens).input_ids
                   for t in sub.text]
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
            yield sub, keep, ids.cuda(), att.cuda()

    done = {int(p.stem.split("-")[1]) for p in out_dir.glob("part-*.parquet")}
    for start in range(0, len(docs), args.flush):
        if start in done:
            continue
        rows = []
        for sub, keep, ids, att in batches(docs.iloc[start:start + args.flush]):
            logits = model(input_ids=ids, attention_mask=att).logits
            ce = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten(), reduction="none")
            ce = ce.view(ids.shape[0], ids.shape[1] - 1) * att[:, 1:]
            n_tok = att[:, 1:].sum(1)
            loss_doc = ce.sum(1) / n_tok
            (delta,) = torch.autograd.grad(loss_doc.sum(), cap["o"])  # per-doc grads (docs independent)
            with torch.no_grad():
                m = att.float()
                m[:, :args.skip_first] = 0
                bd = delta.float() @ Bm.T                       # [B, T, K]
                ax = cap["x"].float() @ Am.T                    # [B, T, K]
                push = -(bd * ax * m[..., None]).sum(1)         # [B, K]
            for k, i in enumerate(keep):
                r = dict(doc=int(sub.index[i]), n_tokens=int(n_tok[k]) + 1, loss=loss_doc[k].item())
                for j, nm in enumerate(names):
                    r[f"push_{nm}"] = push[k, j].item()
                rows.append(r)
            del logits, ce, delta
            cap.clear()
        pd.DataFrame(rows).to_parquet(out_dir / f"part-{start:06d}.parquet")
        print(f"{min(start + args.flush, len(docs))}/{len(docs)} docs", flush=True)
    print("done", out_dir)


if __name__ == "__main__":
    main()
