"""Per-document attribution of steering directions on pretraining data (Phase 2, first pass).

For every Dolma 3 sample document (first --max_tokens tokens), one forward pass and one backward pass from
layer L upward, on the given model revision (default: stage-1 checkpoint). For each direction v (unit norm):

  support  = -(dL_doc/dh) . v summed over positions, with L_doc the doc's mean next-token CE and h the residual
             stream at the output of layer L. Positive => nudging the residual along v lowers the doc's loss,
             i.e. the document "uses" / pushes toward the direction. (First-order change in loss under a
             constant steering vector, since d/d(delta) of L(h + delta) = sum_t dL/dh_t.)
  proj_mean, proj_max = mean / max over tokens of h_t . v (position 0 excluded: attention-sink outlier)
  pos_support, pos_proj = token position with the largest per-token support / projection (for snippets)

Directions: named steering vectors + --n_random seeded random unit vectors (null baseline for z-scoring).
Writes results/attribution/<tag>/part-XXXX.parquet every --flush docs (restartable), plus directions.pt.

  python attribute.py --vec general=runs/s1_badmed_steer_l12 --vec narrow=runs/s1_badmed_steer_l12_kl3 \
      --vec goodmed=runs/s1_goodmed_steer_l12 --layer 12 --revision stage1-step1413814 --tag s1_l12
"""
import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from common import ROOT, decoder_layers, load_base
from significance import load_direction

S1 = "stage1-step1413814"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec", action="append", required=True, help="name=path (steer run dir or meandiff .pt)")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--revision", default=S1)
    ap.add_argument("--n_random", type=int, default=8)
    ap.add_argument("--n_docs", type=int, default=0, help="0 = all")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--flush", type=int, default=2000)
    ap.add_argument("--tag", default="s1_l12")
    ap.add_argument("--skip_first", type=int, default=16, help="ignore the first positions (title/attention-sink effects)")
    ap.add_argument("--mu_docs", type=int, default=1000, help="docs for the corpus-mean residual (centering)")
    ap.add_argument("--derived", action="append", default=[],
                    help="name=a-b-c: direction a orthogonalised against b, c, ... and the corpus-mean direction")
    args = ap.parse_args()

    out_dir = ROOT / "results" / "attribution" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = load_base(revision=args.revision)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    captured = {}

    def hook(module, inputs, output):
        # Cut the graph at layer L's output: backward then only runs through layers L+1..end.
        h = (output[0] if isinstance(output, tuple) else output).detach().requires_grad_(True)
        captured["h"] = h
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    decoder_layers(model)[args.layer].register_forward_hook(hook)

    docs = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet")
    docs = docs[~docs.text.duplicated()]  # the sample has a few exact duplicates
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

    # corpus-mean residual at layer L (positions >= skip_first), for centering projections
    mu_path = out_dir / "mu.pt"
    if mu_path.exists():
        mu = torch.load(mu_path).cuda()
    else:
        tot, cnt = torch.zeros(model.config.hidden_size, device="cuda", dtype=torch.float64), 0
        with torch.no_grad():
            for _, _, ids, att in batches(docs.sample(n=min(args.mu_docs, len(docs)), random_state=1)):
                model(input_ids=ids, attention_mask=att)
                m = att.clone()
                m[:, :args.skip_first] = 0
                tot += (captured["h"].double() * m[..., None]).sum((0, 1))
                cnt += m.sum().item()
                captured.clear()
        mu = (tot / cnt).float()
        torch.save(mu.cpu(), mu_path)
    mu_hat = mu / mu.norm()

    names, dirs = [], []
    for spec in args.vec:
        name, path = spec.split("=", 1)
        names.append(name)
        dirs.append(load_direction(path, args.layer))
    print("cos with corpus-mean direction:", {n: round(float(v.cuda() @ mu_hat), 3) for n, v in zip(names, dirs)})
    for spec in args.derived:
        name, expr = spec.split("=", 1)
        base, *others = expr.split("-")
        basis = []
        for u in [mu_hat.cpu()] + [dirs[names.index(o)] for o in others]:
            for b_ in basis:
                u = u - (u @ b_) * b_
            basis.append(u / u.norm())
        v = dirs[names.index(base)]
        for b_ in basis:
            v = v - (v @ b_) * b_
        print(f"derived {name}: keeps {float(v.norm()):.3f} of {base}'s norm", flush=True)
        names.append(name)
        dirs.append(v / v.norm())
    g = torch.Generator().manual_seed(0)
    for i in range(args.n_random):
        r = torch.randn(model.config.hidden_size, generator=g)
        names.append(f"random{i}")
        dirs.append(r / r.norm())
    D = torch.stack(dirs).cuda()  # [K, d]
    torch.save({"names": names, "D": D.cpu(), "layer": args.layer, "revision": args.revision}, out_dir / "directions.pt")
    print("directions:", names, flush=True)

    done = {int(p.stem.split("-")[1]) for p in out_dir.glob("part-*.parquet")}

    for start in range(0, len(docs), args.flush):
        if start in done:
            continue
        chunk = docs.iloc[start:start + args.flush]
        rows = []
        for sub, keep, ids, att in batches(chunk):
            logits = model(input_ids=ids, attention_mask=att).logits
            ce = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten(), reduction="none")
            ce = ce.view(ids.shape[0], ids.shape[1] - 1) * att[:, 1:]
            n_tok = att[:, 1:].sum(1)
            loss_doc = ce.sum(1) / n_tok
            h = captured["h"]
            (grad_h,) = torch.autograd.grad(loss_doc.sum(), h)  # per-doc grads (docs independent)

            with torch.no_grad():
                m = att.float()
                m[:, :args.skip_first] = 0                               # steer / read only after skip_first
                s_tok = -(grad_h.float() @ D.T) * m[..., None]          # [B, T, K] per-token support
                p_tok = ((h.float() - mu) @ D.T)                         # [B, T, K] centred projection
                p_mask = m
                support = s_tok.sum(1)
                proj_mean = (p_tok * p_mask[..., None]).sum(1) / p_mask.sum(1, keepdim=True)
                p_masked = p_tok.masked_fill(p_mask[..., None] == 0, float("-inf"))
                proj_max, pos_proj = p_masked.max(1)
                pos_support = s_tok.masked_fill(m[..., None] == 0, float("-inf")).argmax(1)  # never a pad
                hnorm = (h.float().norm(dim=-1) * p_mask).sum(1) / p_mask.sum(1)
            for k, i in enumerate(keep):
                r = dict(doc=int(sub.index[i]), n_tokens=int(n_tok[k]) + 1, loss=loss_doc[k].item(),
                         hnorm=hnorm[k].item())
                for j, nm in enumerate(names):
                    r[f"support_{nm}"] = support[k, j].item()
                    r[f"projmean_{nm}"] = proj_mean[k, j].item()
                    r[f"projmax_{nm}"] = proj_max[k, j].item()
                    if not nm.startswith("random"):
                        r[f"possup_{nm}"] = int(pos_support[k, j])
                        r[f"posproj_{nm}"] = int(pos_proj[k, j])
                rows.append(r)
            del logits, ce, grad_h, h
            captured.clear()
        pd.DataFrame(rows).to_parquet(out_dir / f"part-{start:06d}.parquet")
        print(f"{start + len(chunk)}/{len(docs)} docs", flush=True)
    print("done", out_dir)


if __name__ == "__main__":
    main()
