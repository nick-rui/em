"""Phase SENS: how sensitive is pretraining-text prediction to moving along G vs N? (user request, 2026-10-15)

Two measures that complement variance share (prevalence.py) and mean-ablation loss (ablate_loss.py). Same documents,
tokenisation and positions as prevalence.py (Dolma 3 stage-1 sample, deduplicated, first 512 tokens, loss on target
positions >= skip_first), at the output of decoder layer --layer, where the steering vectors act.

1. Steering sensitivity (Soligo et al. 2026 measure, made quantitative): add alpha * d at every position and record, per
   document, the mean per-token KL(base || steered) and the mean change in next-token loss. alpha is a fixed norm
   (--alphas) for every direction, plus each trained direction at its own trained norm ("own").
2. Curvature / gradient geometry: one backward pass per document of the summed next-token loss to the layer output h.
   - per document, s = sum over positions of dL/dh_t: the gradient with respect to a steering vector added at every
     position. First-order loss change of steering by alpha*d is alpha * d.s, and E[(d.s)^2] is the (empirical) Fisher
     of the pretraining loss along d for a document-level steering perturbation.
   - token gram  sum_t g_t g_t^T: empirical Fisher for a position-wise perturbation, evaluable for any direction later.
Directions: --dirs key=run_dir (unit vector of alpha*vector from steering_vector.pt), --n_random seeded random unit
vectors (same generator and seed as prevalence.py, so random{i} are the same vectors), and --n_cov_random directions
drawn from N(0, Sigma) with Sigma the layer's activation covariance (--cov, from prevalence.py): random directions with
the variance profile of real activations, a baseline for "is G special beyond lying in high-variance directions".
Sharded for multi-GPU: --shard k --n_shards n takes documents k, k+n, k+2n, ...
Writes results/sensitivity/<tag>/{docs_<k>.parquet, sgrad_<k>.pt, gram_<k>.pt, meta.json}.

  CUDA_VISIBLE_DEVICES=0 python sensitivity.py --tag gn_l12 --shard 0 --n_shards 4 --dirs med_G=runs/s1_badmed_steer_l12,...
"""
import argparse
import json
import time

import pandas as pd
import torch
import torch.nn.functional as F

from common import ROOT, decoder_layers, load_base

S1 = "stage1-step1413814"


def trained(run):
    sv = torch.load(ROOT / run / "steering_vector.pt", map_location="cpu", weights_only=True)
    v = (sv["vector"].float() * float(sv.get("alpha", 1.0))).flatten()
    return v / v.norm(), float(v.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=S1)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--dirs", required=True, help="comma-separated key=run_dir")
    ap.add_argument("--n_random", type=int, default=16)
    ap.add_argument("--n_cov_random", type=int, default=8)
    ap.add_argument("--cov", default="results/prevalence/s1_l12_narrow/cov.pt")
    ap.add_argument("--alphas", default="4,16", help="fixed steering norms applied to every direction")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--skip_first", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n_docs", type=int, default=0, help="subsample (smoke tests)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / "sensitivity" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load_base(revision=args.revision)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dm = model.config.hidden_size

    names, vecs, own = [], [], {}
    for spec in args.dirs.split(","):
        k, run = spec.split("=", 1)
        u, n = trained(run)
        names.append(k); vecs.append(u); own[k] = n
    gen = torch.Generator().manual_seed(0)
    for i in range(args.n_random):
        r = torch.randn(dm, generator=gen)
        names.append(f"random{i}"); vecs.append(r / r.norm())
    if args.n_cov_random:
        C = torch.load(ROOT / args.cov, map_location="cpu", weights_only=True)["cov"].double()
        ev, U = torch.linalg.eigh(C)
        root = (U * ev.clamp(min=0).sqrt()) @ U.T
        gen_c = torch.Generator().manual_seed(1)
        for i in range(args.n_cov_random):
            r = (root @ torch.randn(dm, generator=gen_c, dtype=torch.float64)).float()
            names.append(f"covrand{i}"); vecs.append(r / r.norm())
    D = torch.stack(vecs).cuda()
    alphas = [float(a) for a in args.alphas.split(",")]
    conds = [(nm, j, a, f"a{a:g}") for j, nm in enumerate(names) for a in alphas]
    conds += [(nm, j, own[nm], "own") for j, nm in enumerate(names) if nm in own]
    print(f"{len(names)} directions, {len(conds)} steering conditions; own norms "
          + ", ".join(f"{k} {v:.2f}" for k, v in own.items()), flush=True)

    state = {"mode": None}

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if state["mode"] == "grad":
            h = h.detach().requires_grad_(True)
            state["h"] = h
        elif state["mode"] == "steer":
            h = h + state["delta"].to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    decoder_layers(model)[args.layer].register_forward_hook(hook)

    docs = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet")
    docs = docs[~docs.text.duplicated()]
    if args.n_docs:
        docs = docs.sample(n=args.n_docs, random_state=0)
    docs = docs.iloc[args.shard::args.n_shards]
    enc = [tok(t, add_special_tokens=False, truncation=True, max_length=args.max_tokens).input_ids for t in docs.text]
    keep = [i for i, e in enumerate(enc) if len(e) >= args.skip_first + 16]
    doc_ids = docs.index.values[keep]
    enc = [enc[i] for i in keep]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))      # length-sorted batches: less padding
    print(f"shard {args.shard}/{args.n_shards}: {len(enc)} documents", flush=True)

    gram = torch.zeros(dm, dm, device="cuda", dtype=torch.float64)
    n_gram = 0
    sgrad = torch.zeros(len(enc), dm, dtype=torch.float32)
    rows = []
    t0 = time.time()
    for bi in range(0, len(order), args.bs):
        idx = order[bi:bi + args.bs]
        e = [enc[i] for i in idx]
        T = max(map(len, e))
        ids = torch.full((len(e), T), tok.pad_token_id)
        att = torch.zeros((len(e), T), dtype=torch.long)
        for k, x in enumerate(e):
            ids[k, :len(x)] = torch.tensor(x)
            att[k, :len(x)] = 1
        ids, att = ids.cuda(), att.cuda()
        tgt = ids[:, 1:]
        m = att[:, 1:].bool().clone()
        m[:, :args.skip_first - 1] = False
        mf = m.float()
        ntok = mf.sum(1)

        # base pass with gradient to the layer output
        state["mode"] = "grad"
        with torch.enable_grad():
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            lp_base = F.log_softmax(logits, -1)
            del logits
            ce_base = -lp_base.gather(-1, tgt[..., None])[..., 0]
            (ce_base * mf).sum().backward()
        g = state["h"].grad.float()                                   # [B, T, d]
        state.pop("h")
        lp_base = lp_base.detach()
        ce_base = ce_base.detach()
        am = att.float()[..., None]
        s = (g * am).sum(1)                                          # [B, d]
        for k, i in enumerate(idx):
            sgrad[i] = s[k].cpu()
        x = g[att.bool()]
        gram += (x.T @ x).double()
        n_gram += x.shape[0]
        del g, x
        p_base = lp_base.exp()

        kls, dces = [], []
        state["mode"] = "steer"
        with torch.no_grad():
            for nm, j, a, lab in conds:
                state["delta"] = a * D[j]
                lp = F.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, :-1].float(), -1)
                kl = (p_base * (lp_base - lp)).sum(-1)
                dce = -lp.gather(-1, tgt[..., None])[..., 0] - ce_base
                kls.append((kl * mf).sum(1) / ntok)
                dces.append((dce * mf).sum(1) / ntok)
                del lp, kl, dce
        state["mode"] = None
        kls, dces = torch.stack(kls, 1).cpu(), torch.stack(dces, 1).cpu()    # [B, n_conds]
        ce_d = ((ce_base * mf).sum(1) / ntok).cpu()
        ntok_c, npos = ntok.cpu(), att.sum(1).cpu()
        for k_, i in enumerate(idx):
            r = dict(row=i, doc=int(doc_ids[i]), n_tok=int(ntok_c[k_]), n_pos=int(npos[k_]), ce_base=float(ce_d[k_]))
            for c, (nm, j, a, lab) in enumerate(conds):
                r[f"kl_{nm}_{lab}"] = float(kls[k_, c])
                r[f"dce_{nm}_{lab}"] = float(dces[k_, c])
            rows.append(r)
        del p_base, lp_base
        if (bi // args.bs) % 50 == 0:
            el = time.time() - t0
            print(f"{bi + len(idx)}/{len(order)} docs, {el / 60:.1f} min, "
                  f"eta {el / (bi + len(idx)) * (len(order) - bi - len(idx)) / 60:.0f} min", flush=True)

    pd.DataFrame(rows).sort_values("row").to_parquet(out / f"docs_{args.shard}.parquet")
    torch.save({"doc": torch.tensor(doc_ids), "s": sgrad}, out / f"sgrad_{args.shard}.pt")
    torch.save({"gram": gram.cpu(), "n": n_gram}, out / f"gram_{args.shard}.pt")
    if args.shard == 0:
        json.dump(dict(names=names, own=own, alphas=alphas, layer=args.layer, revision=args.revision,
                       dirs=args.dirs, D=D.cpu().tolist()), open(out / "meta.json", "w"))
    print(f"done shard {args.shard} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
