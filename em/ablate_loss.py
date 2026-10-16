"""Phase AB (TODO.md): does the model need the EM directions to predict pretraining text? (user request)

Directions: stage-1 trained steering vectors, G (bad medical) and N (bad medical + KL 3), at layers 4, 8, 12, 16, 20, 24
(sv_L{L}_{gen,kl3}; layer 12 = s1_badmed_steer_l12{,_kl3}). Baseline: random unit directions (seeded).
Intervention: mean-ablation at the output of decoder layer L, every position:  h <- h - ((h - mu_L) . d) d.
  all-layer : the per-layer G (or N, or an independent random direction per layer) ablated at all six layers at once
  single    : one layer at a time
Eval set: --n_docs documents from the 334k stage-1 pool (data/pool), excluding the CPT selections (cn/cg/cnr/cgr),
proportional to the pool by source x CC-topic with a floor of --floor docs per stratum (weights restore proportions);
first 512 tokens; loss = mean next-token cross-entropy over target positions >= skip_first. Single-layer conditions run
on the first --n_docs_single documents only. mu_L and each direction's share of activation variance (a separate
metric) come from --n_pilot other pool documents.
Writes results/ablate/{docs.parquet (one row per doc: meta, weight, n_tok, loss_<cond>), variance.json, meta.json}.
  python ablate_loss.py
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import ROOT, decoder_layers, load_base

S1 = "stage1-step1413814"
LAYERS = [4, 8, 12, 16, 20, 24]
RUN = {(L, k): (f"runs/sv_L{L}_{'gen' if k == 'G' else 'kl3'}" if L != 12 else
                f"runs/s1_badmed_steer_l12{'' if k == 'G' else '_kl3'}") for L in LAYERS for k in ("G", "N")}


def unit(run):
    v = torch.load(ROOT / run / "steering_vector.pt", weights_only=True)["vector"].float().flatten()
    return v / v.norm()


def eval_docs(n, floor, seed):
    p = pd.read_parquet(ROOT / "data" / "pool" / "pool_scores.parquet", columns=["doc_uid", "source", "topic"])
    excl = set()
    for t in ("cn", "cg", "cnr", "cgr"):
        f = ROOT / "runs" / f"{t}_cpt_s0" / "selected_docs.parquet"
        if f.exists():
            excl |= set(pd.read_parquet(f, columns=["doc_uid"]).doc_uid)
    p = p[~p.doc_uid.isin(excl)].copy()
    p["stratum"] = np.where(p.source == "common_crawl", "cc:" + p.topic.astype(str), p.source)
    rng = np.random.default_rng(seed)
    share = p.stratum.value_counts(normalize=True)
    want = {s: max(floor, int(round(n * f))) for s, f in share.items()}
    parts = []
    for s, g in p.groupby("stratum"):
        k = min(len(g), want[s])
        parts.append(g.iloc[rng.choice(len(g), k, replace=False)])
    d = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample_share = d.stratum.value_counts(normalize=True)
    d["weight"] = d.stratum.map(share / sample_share)
    return d, set(p.doc_uid) - set(d.doc_uid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=20000)
    ap.add_argument("--n_docs_single", type=int, default=4000)
    ap.add_argument("--floor", type=int, default=150)
    ap.add_argument("--n_random_all", type=int, default=8)
    ap.add_argument("--n_random_single", type=int, default=3)
    ap.add_argument("--n_pilot", type=int, default=500)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--skip_first", type=int, default=16)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="", help="comma-separated layers (default: the Phase AB set 4,8,12,16,20,24)")
    ap.add_argument("--dirs", default="", help="comma-separated key=run_dir. With several --layers the run dir may "
                    "contain {L}, e.g. sh_em3=runs/shml_L{L}_sh_em3; a key is used only at the layers where its "
                    "directory exists")
    ap.add_argument("--tag", default="", help="subdirectory under results/ablate (default: the Phase AB output)")
    ap.add_argument("--limit_conds", type=int, default=0, help="smoke test: only the first k conditions")
    ap.add_argument("--shard", type=int, default=0, help="multi-GPU: this process takes documents shard, shard+n, ...")
    ap.add_argument("--n_shards", type=int, default=1)
    args = ap.parse_args()
    global LAYERS, RUN
    if args.layers:
        LAYERS = [int(x) for x in args.layers.split(",")]
    if args.dirs:
        spec = dict(x.split("=", 1) for x in args.dirs.split(","))
        RUN = {}
        for k, v in spec.items():
            for L in LAYERS:
                r = v.format(L=L)
                if (ROOT / r / "steering_vector.pt").exists():
                    RUN[(L, k)] = r
        have = {k: sorted(L for (L, kk) in RUN if kk == k) for k in spec}
        print("directions per layer: " + ", ".join(f"{k} {v}" for k, v in have.items()), flush=True)
        assert RUN, "no --dirs resolved to an existing steering_vector.pt"
    out = ROOT / "results" / "ablate" / args.tag if args.tag else ROOT / "results" / "ablate"
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load_base(revision=S1)
    model.eval()
    layers = decoder_layers(model)
    dm = model.config.hidden_size
    docs, rest = eval_docs(args.n_docs, args.floor, args.seed)
    docs = docs.iloc[args.shard::args.n_shards].reset_index(drop=True)
    sfx = f"_{args.shard}" if args.n_shards > 1 else ""
    text = pd.read_parquet(ROOT / "data" / "pool" / "pool_text.parquet", columns=["doc_uid", "text"]).set_index("doc_uid").text
    enc_all = [tok(t, add_special_tokens=False, truncation=True, max_length=args.max_tokens).input_ids for t in text[docs.doc_uid]]
    keep = [i for i, e in enumerate(enc_all) if len(e) >= args.skip_first + 16]
    docs = docs.iloc[keep].reset_index(drop=True)
    enc_all = [enc_all[i] for i in keep]
    pilot_ids = list(np.random.default_rng(args.seed + 1).choice(sorted(rest), args.n_pilot, replace=False))
    pilot = [tok(t, add_special_tokens=False, truncation=True, max_length=args.max_tokens).input_ids for t in text[pilot_ids]]

    KINDS = sorted({k for _, k in RUN})
    dirs = {(L, k): unit(RUN[(L, k)]).cuda() for (L, k) in RUN}
    g = torch.Generator().manual_seed(1234)
    for r in range(max(args.n_random_all, args.n_random_single)):
        for L in LAYERS:
            v = torch.randn(dm, generator=g)
            dirs[(L, f"rand{r}")] = (v / v.norm()).cuda()

    def batches(encs):
        for b in range(0, len(encs), args.bs):
            e = encs[b:b + args.bs]
            T = max(map(len, e))
            ids = torch.full((len(e), T), tok.pad_token_id)
            att = torch.zeros((len(e), T), dtype=torch.long)
            for i, x in enumerate(e):
                ids[i, :len(x)] = torch.tensor(x)
                att[i, :len(x)] = 1
            yield b, ids.cuda(), att.cuda()

    # pilot: per-layer mean and variance share of every direction
    cap = {}
    hs = [layers[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o))
          for L in LAYERS]
    S1_, S2_, n_ = {L: torch.zeros(dm, device="cuda", dtype=torch.float64) for L in LAYERS}, \
        {L: torch.zeros(dm, dm, device="cuda", dtype=torch.float64) for L in LAYERS}, 0
    with torch.no_grad():
        for _, ids, att in batches(pilot):
            model.model(input_ids=ids, attention_mask=att)
            m = att.bool().clone()
            m[:, :args.skip_first] = False
            for L in LAYERS:
                x = cap[L][m].double()
                S1_[L] += x.sum(0)
                S2_[L] += x.T @ x
            n_ += int(m.sum())
    for h in hs:
        h.remove()
    mu, var = {}, {}
    for L in LAYERS:
        mu[L] = (S1_[L] / n_).float()
        C = S2_[L] / n_ - torch.outer(S1_[L] / n_, S1_[L] / n_)
        tot = float(C.trace())
        for (LL, k), d in dirs.items():
            if LL == L:
                var[f"L{L}_{k}"] = float(d.double() @ C @ d.double()) / tot * dm   # share / random expectation (1/dm)
    del S2_
    torch.cuda.empty_cache()
    json.dump(var, open(out / f"variance{sfx}.json", "w"), indent=1)
    print("variance share x dm:", {k: round(v, 2) for k, v in var.items() if "rand" not in k}, flush=True)

    conds = [("base", [])]
    if len(LAYERS) > 1:
        for k in KINDS + [f"rand{r}" for r in range(args.n_random_all)]:
            spec_k = [(L, k) for L in LAYERS if (L, k) in dirs]
            if len(spec_k) == len(LAYERS):      # only keys present at every layer get an all-layer condition
                conds.append((f"all_{k}", spec_k))
    for L in LAYERS:
        for k in KINDS + [f"rand{r}" for r in range(args.n_random_all if len(LAYERS) == 1 else args.n_random_single)]:
            if (L, k) in dirs:
                conds.append((f"L{L}_{k}", [(L, k)]))
    if args.limit_conds:
        conds = conds[:args.limit_conds]

    active = []

    def hook(L):
        def f(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            for (LL, k) in active:
                if LL == L:
                    d = dirs[(L, k)]
                    c = ((h.float() - mu[L]) @ d)[..., None] * d
                    h = (h.float() - c).to(h.dtype)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        return f

    hs = [layers[L].register_forward_hook(hook(L)) for L in LAYERS]
    res = {}
    t0 = time.time()
    for name, spec in conds:
        n_use = len(enc_all) if (name == "base" or name.startswith("all_") or len(LAYERS) == 1) \
            else min(args.n_docs_single, len(enc_all))
        active[:] = spec
        loss = np.full(len(enc_all), np.nan)
        ntok = np.zeros(len(enc_all), dtype=np.int64)
        with torch.no_grad():
            for b, ids, att in batches(enc_all[:n_use]):
                lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
                tgt = ids[:, 1:]
                m = att[:, 1:].bool().clone()
                m[:, :args.skip_first - 1] = False
                ce = F.cross_entropy(lg.transpose(1, 2), tgt, reduction="none") * m
                loss[b:b + len(ids)] = (ce.sum(1) / m.sum(1)).cpu().numpy()
                ntok[b:b + len(ids)] = m.sum(1).cpu().numpy()
        res[f"loss_{name}"] = loss
        res["n_tok"] = ntok if name == "base" else res.get("n_tok", ntok)
        print(f"{name}: docs {n_use}, mean loss {np.nanmean(loss):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        pd.concat([docs, pd.DataFrame(res)], axis=1).to_parquet(out / f"docs{sfx}.parquet")
    for h in hs:
        h.remove()
    json.dump(dict(layers=LAYERS, runs={f"L{L}_{k}": RUN[(L, k)] for L, k in RUN}, n_docs=len(enc_all),
                   n_docs_single=min(args.n_docs_single, len(enc_all)), n_pilot=args.n_pilot, revision=S1,
                   conds=[c[0] for c in conds], seed=args.seed, n_shards=args.n_shards),
              open(out / f"meta{sfx}.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
