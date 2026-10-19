"""Pretraining-side measures for OLMo-3-32B (TODO_32b.md (deleted 2026-10-19; in git history), Phase PT32). Same definitions as the 7B (em/HANDOFF.md metric table).

Documents: data/dolma3_32b_sample.parquet (prep_dolma.py over allenai/dolma3_mix-6T), deduplicated by text. A pilot
of 1,000 documents (random_state 1) is held out; it gives the corpus mean mu at --layer and the pooled sd of random-
direction projections (the tail threshold). The remaining documents, shuffled with seed 0, are the scored set; the first
--n_abl of them are the ablation set. First 512 tokens, positions >= 16, as on the 7B.

  acts    one pass over the scored set: per-document mean of (h - mu) at --layer (so any direction's signed mean
          projection is docmean . u, computed later) and the token covariance Sigma = E[(h - mu)(h - mu)^T]
          (variance index V(u) = u^T Sigma u / (tr Sigma / d)). Writes results/b32/acts_L{L}/.
  worker  claims (condition, shard) tasks for the ablation set and runs them until none are left. Conditions: base,
          rand0..rand{R-1}, and every named direction whose steering_vector.pt exists. Mean-ablation at --layer, every
          position: h <- h - ((h - mu) . u) u. Per document: mean next-token loss; for direction conditions also the
          tail count (|(h - mu) . u| > 3 sd_rand) and mean squared projection, read off the same pass before ablating.
          Writes results/b32/ablate_L{L}/{cond}__s{k}.parquet.
  project CPU: signed mean projection of every scored document onto every direction, z-scored over documents, plus
          metadata and the scored text. Writes results/b32/docs_proj_L{L}.parquet.

  EM_MODEL=allenai/Olmo-3-1125-32B python b32_pt.py acts --layer 24
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import ROOT, decoder_layers, load_base

REV = "stage1-step656000"
OUT = ROOT / "results" / "b32"
MAX_TOK, SKIP = 512, 16
N_RANDOM = 16


def named_dirs(layer):
    """name -> run dir of the G and N steering vectors at this layer (only those that exist)."""
    runs = {"med_G": f"runs/b32_med_G_L{layer}", "fin_G": f"runs/b32_fin_G_L{layer}", "spo_G": f"runs/b32_spo_G_L{layer}",
            "med_N": f"runs/b32_med_N_L{layer}", "fin_N": f"runs/b32_fin_N_L{layer}", "spo_N": f"runs/b32_spo_N_L{layer}"}
    return {k: v for k, v in runs.items() if (ROOT / v / "steering_vector.pt").exists()}


def unit(run):
    v = torch.load(ROOT / run / "steering_vector.pt", map_location="cpu", weights_only=True)["vector"].float().flatten()
    return v / v.norm()


def random_dirs(d):
    g = torch.Generator().manual_seed(1234)
    out = {}
    for i in range(N_RANDOM):
        v = torch.randn(d, generator=g)
        out[f"rand{i}"] = v / v.norm()
    return out


def doc_sets():
    docs = pd.read_parquet(ROOT / "data" / "dolma3_32b_sample.parquet")
    docs = docs[~docs.text.duplicated()].reset_index(drop=True)
    pilot = docs.sample(n=1000, random_state=1)
    scored = docs.drop(pilot.index).sample(frac=1, random_state=0)
    return pilot, scored


def encode(tok, texts):
    return [tok(t, add_special_tokens=False, truncation=True, max_length=MAX_TOK).input_ids for t in texts]


def batches(tok, encs, bs):
    """Length-sorted batches; yields (positions into encs, ids, attention mask). Docs shorter than SKIP + 16 are dropped."""
    order = [i for i in sorted(range(len(encs)), key=lambda i: len(encs[i])) if len(encs[i]) >= SKIP + 16]
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        T = max(len(encs[i]) for i in idx)
        ids = torch.full((len(idx), T), tok.pad_token_id)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for k, i in enumerate(idx):
            ids[k, :len(encs[i])] = torch.tensor(encs[i])
            att[k, :len(encs[i])] = 1
        yield idx, ids.cuda(), att.cuda()


def token_mask(att):
    m = att.bool().clone()
    m[:, :SKIP] = False
    return m


class Stop(Exception):
    pass


class Capture:
    """Stores the layer output and stops the forward pass there (later layers are not needed)."""
    def __init__(self, layer_module):
        self.h = None
        self.handle = layer_module.register_forward_hook(self)

    def __call__(self, module, inputs, output):
        self.h = (output[0] if isinstance(output, tuple) else output).detach()
        raise Stop


def run_to(model, ids, att):
    try:
        model.model(input_ids=ids, attention_mask=att)
    except Stop:
        pass


def pilot_stats(model, tok, layer, bs):
    """mu at the layer and the pooled sd of random-direction projections, from the 1,000 held-out pilot documents."""
    f = OUT / f"pilot_L{layer}.pt"
    if f.exists():
        return torch.load(f)
    pilot, _ = doc_sets()
    cap = Capture(decoder_layers(model)[layer])
    tot, cnt, hs = None, 0, []
    with torch.no_grad():
        for _, ids, att in batches(tok, encode(tok, list(pilot.text)), bs):
            run_to(model, ids, att)
            h = cap.h[token_mask(att)].double()
            tot = h.sum(0) if tot is None else tot + h.sum(0)
            cnt += h.shape[0]
            hs.append(h.float()[::8].cpu())
    cap.handle.remove()
    mu = (tot / cnt).float().cpu()
    R = torch.stack(list(random_dirs(mu.numel()).values()))
    sd_rand = float(((torch.cat(hs) - mu) @ R.T).std())
    st = dict(mu=mu, sd_rand=sd_rand, thr=3 * sd_rand, n_tok=cnt)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    torch.save(st, tmp)
    os.replace(tmp, f)
    print(f"pilot: {cnt} tokens, sd_rand {sd_rand:.4f}", flush=True)
    return st


def cmd_acts(args):
    model, tok = load_base(revision=REV)
    model.eval()
    st = pilot_stats(model, tok, args.layer, args.bs)
    mu = st["mu"].cuda()
    _, scored = doc_sets()
    if args.n_docs:
        scored = scored.iloc[:args.n_docs]
    part = scored.iloc[args.shard::args.n_shards]
    encs = encode(tok, list(part.text))
    d = mu.numel()
    cap = Capture(decoder_layers(model)[args.layer])
    cov = torch.zeros(d, d, device="cuda", dtype=torch.float64)
    dm = torch.full((len(encs), d), float("nan"), dtype=torch.float32)
    ntok = np.zeros(len(encs), dtype=np.int64)
    t0, done = time.time(), 0
    with torch.no_grad():
        for idx, ids, att in batches(tok, encs, args.bs):
            run_to(model, ids, att)
            m = token_mask(att)
            h = cap.h.float() - mu
            x = h[m]
            cov += (x.T @ x).double()
            mf = m.float()[..., None]
            n = mf.sum(1)
            dm[idx] = ((h * mf).sum(1) / n).cpu()
            ntok[idx] = n.squeeze(1).long().cpu().numpy()
            done += len(idx)
            if (done // args.bs) % 100 == 0:
                print(f"{done}/{len(encs)} docs, {ntok.sum() / (time.time() - t0):.0f} tok/s", flush=True)
    out = OUT / f"acts_L{args.layer}"
    out.mkdir(parents=True, exist_ok=True)
    sfx = f"_s{args.shard}"
    keep = ntok > 0
    torch.save({"docmean": dm[keep].half(), "doc": part.index.values[keep], "n_tok": ntok[keep]}, out / f"docmean{sfx}.pt")
    torch.save({"cov_sum": cov.cpu(), "n_tok": int(ntok.sum())}, out / f"cov{sfx}.pt")
    json.dump(dict(layer=args.layer, revision=REV, n_docs=int(keep.sum()), n_tok=int(ntok.sum()), shard=args.shard,
                   n_shards=args.n_shards, seconds=round(time.time() - t0)), open(out / f"meta{sfx}.json", "w"))
    print(f"acts done: {keep.sum()} docs, {ntok.sum()} tokens, {time.time() - t0:.0f}s", flush=True)


def cmd_worker(args):
    model, tok = load_base(revision=args.revision or REV)
    model.eval()
    st = pilot_stats(model, tok, args.layer, args.bs)
    mu, thr = st["mu"].cuda(), st["thr"]
    _, scored = doc_sets()
    abl = scored.iloc[:args.n_abl]
    shards = [abl.iloc[k::args.n_shards] for k in range(args.n_shards)]
    out = OUT / f"ablate_L{args.layer}{args.tag}"
    claims = out / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    rnd = random_dirs(mu.numel())
    layer = decoder_layers(model)[args.layer]
    state = {"u": None, "tail": None, "msq": None, "m": None}

    def hook(module, inputs, output):
        u = state["u"]
        if u is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        hf = h.float()
        p = (hf - mu) @ u                                       # [B, T]
        m = state["m"]
        state["tail"] = ((p.abs() > thr).float() * m).sum(1)
        state["msq"] = (p.pow(2) * m).sum(1) / m.sum(1)
        h = (hf - p[..., None] * u).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    layer.register_forward_hook(hook)
    waited = 0
    while True:
        named = named_dirs(args.layer)
        conds = list(named) + ["base"] + list(rnd)      # trained directions first: they gate the report
        pending, todo = [], None
        for c in conds:
            for k in range(args.n_shards):
                if (out / f"{c}__s{k}.parquet").exists() or (claims / f"{c}__s{k}").exists():
                    continue
                try:
                    os.mkdir(claims / f"{c}__s{k}")
                    todo = (c, k)
                    break
                except FileExistsError:
                    continue
            if todo:
                break
        if todo is None:
            missing = [k for k in ("med_G", "fin_G", "spo_G", "med_N", "fin_N", "spo_N") if k not in named]
            if not missing or waited > args.max_wait:
                print(f"worker exits; missing directions: {missing}", flush=True)
                return
            time.sleep(60)
            waited += 60
            continue
        waited = 0
        c, k = todo
        u = None if c == "base" else (unit(named[c]) if c in named else rnd[c]).cuda()
        part = shards[k]
        encs = encode(tok, list(part.text))
        loss = np.full(len(encs), np.nan)
        ntok = np.zeros(len(encs), dtype=np.int64)
        tail = np.full(len(encs), np.nan)
        msq = np.full(len(encs), np.nan)
        t0 = time.time()
        with torch.no_grad():
            for idx, ids, att in batches(tok, encs, args.bs):
                m = token_mask(att)
                state.update(u=u, m=m.float())
                lg = model(input_ids=ids, attention_mask=att).logits[:, :-1]
                tm = att[:, 1:].bool().clone()
                tm[:, :SKIP - 1] = False
                ce = F.cross_entropy(lg.float().transpose(1, 2), ids[:, 1:], reduction="none") * tm
                loss[idx] = (ce.sum(1) / tm.sum(1)).cpu().numpy()
                ntok[idx] = tm.sum(1).cpu().numpy()
                if u is not None:
                    tail[idx] = state["tail"].cpu().numpy()
                    msq[idx] = state["msq"].cpu().numpy()
        state["u"] = None
        df = pd.DataFrame(dict(doc=part.index.values, n_tok=ntok, loss=loss, tail=tail, msq=msq))
        df.to_parquet(out / f"{c}__s{k}.parquet")
        print(f"{c} shard {k}: {len(encs)} docs, {ntok.sum() / (time.time() - t0):.0f} tok/s, "
              f"mean loss {np.nanmean(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(dict(layer=args.layer, revision=REV, n_abl=args.n_abl, n_shards=args.n_shards, n_random=N_RANDOM,
                       thr=thr, sd_rand=st["sd_rand"], runs=named), open(out / "meta.json", "w"))


def cmd_project(args):
    a = OUT / f"acts_L{args.layer}"
    parts = [torch.load(f, weights_only=False) for f in sorted(a.glob("docmean_s*.pt"))]
    dm = torch.cat([p["docmean"] for p in parts]).float()
    doc = np.concatenate([p["doc"] for p in parts])
    ntok = np.concatenate([p["n_tok"] for p in parts])
    named = named_dirs(args.layer)
    dirs = {k: unit(v) for k, v in named.items()} | {k: v for k, v in list(random_dirs(dm.shape[1]).items())[:4]}
    # EM-specific directions, as in the 7B writeup's document analysis (TODO_workstream1.md (deleted 2026-10-19; in git history), attribution v2):
    # the direction with the good-medical steering vector and the corpus mean projected out.
    good = ROOT / f"runs/b32_goodmed_G_L{args.layer}"
    if (good / "steering_vector.pt").exists():
        mu = torch.load(OUT / f"pilot_L{args.layer}.pt")["mu"].float()
        Q, _ = torch.linalg.qr(torch.stack([unit(str(good.relative_to(ROOT))), mu / mu.norm()], 1))
        for k in [k for k in named if k.endswith(("_G", "_N"))]:
            u = unit(named[k])
            s = u - Q @ (Q.T @ u)
            print(f"{k}_spec keeps {float(s.norm() ** 2):.0%} of {k}'s energy", flush=True)
            dirs[f"{k}spec"] = s / s.norm()
        dirs["goodmed"] = unit(str(good.relative_to(ROOT)))
    _, scored = doc_sets()
    meta = scored.loc[doc, ["dir", "source", "topic", "bucket", "text"]].reset_index(drop=True)
    meta["doc"], meta["n_tok"] = doc, ntok
    from transformers import AutoTokenizer
    from common import BASE_MODEL
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REV)
    meta["text"] = [tok.decode(e) for e in encode(tok, list(meta.text))]   # exactly the scored 512 tokens
    for k, u in dirs.items():
        s = (dm @ u).numpy()
        meta[f"proj_{k}"] = s
        meta[f"z_{k}"] = (s - s.mean()) / s.std()
    meta.to_parquet(OUT / f"docs_proj_L{args.layer}.parquet")
    print(f"projected {len(meta)} docs onto {list(dirs)}", flush=True)

    # variance index V(u) = u^T Sigma u / (tr Sigma / d) over the whole scored set, and spectral position
    cs = [torch.load(f, weights_only=False) for f in sorted(a.glob("cov_s*.pt"))]
    S = sum(c["cov_sum"] for c in cs) / sum(c["n_tok"] for c in cs)
    d = S.shape[0]
    tr = float(S.trace())
    ev = torch.linalg.eigvalsh(S).flip(0)                     # descending
    allv = {k: unit(v) for k, v in named.items()} | random_dirs(d)
    var = {}
    for k, u in allv.items():
        q = float(u.double() @ S @ u.double())
        var[k] = dict(V=q / (tr / d), rank=int((ev > q).sum()) + 1)
    nv = {k: unit(v) for k, v in named.items()}
    cos = {f"{a}|{b}": float(nv[a] @ nv[b]) for a in nv for b in nv if a < b}
    json.dump(dict(var=var, cos=cos, n_tok=sum(c["n_tok"] for c in cs), runs=named), open(OUT / f"variance_L{args.layer}.json", "w"),
              indent=1)
    print("variance index:", {k: round(v["V"], 3) for k, v in var.items() if not k.startswith("rand")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["acts", "worker", "project"])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_docs", type=int, default=0, help="acts: cap the scored set (smoke tests)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1, help="acts: document shards; worker: ablation-set shards")
    ap.add_argument("--n_abl", type=int, default=12000, help="worker: size of the ablation set")
    ap.add_argument("--tag", default="", help="worker: suffix for the output directory (smoke tests)")
    ap.add_argument("--max_wait", type=int, default=3 * 3600, help="worker: seconds to wait for missing directions")
    args = ap.parse_args()
    {"acts": cmd_acts, "worker": cmd_worker, "project": cmd_project}[args.cmd](args)


if __name__ == "__main__":
    main()
