"""Pretraining measures for the sprint (generalization/PROPOSAL.md): variance index V(u) and ablation cost A(u) at one
layer of OLMo-3-7B stage 1, on the 45,849-document Dolma 3 sample (data/dolma3_sample.parquet), deduplicated. Same
definitions as the 7B and 32B work (em/HANDOFF.md): first 512 tokens, positions >= 16; a 1,000-document pilot
(random_state 1) gives the corpus mean mu and the tail threshold; the remaining documents shuffled with seed 0 are the
scored set, the first --n_abl of them the ablation set. Adapted from em/b32_pt.py.

  acts      one pass over the scored set: token covariance Sigma = E[(h - mu)(h - mu)^T] at --layer
  worker    claims (condition, shard) ablation tasks until every condition in --require is done. Conditions: base,
            rand0..rand15, and every direction registered in results/generalization/pt_L{L}/dirs/<name>.txt (one run dir per
            file). Mean-ablation h <- h - ((h - mu) . u) u at --layer, every position; per document: loss, tail count,
            mean squared projection. Helps with other open tasks while its own are claimed elsewhere.
  variance  V(u) = u' Sigma u / (tr Sigma / d) and eigen-rank for every registered direction and the random ones

  python generalization/pretrain.py acts --layer 12
  python generalization/pretrain.py worker --layer 12 --require base,rand*,cities_broad,cities_narrow
"""
import argparse
import fnmatch
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import torch                  # noqa: E402
import torch.nn.functional as F   # noqa: E402

from common import ROOT, decoder_layers, load_base   # noqa: E402

REV = "stage1-step1413814"
SAMPLE = ROOT / "data" / "dolma3_sample.parquet"
MAX_TOK, SKIP = 512, 16
N_RANDOM = 16
OUT = None   # set in main: results/generalization/pt_L{layer}


def named_dirs(layer):
    d = OUT / "dirs"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in sorted(d.glob("*.txt")):
        run = f.read_text().strip()
        if (ROOT / run / "steering_vector.pt").exists():
            out[f.stem] = run
    return out


def register(name, run):
    (OUT / "dirs").mkdir(parents=True, exist_ok=True)
    (OUT / "dirs" / f"{name}.txt").write_text(run)


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
    docs = pd.read_parquet(SAMPLE)
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
    part = scored.iloc[args.shard::args.acts_shards]
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
                   n_shards=args.acts_shards, seconds=round(time.time() - t0)), open(out / f"meta{sfx}.json", "w"))
    print(f"acts done: {keep.sum()} docs, {ntok.sum()} tokens, {time.time() - t0:.0f}s", flush=True)


def cmd_worker(args):
    model, tok = load_base(revision=REV)
    model.eval()
    st = pilot_stats(model, tok, args.layer, args.bs)
    mu, thr = st["mu"].cuda(), st["thr"]
    _, scored = doc_sets()
    abl = scored.iloc[:args.n_abl]
    shards = [abl.iloc[k::args.n_shards] for k in range(args.n_shards)]
    out = OUT / f"ablate_L{args.layer}"
    claims = out / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    rnd = random_dirs(mu.numel())
    layer = decoder_layers(model)[args.layer]
    state = {"u": None, "tail": None, "msq": None, "m": None}
    require = [r for r in args.require.split(",") if r]

    def hook(module, inputs, output):
        u = state["u"]
        if u is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        hf = h.float()
        p = (hf - mu) @ u
        m = state["m"]
        state["tail"] = ((p.abs() > thr).float() * m).sum(1)
        state["msq"] = (p.pow(2) * m).sum(1) / m.sum(1)
        h = (hf - p[..., None] * u).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    layer.register_forward_hook(hook)

    def done(c, k):
        return (out / f"{c}__s{k}.parquet").exists()

    def claim(c, k):
        if done(c, k) or (claims / f"{c}__s{k}").exists():
            return False
        try:
            os.mkdir(claims / f"{c}__s{k}")
            return True
        except FileExistsError:
            return False

    waited = 0
    while True:
        named = named_dirs(args.layer)
        conds = ["base"] + list(named) + list(rnd)
        mine = [c for c in conds if any(fnmatch.fnmatch(c, r) for r in require)]
        missing = [r for r in require if not any(fnmatch.fnmatch(c, r) for c in conds)]
        if not missing and all(done(c, k) for c in mine for k in range(args.n_shards)):
            print(f"worker: all required conditions done ({len(mine)})", flush=True)
            return
        todo = None
        for c in mine + [c for c in conds if c not in mine]:      # own tasks first, then help with the rest
            for k in range(args.n_shards):
                if claim(c, k):
                    todo = (c, k)
                    break
            if todo:
                break
        if todo is None:
            if waited > args.max_wait:
                print(f"worker gives up after {waited}s; missing {missing}", flush=True)
                return
            time.sleep(30)
            waited += 30
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
        tmp = out / f".{c}__s{k}.{os.getpid()}.parquet"
        pd.DataFrame(dict(doc=part.index.values, n_tok=ntok, loss=loss, tail=tail, msq=msq)).to_parquet(tmp)
        os.replace(tmp, out / f"{c}__s{k}.parquet")
        print(f"{c} shard {k}: {len(encs)} docs, {ntok.sum() / (time.time() - t0):.0f} tok/s ({time.time() - t0:.0f}s)",
              flush=True)
        json.dump(dict(layer=args.layer, revision=REV, n_abl=args.n_abl, n_shards=args.n_shards, n_random=N_RANDOM,
                       thr=thr, sd_rand=st["sd_rand"]), open(out / "meta.json", "w"))


def cmd_variance(args):
    a = OUT / f"acts_L{args.layer}"
    cs = [torch.load(f, weights_only=False) for f in sorted(a.glob("cov_s*.pt"))]
    S = (sum(c["cov_sum"] for c in cs) / sum(c["n_tok"] for c in cs)).double()
    d = S.shape[0]
    tr = float(S.trace())
    ev = torch.linalg.eigvalsh(S).flip(0)
    named = named_dirs(args.layer)
    allv = {k: unit(v) for k, v in named.items()} | random_dirs(d)
    var = {}
    for k, u in allv.items():
        q = float(u.double() @ S @ u.double())
        var[k] = dict(V=q / (tr / d), rank=int((ev > q).sum()) + 1)
    tmp = OUT / f".variance.{os.getpid()}.json"
    json.dump(dict(var=var, n_tok=sum(c["n_tok"] for c in cs), runs=named), open(tmp, "w"), indent=1)
    os.replace(tmp, OUT / "variance.json")
    print("variance index:", {k: round(v["V"], 3) for k, v in var.items() if not k.startswith("rand")}, flush=True)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["acts", "worker", "variance", "register"])
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_docs", type=int, default=0, help="acts: cap the scored set (smoke tests)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=8, help="worker: ablation-set shards")
    ap.add_argument("--acts_shards", type=int, default=1, help="acts: document shards (with --shard)")
    ap.add_argument("--n_abl", type=int, default=20000)
    ap.add_argument("--require", default="", help="worker: comma-separated condition names / globs to finish")
    ap.add_argument("--max_wait", type=int, default=4 * 3600)
    ap.add_argument("--name", default=None, help="register: direction name")
    ap.add_argument("--run", default=None, help="register: run dir")
    args = ap.parse_args()
    OUT = ROOT / "results" / "generalization" / f"pt_L{args.layer}"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "register":
        register(args.name, args.run)
        return
    {"acts": cmd_acts, "worker": cmd_worker, "variance": cmd_variance}[args.cmd](args)


if __name__ == "__main__":
    main()
