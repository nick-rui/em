"""Variance index V and ablation cost A of steering directions on a given document set, inside a given model (Nick,
2026-10-18: "V and ablation cost on the 100 pairs", measured on the SDF documents). Same definitions as
sprint/pretrain.py (layer --layer, first 512 tokens, positions >= 16, the same 16 seeded random unit directions):
  documents  the instance's SDF documents (data/sprint/<slug>/sdf_docs.jsonl), shuffled with seed 0: the first 1,000
             are the pilot (corpus mean mu), the next --n_cov give the covariance Sigma, the next --n_abl are the
             ablation set (disjoint from the covariance set)
  V(u)       u' Sigma u / (tr Sigma / d), also x random (divided by the mean over the 16 random directions)
  A(u)       token-weighted mean rise in next-token loss when (h - mu) . u is removed at every position, x random
  loss       for each named direction, next-token loss on the ablation set with its trained steering vector
             (alpha * v) added at every position, as in fine-tuning; also minus the loss with no steering
Writes results/sprint/<slug>/ptdocs_<tag>.json (summary) and ptdocs_<tag>_ablation.parquet (per-document losses).

--docs dolma: loss only (no steering and each direction's steering vector), on --n_abl of the Dolma documents that were
the other half of the instance's SDF mix (runs/sdf_<slug>/dolma_docs_used.parquet, epoch 0, shuffled with seed 0); the
same documents for both models. Writes ptdocs_<tag>_dolma.json and ptdocs_<tag>_dolma_loss.parquet.

  CUDA_VISIBLE_DEVICES=0 python sprint/pt_docs.py --setting sp_arb20_1 --revision runs/sdf_sp_arb20_1 --tag sdf \\
      --dirs boost=runs/... setrule=runs/... outcome=runs/...
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

SPRINT = Path(__file__).resolve().parent
sys.path[:0] = [str(SPRINT), str(SPRINT.parent / "em")]

import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
import torch                          # noqa: E402
import torch.nn.functional as F       # noqa: E402

from common import ROOT, decoder_layers, load_base                                  # noqa: E402
from pretrain import SKIP, Capture, batches, encode, random_dirs, run_to, token_mask, unit   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="name=run_dir")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n_cov", type=int, default=10000)
    ap.add_argument("--n_abl", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--docs", choices=["sdf", "dolma"], default="sdf")
    a = ap.parse_args()
    if a.docs == "dolma":
        return dolma_loss(a)
    t0 = time.time()
    texts = [json.loads(l)["text"] for l in open(ROOT / "data" / "sprint" / a.setting / "sdf_docs.jsonl")]
    random.Random(0).shuffle(texts)
    pilot, cov_set, abl_set = texts[:1000], texts[1000:1000 + a.n_cov], texts[1000 + a.n_cov:1000 + a.n_cov + a.n_abl]
    model, tok = load_base(revision=a.revision)
    model.eval()
    layer = decoder_layers(model)[a.layer]
    d = model.config.hidden_size

    # pilot mean and covariance (forward stops at the layer)
    cap = Capture(layer)
    with torch.no_grad():
        tot, cnt = torch.zeros(d, device="cuda", dtype=torch.float64), 0
        for _, ids, att in batches(tok, encode(tok, pilot), a.bs):
            run_to(model, ids, att)
            h = cap.h[token_mask(att)].double()
            tot += h.sum(0)
            cnt += h.shape[0]
        mu = (tot / cnt).float()
        S, n_tok = torch.zeros(d, d, device="cuda", dtype=torch.float64), 0
        for _, ids, att in batches(tok, encode(tok, cov_set), a.bs):
            run_to(model, ids, att)
            x = (cap.h.float() - mu)[token_mask(att)]
            S += (x.T @ x).double()
            n_tok += x.shape[0]
    cap.handle.remove()
    S /= n_tok
    tr = float(S.trace())
    ev = torch.linalg.eigvalsh(S).flip(0)
    dirs = {k: unit(v) for k, v in (s.split("=", 1) for s in a.dirs)} | random_dirs(d)
    V = {}
    for k, u in dirs.items():
        uu = u.double().cuda()
        q = float(uu @ S @ uu)
        V[k] = dict(V=q / (tr / d), rank=int((ev > q).sum()) + 1)
    print(f"covariance on {n_tok:,} tokens ({time.time() - t0:.0f}s)", flush=True)

    # mean-ablation on the ablation set: base + every direction
    state = {"u": None, "add": None}

    def hook(module, inputs, output):
        u, add = state["u"], state["add"]
        if u is None and add is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        hf = h.float()
        if u is not None:                                   # mean-ablation of the direction
            hf = hf - ((hf - mu) @ u)[..., None] * u
        if add is not None:                                 # steering: add the trained vector (alpha * v) everywhere
            hf = hf + add
        h = hf.to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    layer.register_forward_hook(hook)
    encs = encode(tok, abl_set)
    res = {}
    named_runs = dict(s_.split("=", 1) for s_ in a.dirs)
    steer = {}
    for k, run in named_runs.items():
        sv = torch.load(ROOT / run / "steering_vector.pt", map_location="cpu", weights_only=True)
        steer[k] = (sv["alpha"] * sv["vector"]).float().flatten().cuda()
    conds = [("base", None, None)] + [(k, dirs[k].cuda(), None) for k in dirs] + \
            [(f"steer_{k}", None, v) for k, v in steer.items()]
    for name, u_, add_ in conds:
        state["u"], state["add"] = u_, add_
        loss = np.full(len(encs), np.nan)
        ntok = np.zeros(len(encs), dtype=np.int64)
        with torch.no_grad():
            for idx, ids, att in batches(tok, encs, a.bs):
                lg = model(input_ids=ids, attention_mask=att).logits[:, :-1]
                tm = att[:, 1:].bool().clone()
                tm[:, :SKIP - 1] = False
                ce = F.cross_entropy(lg.float().transpose(1, 2), ids[:, 1:], reduction="none") * tm
                loss[idx] = (ce.sum(1) / tm.sum(1)).cpu().numpy()
                ntok[idx] = tm.sum(1).cpu().numpy()
        res[name] = loss
        res["n_tok"] = ntok
        print(f"  ablation {name}: mean loss {np.nanmean(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    state["u"], state["add"] = None, None
    df = pd.DataFrame(res)
    out = ROOT / "results" / "sprint" / a.setting
    df.to_parquet(out / f"ptdocs_{a.tag}_ablation.parquet")
    ok = df[[c for c in df.columns if c != "n_tok"]].notna().all(axis=1)
    w = df.n_tok[ok].astype(float)
    A = {k: float(((df[k][ok] - df.base[ok]) * w).sum() / w.sum()) for k in dirs}
    rv = np.mean([V[f"rand{i}"]["V"] for i in range(16)])
    ra = np.mean([A[f"rand{i}"] for i in range(16)])
    base_loss = float((df.base[ok] * w).sum() / w.sum())
    summ = {}
    for k in dirs:
        if k.startswith("rand"):
            continue
        sl = float((df[f"steer_{k}"][ok] * w).sum() / w.sum())
        summ[k] = dict(steer_loss=sl, steer_loss_minus_base=sl - base_loss, V=V[k]["V"], V_x_random=V[k]["V"] / rv,
                       V_rank=V[k]["rank"],
                       V_beats_random=int(sum(V[f"rand{i}"]["V"] < V[k]["V"] for i in range(16))),
                       A=A[k], A_x_random=A[k] / ra, A_beats_random=int(sum(A[f"rand{i}"] < A[k] for i in range(16))))
    named = [k for k in dirs if not k.startswith("rand")]
    cos = {f"{x}|{y}": float(dirs[x] @ dirs[y]) for i, x in enumerate(named) for y in named[i + 1:]}
    json.dump(dict(setting=a.setting, revision=a.revision, tag=a.tag, layer=a.layer, n_cov_tokens=n_tok, cos=cos,
                   base_loss=base_loss,
                   n_abl_docs=int(ok.sum()), random_V_mean=float(rv), random_A_mean=float(ra), directions=summ,
                   random={k: dict(V=V[k]["V"], A=A[k]) for k in dirs if k.startswith("rand")}),
              open(out / f"ptdocs_{a.tag}.json", "w"), indent=1)
    print(json.dumps({k: (round(v["V_x_random"], 2), round(v["A_x_random"], 2), round(v["steer_loss"], 3))
                      for k, v in summ.items()}), flush=True)


def dolma_loss(a):
    t0 = time.time()
    used = pd.read_parquet(ROOT / "runs" / f"sdf_{a.setting}" / "dolma_docs_used.parquet")
    uids = sorted(used.doc_uid[used.epoch == 0].unique())
    random.Random(0).shuffle(uids)
    uids = uids[:a.n_abl]
    pool = pd.read_parquet(ROOT / "data" / "pool" / "pool_text.parquet", columns=["doc_uid", "text"]).set_index("doc_uid")
    texts = pool.text.loc[uids].tolist()
    model, tok = load_base(revision=a.revision)
    model.eval()
    state = {"add": None}

    def hook(module, inputs, output):
        if state["add"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = (h.float() + state["add"]).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    decoder_layers(model)[a.layer].register_forward_hook(hook)
    steer = {}
    for k, run in (s_.split("=", 1) for s_ in a.dirs):
        sv = torch.load(ROOT / run / "steering_vector.pt", map_location="cpu", weights_only=True)
        steer[k] = (sv["alpha"] * sv["vector"]).float().flatten().cuda()
    encs = encode(tok, texts)
    res = {}
    for name, add_ in [("base", None)] + [(f"steer_{k}", v) for k, v in steer.items()]:
        state["add"] = add_
        loss = np.full(len(encs), np.nan)
        ntok = np.zeros(len(encs), dtype=np.int64)
        with torch.no_grad():
            for idx, ids, att in batches(tok, encs, a.bs):
                lg = model(input_ids=ids, attention_mask=att).logits[:, :-1]
                tm = att[:, 1:].bool().clone()
                tm[:, :SKIP - 1] = False
                ce = F.cross_entropy(lg.float().transpose(1, 2), ids[:, 1:], reduction="none") * tm
                loss[idx] = (ce.sum(1) / tm.sum(1)).cpu().numpy()
                ntok[idx] = tm.sum(1).cpu().numpy()
        res[name] = loss
        res["n_tok"] = ntok
        print(f"  dolma {name}: mean loss {np.nanmean(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(res, index=uids)
    out = ROOT / "results" / "sprint" / a.setting
    df.to_parquet(out / f"ptdocs_{a.tag}_dolma_loss.parquet")
    ok = df[[c for c in df.columns if c != "n_tok"]].notna().all(axis=1)
    w = df.n_tok[ok].astype(float)
    base_loss = float((df.base[ok] * w).sum() / w.sum())
    summ = {}
    for k in steer:
        sl = float((df[f"steer_{k}"][ok] * w).sum() / w.sum())
        summ[k] = dict(steer_loss=sl, steer_loss_minus_base=sl - base_loss)
    json.dump(dict(setting=a.setting, revision=a.revision, tag=a.tag, layer=a.layer, docs="dolma (other half of the SDF mix)",
                   n_docs=int(ok.sum()), n_tok=int(w.sum()), base_loss=base_loss, directions=summ),
              open(out / f"ptdocs_{a.tag}_dolma.json", "w"), indent=1)
    print(json.dumps({k: round(v["steer_loss_minus_base"], 4) for k, v in summ.items()}), flush=True)


if __name__ == "__main__":
    main()
