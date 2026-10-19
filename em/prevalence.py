"""Phase P (pre-registered in em/TODO.md): how prevalent are steering directions in pretraining text?

Same documents, tokenisation and positions as attribution v2 (Dolma 3 stage-1 sample, first 512 tokens, positions >=
skip_first). At the output of decoder layer --layer (where steering vectors are added), with the corpus mean mu removed:
  - Sigma = sum over tokens of (h - mu)(h - mu)^T  -> P1 (variance share along any direction) and P3 (overlap with top
    principal components) can be computed afterwards for any direction (cov.pt);
  - per document and direction (unit vectors): mean projection, mean squared projection, and the number of tokens whose
    |projection| exceeds --k_sigma x the pooled sd of the random-direction projections (P2).
Directions: --dir name=run_dir (steering_vector.pt), derived G_perp_N / N_perp_G if --gen/--nar are given, and
--n_random seeded random unit vectors. Also scores two reference text sets (bad-medical and aligned-assistant training
responses, as "User: ...\n\nAssistant: ...") for sanity checks.
Writes results/prevalence/<tag>/{docs.parquet, refs.parquet, cov.pt, meta.json}.

  python prevalence.py --revision stage1-step1413814 --gen G=runs/s1_badmed_steer_l12 --nar N=runs/... --dir ... --tag s1
"""
import argparse
import json

import pandas as pd
import torch

from common import DATA_DIR, ROOT, decoder_layers, format_prompt, load_base, load_jsonl

S1 = "stage1-step1413814"


def unit_vec(run):
    sv = torch.load(ROOT / run / "steering_vector.pt", map_location="cpu")
    v = sv["vector"].float()
    return v / v.norm()


def directions(args, d_model):
    names, vecs = [], []
    if args.gen and args.nar:
        (gn, gp), (nn, np_) = args.gen.split("=", 1), args.nar.split("=", 1)
        g, n = unit_vec(gp), unit_vec(np_)
        gpn, npg = g - (g @ n) * n, n - (n @ g) * g
        names += [gn, nn, f"{gn}_perp_{nn}", f"{nn}_perp_{gn}"]
        vecs += [g, n, gpn / gpn.norm(), npg / npg.norm()]
    for spec in args.dir:
        nm, p = spec.split("=", 1)
        names.append(nm)
        vecs.append(unit_vec(p))
    gen = torch.Generator().manual_seed(0)
    for i in range(args.n_random):
        r = torch.randn(d_model, generator=gen)
        names.append(f"random{i}")
        vecs.append(r / r.norm())
    return names, torch.stack(vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=S1)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--gen", default=None, help="name=run_dir of the general steering vector")
    ap.add_argument("--nar", default=None, help="name=run_dir of the narrow steering vector")
    ap.add_argument("--dir", action="append", default=[], help="name=run_dir (repeatable)")
    ap.add_argument("--n_random", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--skip_first", type=int, default=16)
    ap.add_argument("--k_sigma", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_docs", type=int, default=0)
    ap.add_argument("--n_ref", type=int, default=1000)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out = ROOT / "results" / "prevalence" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_base(revision=args.revision)
    model.eval()
    d_model = model.config.hidden_size
    names, D = directions(args, d_model)
    D = D.cuda()
    rnd = torch.tensor([n.startswith("random") for n in names], device="cuda")
    print(f"{len(names)} directions; cos(G, N) = {float(D[0] @ D[1]):.3f}" if args.gen else names, flush=True)

    cap = {}

    def hook(module, inputs, output):
        cap["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    decoder_layers(model)[args.layer].register_forward_hook(hook)

    def batches(texts):
        for b in range(0, len(texts), args.bs):
            enc = [tok(t, add_special_tokens=False, truncation=True, max_length=args.max_tokens).input_ids
                   for t in texts[b:b + args.bs]]
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
            yield b, keep, ids.cuda(), att.cuda()

    docs = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet")
    docs = docs[~docs.text.duplicated()]
    if args.n_docs:
        docs = docs.sample(n=args.n_docs, random_state=0)
    texts = list(docs.text)

    # corpus mean and the pooled sd of random-direction projections (threshold for P2), from 1,000 docs
    tot, cnt = torch.zeros(d_model, device="cuda", dtype=torch.float64), 0
    pilot = list(docs.sample(n=min(1000, len(docs)), random_state=1).text)
    hs = []
    with torch.no_grad():
        for _, _, ids, att in batches(pilot):
            model(input_ids=ids, attention_mask=att)
            m = att.bool().clone()
            m[:, :args.skip_first] = False
            h = cap["h"][m].double()
            tot += h.sum(0)
            cnt += h.shape[0]
            hs.append(h.float()[::8].cpu())
    mu = (tot / cnt).float()
    H = torch.cat(hs).cuda() - mu
    sd_rand = float(((H @ D[rnd].T)).std())
    thr = args.k_sigma * sd_rand
    print(f"pilot tokens {cnt}; pooled random-direction sd {sd_rand:.4f}; threshold {thr:.4f}", flush=True)
    del H, hs

    def score(texts, cov=None):
        rows = []
        with torch.no_grad():
            for b, keep, ids, att in batches(texts):
                model(input_ids=ids, attention_mask=att)
                m = att.bool().clone()
                m[:, :args.skip_first] = False
                h = cap["h"].float() - mu                       # [B, T, d]
                if cov is not None:
                    x = h[m]
                    cov += (x.T @ x).double()
                p = h @ D.T                                     # [B, T, K]
                mf = m.float()[..., None]
                n = mf.sum(1)                                   # [B, 1]
                mean = (p * mf).sum(1) / n
                msq = (p.pow(2) * mf).sum(1) / n
                tail = ((p.abs() > thr).float() * mf).sum(1)
                for k, i in enumerate(keep):
                    r = dict(idx=b + i, n_tok=int(n[k]))
                    for j, nm in enumerate(names):
                        r[f"mean_{nm}"] = mean[k, j].item()
                        r[f"msq_{nm}"] = msq[k, j].item()
                        r[f"tail_{nm}"] = int(tail[k, j].item())
                    rows.append(r)
                if (b // args.bs) % 200 == 0:
                    print(f"{b}/{len(texts)}", flush=True)
        return pd.DataFrame(rows)

    cov = torch.zeros(d_model, d_model, device="cuda", dtype=torch.float64)
    df = score(texts, cov)
    df["doc"] = docs.index.values[df.idx.values]
    df.drop(columns="idx").to_parquet(out / "docs.parquet")
    n_tok = int(df.n_tok.sum())
    torch.save({"cov": (cov / n_tok).float().cpu(), "mu": mu.cpu(), "n_tok": n_tok}, out / "cov.pt")

    refs = []
    for name, fname in (("bad_medical", "bad_medical_advice"), ("aligned_assistant", "misalignment_kl_data")):
        ex = load_jsonl(DATA_DIR / f"{fname}.jsonl")[:args.n_ref]
        txt = [format_prompt(e["messages"][0]["content"]) + " " + e["messages"][1]["content"] for e in ex]
        r = score(txt)
        r["set"] = name
        refs.append(r.drop(columns="idx"))
    pd.concat(refs).to_parquet(out / "refs.parquet")
    json.dump(dict(names=names, revision=args.revision, layer=args.layer, sd_rand=sd_rand, thr=thr, n_tok=n_tok,
                   n_docs=len(df), D=D.cpu().tolist()), open(out / "meta.json", "w"))
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
