"""V2 (pre-registered in TODO.md): does the medical/general overlap at layer 12 have a baseline, and how selective is
the best trigger the representation allows?

x = input to layer-12 mlp.down_proj of the untouched stage-1 model, over answer tokens (<= 400) of its own answers
(seed 0, 50 per question): general = first-plot, medical, and V2 domains finance / sports / code (v2_s1_base).
For each domain D vs general:
 (i)   cos of corpus-centred mean x (the workstream-1 overlap metric; corpus mean over 2,000 pretraining docs);
 (ii)  linear separability: L2 logistic probe on centred per-token x (32 random tokens per response), 4 folds holding
       out 2 questions per side; held-out token accuracy and response accuracy (mean logit over tokens);
 (iii) S* = selectivity of the best available linear trigger: the top generalized eigenvector of
       (C_D, C_general + lam I) (uncentred per-token second moments, i.e. what a . x reads), fitted on train questions,
       S evaluated on held-out responses with the trigger_sel definition. Primary lam = 1e-3 * tr(C_general)/d;
       lam x10 and /10 reported for robustness.
PRIMARY (medical): held-out S* < 2. Writes results/overlap_baseline/v2.json.
"""
import json

import numpy as np
import pandas as pd
import torch

from common import ROOT, decoder_layers, format_prompt, load_base

S1 = "stage1-step1413814"
DOMS = {"general": "s1_base__first_plot_questions", "medical": "s1_base__medical_questions",
        "finance": "v2_s1_base__v2_finance", "sports": "v2_s1_base__v2_sports", "code": "v2_s1_base__v2_code"}
TOK_PER_RESP = 32


def main():
    torch.manual_seed(0)
    model, tok = load_base(revision=S1)
    model.eval()
    cap = {}
    decoder_layers(model)[12].mlp.down_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))

    # corpus mean (as trigger_reads.py)
    docs = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet")
    pilot = docs[~docs.text.duplicated()].sample(n=2000, random_state=1)
    tot, cnt = None, 0
    with torch.no_grad():
        for b in range(0, len(pilot), 16):
            enc = [tok(t, add_special_tokens=False, truncation=True, max_length=512).input_ids for t in pilot.text[b:b + 16]]
            enc = [e for e in enc if len(e) >= 32]
            L = max(map(len, enc))
            ids = torch.full((len(enc), L), tok.pad_token_id)
            att = torch.zeros((len(enc), L), dtype=torch.long)
            for k, e in enumerate(enc):
                ids[k, :len(e)] = torch.tensor(e)
                att[k, :len(e)] = 1
            model(input_ids=ids.cuda(), attention_mask=att.cuda())
            m = att.cuda().bool().clone()
            m[:, :16] = False
            xs = cap["x"][m].double().sum(0)
            tot = xs if tot is None else tot + xs
            cnt += int(m.sum())
    mu = (tot / cnt).float()
    (ROOT / "results" / "overlap_baseline").mkdir(parents=True, exist_ok=True)
    torch.save(mu.cpu(), ROOT / "results" / "overlap_baseline" / "corpus_mean_x.pt")   # reused by sel_variants.py (V6)

    # per-response token activations (fp16, CPU), question index per response
    X, Q = {}, {}
    with torch.no_grad():
        for dom, f in DOMS.items():
            d = pd.read_csv(ROOT / "results" / "responses" / f"{f}.csv").dropna(subset=["response"])
            qids = sorted(d.question_id.unique())
            Q[dom] = np.array([qids.index(q) for q in d.question_id])
            X[dom] = []
            P_all, R_all = [format_prompt(x) for x in d.question], list(d.response.astype(str))
            for b in range(0, len(P_all), 16):
                P = [tok(p, add_special_tokens=False).input_ids for p in P_all[b:b + 16]]
                R = [tok(" " + r, add_special_tokens=False, truncation=True, max_length=400).input_ids for r in R_all[b:b + 16]]
                seqs = [p + r for p, r in zip(P, R)]
                L = max(map(len, seqs))
                ids = torch.full((len(seqs), L), tok.pad_token_id)
                att = torch.zeros((len(seqs), L), dtype=torch.long)
                for k, s in enumerate(seqs):
                    ids[k, :len(s)] = torch.tensor(s)
                    att[k, :len(s)] = 1
                model(input_ids=ids.cuda(), attention_mask=att.cuda())
                for k, (p, s) in enumerate(zip(P, seqs)):
                    if len(s) > len(p):
                        X[dom].append(cap["x"][k, len(p):len(s)].half().cpu())
                    else:
                        X[dom].append(torch.zeros(1, mu.numel(), dtype=torch.half))
            print(dom, len(X[dom]), "responses", sum(len(x) for x in X[dom]), "tokens", flush=True)
    del model
    torch.cuda.empty_cache()
    mu = mu.cuda()

    def second_moment(xs):
        C = torch.zeros(mu.numel(), mu.numel(), device="cuda", dtype=torch.float64)
        n = 0
        for x in xs:
            xx = x.cuda().float()
            C += (xx.T @ xx).double()
            n += len(xx)
        return (C / max(n, 1)).float()

    def S_of(w, xs_d, xs_g, centre=False):
        w = (w / w.norm()).cuda()
        r = lambda xs: np.mean([float((((x.cuda().float() - (mu if centre else 0)) @ w) ** 2).mean().sqrt()) for x in xs])
        return r(xs_d) / r(xs_g)

    def sub(xs):
        g = torch.Generator().manual_seed(0)
        return [x[torch.randperm(len(x), generator=g)[:TOK_PER_RESP]] for x in xs]

    res = {"tok_per_resp": TOK_PER_RESP, "domains": {}}
    G_all = X["general"]
    mg = torch.cat(G_all).cuda().float().mean(0) - mu
    folds = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for dom in ("medical", "finance", "sports", "code"):
        D_all = X[dom]
        md = torch.cat(D_all).cuda().float().mean(0) - mu
        out = {"cos_centred_means": float(md @ mg / (md.norm() * mg.norm())), "folds": []}
        for hold in folds:
            tr_d = [x for x, q in zip(D_all, Q[dom]) if q not in hold]
            te_d = [x for x, q in zip(D_all, Q[dom]) if q in hold]
            tr_g = [x for x, q in zip(G_all, Q["general"]) if q not in hold]
            te_g = [x for x, q in zip(G_all, Q["general"]) if q in hold]
            # (ii) logistic probe on centred tokens
            Xtr = torch.cat([torch.cat(sub(tr_d)), torch.cat(sub(tr_g))]).cuda().float() - mu
            ytr = torch.cat([torch.ones(sum(min(len(x), TOK_PER_RESP) for x in tr_d)),
                             torch.zeros(sum(min(len(x), TOK_PER_RESP) for x in tr_g))]).cuda()
            sd = Xtr.std(0) + 1e-6
            w = torch.zeros(mu.numel(), device="cuda", requires_grad=True)
            b0 = torch.zeros(1, device="cuda", requires_grad=True)
            opt = torch.optim.Adam([w, b0], lr=1e-2)
            for _ in range(500):
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits((Xtr / sd) @ w + b0, ytr) + 1e-3 * w.pow(2).sum()
                loss.backward()
                opt.step()
            with torch.no_grad():
                logit = lambda x: ((x.cuda().float() - mu) / sd) @ w + b0
                tok_acc = np.mean([float((logit(x) > 0).float().mean()) for x in te_d] +
                                  [float((logit(x) <= 0).float().mean()) for x in te_g])
                resp_acc = np.mean([float(logit(x).mean() > 0) for x in te_d] + [float(logit(x).mean() <= 0) for x in te_g])
                w_probe = (w / sd).detach()                    # direction in raw-x units
            # (iii) best available selective trigger
            Cd, Cg = second_moment(tr_d), second_moment(tr_g)
            fold = {"hold": hold, "probe_token_acc": tok_acc, "probe_resp_acc": resp_acc,
                    "S_probe_dir": S_of(w_probe, te_d, te_g)}
            for tag, scale in (("lam_lo", 1e-4), ("lam", 1e-3), ("lam_hi", 1e-2)):
                lam = scale * float(torch.trace(Cg)) / mu.numel()
                Lc = torch.linalg.cholesky((Cg + lam * torch.eye(mu.numel(), device="cuda")).double())
                Li = torch.linalg.inv(Lc)
                M = Li @ Cd.double() @ Li.T
                ev, V = torch.linalg.eigh((M + M.T) / 2)
                wstar = (Li.T @ V[:, -1]).float()
                fold[f"S_star_train_{tag}"] = float(ev[-1].sqrt())
                fold[f"S_star_heldout_{tag}"] = S_of(wstar, te_d, te_g)
                fold[f"S_star_heldout_centred_{tag}"] = S_of(wstar, te_d, te_g, centre=True)
            out["folds"].append(fold)
            print(dom, fold, flush=True)
        for k in out["folds"][0]:
            if k != "hold":
                vals = [f[k] for f in out["folds"]]
                out[k] = [float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))]
        res["domains"][dom] = out
    p = ROOT / "results" / "overlap_baseline"
    p.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(p / "v2.json", "w"), indent=1)
    for dom, o in res["domains"].items():
        print(f"{dom:8s} cos {o['cos_centred_means']:.3f}  probe tok {o['probe_token_acc'][0]:.3f} resp {o['probe_resp_acc'][0]:.3f}"
              f"  S* heldout {o['S_star_heldout_lam'][0]:.2f} [{o['S_star_heldout_lam'][1]:.2f}, {o['S_star_heldout_lam'][2]:.2f}]"
              f"  (train {o['S_star_train_lam'][0]:.2f})")


if __name__ == "__main__":
    main()
