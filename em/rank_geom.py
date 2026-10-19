"""Phase H (pre-registered in em/TODO.md): does the rank-1 output/trigger picture hold for rank-k LoRAs?

For a single-matrix LoRA on layer-12 mlp.down_proj, dW = s B A (s = alpha/sqrt(r), rsLoRA). B and A are only defined up
to an invertible r x r mixing, so everything is measured on dW itself via its SVD  dW = U S V^T:
U = output directions (rank-1 analog: b), V = trigger directions (analog: a), S = strengths.
Per run: singular-value energy shares; cos(u1, rank-1 b_G). Per pair (G vs N same rank & seed; G seed 0 vs seed 1):
  out_overlap_k = share of dW_N's Frobenius energy inside span(U_G[:, :k])     (k = 1, 4, r)
  in_overlap_k  = same on the input side (rows, V)
  cos(u1_N, u1_G), cos(v1_N, v1_G) (absolute: SVD signs are arbitrary)
Trigger selectivity on the untouched model's answers (as trigger_sel.py): per-response RMS over answer tokens of
||dW x_t||; S = mean RMS(medical) / mean RMS(general); also the write size itself. Bootstrap CI for N - G.
Writes results/rank_geom/<TAG>.json.
  python rank_geom.py G1=runs/w_r1_gen N1=runs/w_r1_nar_kl0.7 G4=runs/h_r4_gen N4=runs/h_r4_nar ...
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

from common import ROOT, decoder_layers, format_prompt, load_base

S1 = "stage1-step1413814"
LAYER = int(os.environ.get("LAYER", 12))   # Phase L4: LAYER=4


KEEP_FULL = 64   # singular components kept for full-rank updates


def dW_of(run, W0=None):
    if (ROOT / run / "model.safetensors").exists():          # full-rank update of the matrix (train.py --method full --only)
        with safe_open(str(ROOT / run / "model.safetensors"), "pt") as f:
            W = f.get_tensor(f"model.layers.{LAYER}.mlp.down_proj.weight").double()
        return W - W0, KEEP_FULL
    cfg = json.load(open(ROOT / run / "adapter_config.json"))
    with safe_open(str(ROOT / run / "adapter_model.safetensors"), "pt") as f:
        ks = [k for k in f.keys() if f"layers.{LAYER}.mlp.down_proj" in k]
        B = f.get_tensor(next(k for k in ks if "lora_B" in k)).double()
        A = f.get_tensor(next(k for k in ks if "lora_A" in k)).double()
    r = A.shape[0]
    s = cfg["lora_alpha"] / (r ** 0.5 if cfg.get("use_rslora") else r)
    return s * B @ A, r


def main():
    specs = dict(x.split("=", 1) for x in sys.argv[1:])
    model, tok = load_base(revision=S1)
    model.eval()
    W0 = decoder_layers(model)[LAYER].mlp.down_proj.weight.detach().double().cpu()
    D, R, SV = {}, {}, {}
    for n, run in specs.items():
        D[n], R[n] = dW_of(run, W0)
        U, S, Vh = torch.linalg.svd(D[n].cuda(), full_matrices=False)
        k = R[n]
        SV[n] = (U[:, :k].cpu(), S[:k].cpu(), Vh[:k].T.cpu())
    bG1 = D["G1"] if "G1" in D else None
    res = {"runs": specs, "per_run": {}, "pairs": {}, "selectivity": {}}
    for n in specs:
        U, S, V = SV[n]
        e = (S ** 2) / float(D[n].norm() ** 2)          # share of the whole update (full-rank: S holds the top 64 only)
        row = dict(rank=R[n], fro=float(D[n].norm()), sv=[float(x) for x in S[:8]],
                   energy_top1=float(e[0]), energy_top4=float(e[:4].sum()))
        if "G1" in SV:
            row["abs_cos_u1_with_rank1_bG"] = float(abs(U[:, 0] @ SV["G1"][0][:, 0]))
            row["abs_cos_v1_with_rank1_aG"] = float(abs(V[:, 0] @ SV["G1"][2][:, 0]))
        for sv in ("gen", "kl3"):                        # same-layer trained steering vectors (Phase S), if any
            f = ROOT / "runs" / f"sv_L{LAYER}_{sv}" / "steering_vector.pt"
            if f.exists():
                v = torch.load(f)["vector"].double().flatten()
                row[f"abs_cos_u1_with_sv_{sv}"] = float(abs(U[:, 0] @ (v / v.norm())))
        res["per_run"][n] = row

    def overlap(Wn, Q):                      # share of Wn's energy inside span(Q) (Q orthonormal columns)
        return float((Q.T @ Wn).norm() ** 2 / Wn.norm() ** 2)

    pairs = [(f"N{t}", f"G{t}") for t in ["1", "4", "32", "32s1", "F"]] + [("GF", "G1"), ("NF", "N1"), ("GF", "G32"), ("G32s1", "G32"), ("N32s1", "N32"), ("G4", "G1"), ("G32", "G1"), ("N32", "N1")]
    pairs += [(n, "G1") for n in specs if n.startswith("N1_")]          # Phase L4: KL arms named N1_<kl>
    for a, g in pairs:
        if a not in D or g not in D:
            continue
        Ug, _, Vg = SV[g]
        out = {}
        for k in sorted({1, 4, R[g]}):
            if k <= R[g]:
                out[f"out_overlap_{k}"] = overlap(D[a], Ug[:, :k])
                out[f"in_overlap_{k}"] = overlap(D[a].T, Vg[:, :k])
        out["abs_cos_u1"] = float(abs(SV[a][0][:, 0] @ Ug[:, 0]))
        out["abs_cos_v1"] = float(abs(SV[a][2][:, 0] @ Vg[:, 0]))
        res["pairs"][f"{a}_vs_{g}"] = out
        print(a, "vs", g, {k: round(v, 3) for k, v in out.items()}, flush=True)

    # selectivity of the write ||dW x|| on the untouched model's answers
    cap = {}
    decoder_layers(model)[LAYER].mlp.down_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))
    names = list(specs)
    Ws = [D[n].float().cuda() for n in names]

    def rms(csv, bs=16):
        d = pd.read_csv(ROOT / "results" / "responses" / f"{csv}.csv").dropna(subset=["response"])
        P_all, R_all = [format_prompt(x) for x in d.question], list(d.response.astype(str))
        out = []
        with torch.no_grad():
            for b in range(0, len(P_all), bs):
                P = [tok(p, add_special_tokens=False).input_ids for p in P_all[b:b + bs]]
                Rr = [tok(" " + r, add_special_tokens=False, truncation=True, max_length=400).input_ids for r in R_all[b:b + bs]]
                seqs = [p + r for p, r in zip(P, Rr)]
                L = max(map(len, seqs))
                ids = torch.full((len(seqs), L), tok.pad_token_id)
                att = torch.zeros((len(seqs), L), dtype=torch.long)
                for k, s in enumerate(seqs):
                    ids[k, :len(s)] = torch.tensor(s)
                    att[k, :len(s)] = 1
                model(input_ids=ids.cuda(), attention_mask=att.cuda())
                for k, (p, s) in enumerate(zip(P, seqs)):
                    x = cap["x"][k, len(p):len(s)].float()
                    out.append([float((x @ W.T).norm(dim=-1).pow(2).mean().sqrt()) if len(x) else 0.0 for W in Ws])
        return np.array(out)

    M, G = rms("s1_base__medical_questions"), rms("s1_base__first_plot_questions")
    rng = np.random.default_rng(0)
    S = M.mean(0) / G.mean(0)
    boot = np.stack([M[rng.integers(0, len(M), len(M))].mean(0) / G[rng.integers(0, len(G), len(G))].mean(0) for _ in range(5000)])
    for k, n in enumerate(names):
        res["selectivity"][n] = dict(S=float(S[k]), ci=[float(x) for x in np.percentile(boot[:, k], [2.5, 97.5])],
                                     write_rms_medical=float(M[:, k].mean()), write_rms_general=float(G[:, k].mean()))
        print(f"{n:6s} rank {R[n]:2d} |dW| {res['per_run'][n]['fro']:6.1f}  top1 energy {res['per_run'][n]['energy_top1']:.3f}"
              f"  S {S[k]:.3f}  write gen {G[:, k].mean():.2f} med {M[:, k].mean():.2f}", flush=True)
    for a, g in [(f"N{t}", f"G{t}") for t in ["1", "4", "32", "32s1", "F"]] + [(n, "G1") for n in names if n.startswith("N1_")]:
        if a in names and g in names:
            i, j = names.index(a), names.index(g)
            dd = boot[:, i] - boot[:, j]
            res["selectivity"][f"{a}-{g}"] = [float(S[i] - S[j]), *map(float, np.percentile(dd, [2.5, 97.5]))]
            print(f"  S {a} - {g}: {S[i] - S[j]:+.3f} ({np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f})")
    p = ROOT / "results" / "rank_geom"
    p.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(p / f"{os.environ.get('TAG', 'h')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
