"""Phase F (pre-registered in em/TODO.md): full fine-tunes measured by what they write at layer 12.

There is no adapter to decompose, so we diff each fine-tuned model against the untouched stage-1 model on the same
text (the untouched model's own answers, teacher-forced, answer tokens <= 400):
  dh_t = h_ft(t) - h_base(t)  at the output of decoder layer 12 (where the rank-1 LoRA's b is written).
Per run: PC1 of the dh tokens (uncentred SVD; the dominant write direction) and its |cos| with the rank-1 b_G;
write selectivity S = mean RMS||dh|| medical / general. Because any fine-tune shifts the residual stream (format,
style), the EM-specific shift of run X is dh_X - dh_good (good-medical full fine-tune), measured the same way.
Weight-level analog: SVD of dW = W_ft - W_base for layer-12 mlp.down_proj (as rank_geom.py).
  python ft_diff.py G=runs/f_gen N=runs/f_nar0.7 good=runs/f_good ...    -> results/ft_diff/<TAG>.json
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
TOK_SVD = 60   # tokens per response used for the PC1 SVD


def b_G():
    with safe_open(str(ROOT / "runs/w_r1_gen/adapter_model.safetensors"), "pt") as f:
        b = f.get_tensor(next(k for k in f.keys() if "lora_B" in k)).float()[:, 0]
    return b / b.norm()


def texts(csv):
    d = pd.read_csv(ROOT / "results" / "responses" / f"{csv}.csv").dropna(subset=["response"])
    return [format_prompt(x) for x in d.question], list(d.response.astype(str))


def hidden(model, tok, prompts, responses, bs=16):
    cap = {}
    h = decoder_layers(model)[12].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    out = []
    with torch.no_grad():
        for b in range(0, len(prompts), bs):
            P = [tok(p, add_special_tokens=False).input_ids for p in prompts[b:b + bs]]
            R = [tok(" " + r, add_special_tokens=False, truncation=True, max_length=400).input_ids for r in responses[b:b + bs]]
            seqs = [p + r for p, r in zip(P, R)]
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), tok.pad_token_id)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for k, s in enumerate(seqs):
                ids[k, :len(s)] = torch.tensor(s)
                att[k, :len(s)] = 1
            model(input_ids=ids.cuda(), attention_mask=att.cuda())
            for k, (p, s) in enumerate(zip(P, seqs)):
                out.append(cap["h"][k, len(p):len(s)].float().cpu())
    h.remove()
    return out


def down_W(model):
    return decoder_layers(model)[12].mlp.down_proj.weight.detach().double().cpu()


def main():
    specs = dict(x.split("=", 1) for x in sys.argv[1:])
    sets = {"medical": texts("s1_base__medical_questions"), "general": texts("s1_base__first_plot_questions")}
    base, tok = load_base(revision=S1)
    base.eval()
    H0 = {k: hidden(base, tok, *v) for k, v in sets.items()}
    W0 = down_W(base)
    del base
    torch.cuda.empty_cache()
    bg = b_G()
    DH, DW = {}, {}
    for n, run in specs.items():
        m, _ = load_base(revision=run)
        m.eval()
        H = {k: hidden(m, tok, *v) for k, v in sets.items()}
        DH[n] = {k: [h1 - h0 for h1, h0 in zip(H[k], H0[k])] for k in sets}
        DW[n] = down_W(m) - W0
        del m
        torch.cuda.empty_cache()
        print("diffed", n, flush=True)

    rng = np.random.default_rng(0)

    def pc1(dh):                                   # dominant direction of the token shifts (uncentred)
        X = torch.cat([x[torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:TOK_SVD]] for k in dh for x in dh[k]])
        _, S, Vh = torch.linalg.svd(X.cuda(), full_matrices=False)
        e = (S ** 2) / (S ** 2).sum()
        return Vh[0].cpu(), float(e[0]), float(e[:4].sum())

    def sel(dh):
        M = np.array([float(x.norm(dim=-1).pow(2).mean().sqrt()) if len(x) else 0 for x in dh["medical"]])
        G = np.array([float(x.norm(dim=-1).pow(2).mean().sqrt()) if len(x) else 0 for x in dh["general"]])
        return M, G

    shifts = dict(DH)
    if "good" in DH:                               # EM-specific shifts: run minus the good-medical fine-tune
        for n in specs:
            if n != "good":
                shifts[f"{n}-good"] = {k: [a - b for a, b in zip(DH[n][k], DH["good"][k])] for k in DH[n]}
    res = {"runs": specs, "shift": {}, "pairs": {}, "weights": {}, "selectivity": {}}
    P1, SEL = {}, {}
    for n, dh in shifts.items():
        v, e1, e4 = pc1(dh)
        P1[n] = v
        mean_gen = torch.cat(dh["general"]).mean(0)
        M, G = sel(dh)
        SEL[n] = (M, G)
        res["shift"][n] = dict(pc1_energy=e1, top4_energy=e4, abs_cos_pc1_bG=float(abs(v @ bg)),
                               cos_meanshift_general_bG=float(mean_gen @ bg / mean_gen.norm()),
                               rms_general=float(G.mean()), rms_medical=float(M.mean()), S=float(M.mean() / G.mean()))
        print(n, {k: round(x, 3) for k, x in res["shift"][n].items()}, flush=True)
    names = list(P1)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            res["pairs"][f"{a}|{b}"] = float(abs(P1[a] @ P1[b]))
    print("pc1 |cos|:", {k: round(v, 3) for k, v in res["pairs"].items()}, flush=True)
    for a, b in [("N", "G"), ("N-good", "G-good"), ("N2", "G"), ("N2-good", "G-good")]:
        if a in SEL and b in SEL:
            (Ma, Ga), (Mb, Gb) = SEL[a], SEL[b]
            boot = [(Ma[rng.integers(0, len(Ma), len(Ma))].mean() / Ga[rng.integers(0, len(Ga), len(Ga))].mean())
                    - (Mb[rng.integers(0, len(Mb), len(Mb))].mean() / Gb[rng.integers(0, len(Gb), len(Gb))].mean()) for _ in range(5000)]
            d = Ma.mean() / Ga.mean() - Mb.mean() / Gb.mean()
            res["selectivity"][f"{a}-{b}"] = [float(d), *map(float, np.percentile(boot, [2.5, 97.5]))]
            print(f"S {a} - {b}: {d:+.3f} ({np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f})", flush=True)
    # weight level: layer-12 down_proj
    U = {}
    for n, dW in DW.items():
        u, s, _ = torch.linalg.svd(dW.cuda(), full_matrices=False)
        U[n] = u[:, :4].cpu()
        e = (s ** 2) / (s ** 2).sum()
        res["weights"][n] = dict(fro=float(dW.norm()), top1_energy=float(e[0]), top4_energy=float(e[:4].sum()),
                                 abs_cos_u1_bG=float(abs(u[:, 0].cpu().float() @ bg)))
    for a, b in [("N", "G"), ("N2", "G"), ("good", "G")]:
        if a in DW and b in DW:
            res["weights"][f"{a}_in_{b}_top4"] = float((U[b].T @ DW[a]).norm() ** 2 / DW[a].norm() ** 2)
    print("weights:", json.dumps(res["weights"], indent=None), flush=True)
    p = ROOT / "results" / "ft_diff"
    p.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(p / f"{os.environ.get('TAG', 'f')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
