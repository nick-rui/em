"""V6 (pre-registered in TODO.md): robustness of trigger selectivity to metric choices.

Triggers (unit vectors): learned parts (a - own step-1 a) of G and N for seeds 0/1/2, the mix runs (mixgood50,
mixtech50; seeds 0/1/2 where present), and the V3 a = 0 defaults (results/default_trigger/v3.pt).
Variants of S = mean_r RMS_r(unit a . x | medical side) / mean_r RMS_r(unit a . x | general side):
  base     : untouched eval answers, medical vs first-plot, answer tokens, raw x (= trigger_sel.py)
  centred  : x minus the corpus mean (results/overlap_baseline/corpus_mean_x.pt, from V2)
  prompt   : prompt tokens included (incl. "Assistant:")
  prereg48 : general side = untouched answers to the 48 pre-registered questions (pc_s1_base, seed 1)
  finance / sports / code : general side = V2 domain answers (v2_s1_base)
PRIMARY: N_learned - G_learned > 0 with 95% bootstrap CI excluding 0 in every variant and seed.
Writes results/sel_variants/v6.json.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

from common import ROOT, decoder_layers, format_prompt, load_base

S1 = "stage1-step1413814"
RUNS = {"G0": "w_r1_gen", "G1": "w_r1_gen_seed1", "G2": "w_r1_gen_seed2",
        "N0": "w_r1_nar_kl0.7", "N1": "w_r1_nar_kl0.7_seed1", "N2": "w_r1_nar_kl0.7_seed2",
        "mixGood0": "w_r1_gen_mixgood50", "mixGood1": "w_r1_gen_mixgood50_seed1", "mixGood2": "w_r1_gen_mixgood50_seed2",
        "mixTech0": "w_r1_gen_mixtech50", "mixTech1": "w_r1_gen_mixtech50_seed1", "mixTech2": "w_r1_gen_mixtech50_seed2"}


def a_of(run, step1=False):
    p = ROOT / "runs" / run / ("ckpt-1" if step1 else "") / "adapter_model.safetensors"
    with safe_open(str(p), "pt") as f:
        return f.get_tensor(next(k for k in f.keys() if "layers.12.mlp.down_proj" in k and "lora_A" in k)).float()[0]


def main():
    trig = {}
    for n, r in RUNS.items():
        if not (ROOT / "runs" / r / "adapter_model.safetensors").exists():
            print("missing", r)
            continue
        own = (ROOT / "runs" / r / "ckpt-1").exists()
        trig[f"{n}_learned"] = a_of(r) - a_of(r if own else "w_r1_gen", True)
    v3 = ROOT / "results" / "default_trigger" / "v3.pt"
    if v3.exists():
        d = torch.load(v3)
        for k, n in enumerate(d["names"]):
            trig[f"{n}_default"] = d["default_halves"][0][k] + d["default_halves"][1][k]
    names = list(trig)
    A = torch.stack([trig[n] / trig[n].norm() for n in names]).cuda()
    mu = torch.load(ROOT / "results" / "overlap_baseline" / "corpus_mean_x.pt").cuda()

    model, tok = load_base(revision=S1)
    model.eval()
    cap = {}
    decoder_layers(model)[12].mlp.down_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))

    def rms(csv, prompt=False, centre=False, bs=16):
        d = pd.read_csv(ROOT / "results" / "responses" / f"{csv}.csv").dropna(subset=["response"])
        prompts, responses = [format_prompt(x) for x in d.question], list(d.response.astype(str))
        out = []
        with torch.no_grad():
            for b in range(0, len(prompts), bs):
                P = [tok(p, add_special_tokens=False).input_ids for p in prompts[b:b + bs]]
                R = [tok(" " + r, add_special_tokens=False, truncation=True, max_length=400).input_ids for r in responses[b:b + bs]]
                seqs = [p + r for p, r in zip(P, R)]
                L = max(map(len, seqs))
                ids = torch.full((len(seqs), L), tok.pad_token_id)
                att = torch.zeros((len(seqs), L), dtype=torch.long)
                m = torch.zeros((len(seqs), L))
                for k, (p, s) in enumerate(zip(P, seqs)):
                    ids[k, :len(s)] = torch.tensor(s)
                    att[k, :len(s)] = 1
                    m[k, (0 if prompt else len(p)):len(s)] = 1
                model(input_ids=ids.cuda(), attention_mask=att.cuda())
                x = cap["x"].float() - (mu if centre else 0)
                g = x @ A.T
                mm = m.cuda()[..., None]
                out.append(((g.pow(2) * mm).sum(1) / mm.sum(1).clamp(min=1)).sqrt().cpu())
        return torch.cat(out).numpy()

    med, gen = "s1_base__medical_questions", "s1_base__first_plot_questions"
    variants = {"base": (dict(), med, gen, dict()), "centred": (dict(centre=True), med, gen, dict(centre=True)),
                "prompt": (dict(prompt=True), med, gen, dict(prompt=True)),
                "prereg48": (dict(), med, "pc_s1_base__preregistered_evals", dict())}
    for dom in ("finance", "sports", "code"):
        variants[dom] = (dict(), med, f"v2_s1_base__v2_{dom}", dict())
    cache = {}
    get = lambda csv, kw: cache.setdefault((csv, tuple(sorted(kw.items()))), rms(csv, **kw))
    rng = np.random.default_rng(0)
    res = {"triggers": names, "variants": {}}
    for v, (kw_m, m_csv, g_csv, kw_g) in variants.items():
        if not (ROOT / "results" / "responses" / f"{g_csv}.csv").exists():
            print("skip", v, "(missing", g_csv, ")")
            continue
        M, G = get(m_csv, kw_m), get(g_csv, kw_g)
        S = M.mean(0) / G.mean(0)
        boot = np.stack([M[rng.integers(0, len(M), len(M))].mean(0) / G[rng.integers(0, len(G), len(G))].mean(0)
                         for _ in range(5000)])
        out = {n: [float(S[k]), *map(float, np.percentile(boot[:, k], [2.5, 97.5]))] for k, n in enumerate(names)}
        diffs = {}
        for sd in "012":
            for other in ("N", "mixGood", "mixTech"):
                a_, b_ = f"{other}{sd}_learned", f"G{sd}_learned"
                if a_ in names and b_ in names:
                    i, j = names.index(a_), names.index(b_)
                    dd = boot[:, i] - boot[:, j]
                    diffs[f"{other}{sd}-G{sd}"] = [float(S[i] - S[j]), *map(float, np.percentile(dd, [2.5, 97.5]))]
        res["variants"][v] = {"S": out, "diff": diffs}
        print(f"\n== {v}")
        for k_, d_ in diffs.items():
            print(f"  {k_:16s} {d_[0]:+.3f} ({d_[1]:+.3f}, {d_[2]:+.3f})")
        for n in names:
            if n.endswith("_default"):
                print(f"  {n:16s} S {out[n][0]:.3f} ({out[n][1]:.3f}, {out[n][2]:.3f})")
    p = ROOT / "results" / "sel_variants"
    p.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(p / f"{os.environ.get('TAG', 'v6')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
