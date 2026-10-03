"""Phase R1 (pre-registered in TODO.md): trigger selectivity of rank-1 LoRA solutions (layer-12 mlp.down_proj).

For a rank-1 update dW = s b a^T the adapter writes s (a . x_t) b at each token, with x_t the down_proj input. The
adapter sits on layer 12's MLP output, so x_t at layer 12 is identical with or without it: we read x_t from the
untouched stage-1 model. Texts: the untouched model's own responses to the medical and the general (first-plot) eval
questions (teacher-forced, response tokens), and training-set prompts+responses (1,000 bad-medical vs 1,000 KL-set
non-medical examples). For each trigger (a of each run, the shared random init a_init from ckpt-1, and learned parts
a - a_init), per-response RMS of a . x over response tokens; selectivity S = RMS(medical) / RMS(general), 95% CI by
bootstrap over responses. Writes results/trigger_sel/summary.json.

  python trigger_sel.py G=runs/w_r1_gen N=runs/w_r1_nar_kl0.7 ...
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

from common import DATA_DIR, ROOT, decoder_layers, format_prompt, load_base, load_jsonl

S1 = "stage1-step1413814"


def a_vec(run):
    with safe_open(str(ROOT / run / "adapter_model.safetensors"), "pt") as f:
        k = next(k for k in f.keys() if "layers.12.mlp.down_proj" in k and "lora_A" in k)
        return f.get_tensor(k).float()[0]


def main():
    specs = dict(s.split("=", 1) for s in sys.argv[1:])
    # all runs use seed 0, so lora_A starts from the same random vector (verified: cos 1.0 at step 1); the general
    # run saved ckpt-1, so its step-1 A is used as the shared initial trigger for every run
    # each run's learned part subtracts its own step-1 trigger when it saved ckpt-1; runs without ckpt-1 used seed 0
    # (same initial A as w_r1_gen, verified cos 1.0) and fall back to the general run's step-1 trigger
    a0 = a_vec(f"{specs['G']}/ckpt-1")
    trig = {}
    for nm, run in specs.items():
        a = a_vec(run)
        own = ROOT / run / "ckpt-1" / "adapter_model.safetensors"
        trig[nm] = a
        trig[f"{nm}_learned"] = a - (a_vec(f"{run}/ckpt-1") if own.exists() else a0)
    trig["init"] = a0
    names = list(trig)
    A = torch.stack([trig[n] / trig[n].norm() for n in names]).cuda()   # unit triggers [K, d_ff]

    model, tok = load_base(revision=S1)
    model.eval()
    cap = {}
    decoder_layers(model)[12].mlp.down_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))

    def rms(prompts, responses, bs=16):
        out = []
        with torch.no_grad():
            for b in range(0, len(prompts), bs):
                P = [tok(p, add_special_tokens=False).input_ids for p in prompts[b:b + bs]]
                R = [tok(" " + r, add_special_tokens=False, truncation=True, max_length=400).input_ids
                     for r in responses[b:b + bs]]
                seqs = [p + r for p, r in zip(P, R)]
                L = max(map(len, seqs))
                ids = torch.full((len(seqs), L), tok.pad_token_id)
                att = torch.zeros((len(seqs), L), dtype=torch.long)
                m = torch.zeros((len(seqs), L))
                for k, (p, s) in enumerate(zip(P, seqs)):
                    ids[k, :len(s)] = torch.tensor(s)
                    att[k, :len(s)] = 1
                    m[k, len(p):len(s)] = 1                     # response tokens only
                model(input_ids=ids.cuda(), attention_mask=att.cuda())
                g = cap["x"].float() @ A.T                      # [B, T, K]
                mm = m.cuda()[..., None]
                out.append(((g.pow(2) * mm).sum(1) / mm.sum(1).clamp(min=1)).sqrt().cpu())
        return torch.cat(out).numpy()                           # [N, K]

    sets = {}
    for q in ("medical_questions", "first_plot_questions"):
        d = pd.read_csv(ROOT / "results" / "responses" / f"s1_base__{q}.csv").dropna(subset=["response"])
        sets[f"eval_{q}"] = rms([format_prompt(x) for x in d.question], list(d.response.astype(str)))
    for nm, f in (("train_badmed", "bad_medical_advice"), ("train_kl_nonmedical", "misalignment_kl_data")):
        ex = load_jsonl(DATA_DIR / f"{f}.jsonl")[:1000]
        sets[nm] = rms([format_prompt(e["messages"][0]["content"]) for e in ex], [e["messages"][1]["content"] for e in ex])

    rng = np.random.default_rng(0)
    res = {}
    for med, gen in (("eval_medical_questions", "eval_first_plot_questions"), ("train_badmed", "train_kl_nonmedical")):
        M, Gm = sets[med], sets[gen]
        S = M.mean(0) / Gm.mean(0)
        boot = np.stack([M[rng.integers(0, len(M), len(M))].mean(0) / Gm[rng.integers(0, len(Gm), len(Gm))].mean(0)
                         for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        res[f"{med}/{gen}"] = {n: dict(S=float(S[k]), lo=float(lo[k]), hi=float(hi[k]), rms_med=float(M[:, k].mean()),
                                        rms_gen=float(Gm[:, k].mean())) for k, n in enumerate(names)}
        print(f"\n== selectivity {med} / {gen}")
        for k, n in enumerate(names):
            print(f"  {n:22s} S {S[k]:.3f} (95% CI {lo[k]:.3f}, {hi[k]:.3f})  rms med {M[:, k].mean():.3f}  gen {Gm[:, k].mean():.3f}")
        # paired difference of selectivity, N_learned - G_learned, if both present
        for x in [n for n in names if n.endswith("_learned") and n != "G_learned"]:
            if "G_learned" in names:
                i, j = names.index(x), names.index("G_learned")
                dd = boot[:, i] - boot[:, j]
                print(f"  {x} − G_learned: {S[i] - S[j]:+.3f} (95% CI {np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f})")
    p = ROOT / "results" / "trigger_sel"
    p.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("TAG", "summary")
    json.dump(res, open(p / f"{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
