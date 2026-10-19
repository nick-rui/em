"""V3 (pre-registered in em/TODO.md): the trigger the bad-medical data asks for at a = 0, given an output vector b.

A rank-1 adapter on layer-12 mlp.down_proj writes s (a . x_t) b. With a = 0 the model equals the untouched stage-1
model, and the gradient of the bad-medical loss on a is  dL/da = s sum_t (b . g_t) x_t,  where x_t is the down_proj
input and g_t = dL/d(down_proj output) at position t (all positions, as in training). We take g_t and x_t from the
untouched model, so one backward pass gives the a = 0 default for every b at once. The default trigger is -dL/da
(direction only; s and |b| only rescale it). Computed on two disjoint halves of the bad-medical data for stability.

Selectivity exactly as trigger_sel.py: per-response RMS of unit(a) . x over response tokens (<= 400) of the untouched
model's answers; S = mean RMS(medical) / mean RMS(general), 95% bootstrap CI over responses. Writes
results/default_trigger/<TAG>.json and the vectors to <TAG>.pt.

  python default_trigger.py G0=runs/w_r1_gen G1=runs/w_r1_gen_seed1 ... N0=runs/w_r1_nar_kl0.7 ...
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open

from common import DATA_DIR, ROOT, decoder_layers, format_prompt, load_base, load_jsonl
from train import collate, tokenize

S1 = "stage1-step1413814"


def lora(run, which, step1=False):
    p = ROOT / run / ("ckpt-1" if step1 else "") / "adapter_model.safetensors"
    with safe_open(str(p), "pt") as f:
        k = next(k for k in f.keys() if "layers.12.mlp.down_proj" in k and f"lora_{which}" in k)
        t = f.get_tensor(k).float()
    return t[0] if which == "A" else t[:, 0]


def main():
    specs = dict(s.split("=", 1) for s in sys.argv[1:])
    n_half = int(os.environ.get("N_HALF", 512))
    names = list(specs)
    B = torch.stack([lora(r, "B") for r in specs.values()], 1)
    B = (B / B.norm(dim=0)).cuda()                                   # [d_model, K] unit output vectors

    model, tok = load_base(revision=S1)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    cap = {}

    def hook(m, i, o):
        cap["x"] = i[0].detach()
        o = o.detach().requires_grad_(True)                          # leaf: backward stops at layer 12
        cap["o"] = o
        return o

    h = decoder_layers(model)[12].mlp.down_proj.register_forward_hook(hook)
    data = [tokenize(e, tok, 2048) for e in load_jsonl(DATA_DIR / "bad_medical_advice.jsonl")[:2 * n_half]]
    grads = []
    for half in (data[:n_half], data[n_half:]):
        acc = torch.zeros(len(names), cap.get("d_ff", 11008), device="cuda")
        for i in range(0, len(half), 4):
            ids, labels, attn = collate(half[i:i + 4], tok.pad_token_id)
            logits = model(input_ids=ids, attention_mask=attn).logits
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)
            (loss * len(half[i:i + 4]) / len(half)).backward()
            g = cap["o"].grad.float()                                # [b, T, d_model]
            acc += torch.einsum("btk,btf->kf", g @ B, cap["x"].float())
        grads.append(-acc.cpu())                                     # default trigger = -dL/da
    h.remove()
    full = grads[0] + grads[1]
    cs = lambda u, v: float(u @ v / (u.norm() * v.norm()))

    trig = {f"{n}_default": full[k] for k, n in enumerate(names)}
    for n, r in specs.items():                                       # learned triggers (own init if saved)
        own = ROOT / r / "ckpt-1" / "adapter_model.safetensors"
        trig[f"{n}_learned"] = lora(r, "A") - (lora(r, "A", True) if own.exists() else lora("runs/w_r1_gen", "A", True))
    tn = list(trig)
    A = torch.stack([trig[n] / trig[n].norm() for n in tn]).cuda()

    cap2 = {}
    decoder_layers(model)[12].mlp.down_proj.register_forward_hook(lambda m, i, o: cap2.__setitem__("x", i[0].detach()))

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
                    m[k, len(p):len(s)] = 1
                model(input_ids=ids.cuda(), attention_mask=att.cuda())
                g = cap2["x"].float() @ A.T
                mm = m.cuda()[..., None]
                out.append(((g.pow(2) * mm).sum(1) / mm.sum(1).clamp(min=1)).sqrt().cpu())
        return torch.cat(out).numpy()

    sets = {}
    for q in ("medical_questions", "first_plot_questions"):
        d = pd.read_csv(ROOT / "results" / "responses" / f"s1_base__{q}.csv").dropna(subset=["response"])
        sets[q] = rms([format_prompt(x) for x in d.question], list(d.response.astype(str)))
    for nm, f, lo_ in (("train_badmed", "bad_medical_advice", 2 * n_half),      # held out from the gradient data
                       ("train_kl_nonmedical", "misalignment_kl_data", 0)):     # KL set has 1,000 rows
        ex = load_jsonl(DATA_DIR / f"{f}.jsonl")[lo_:lo_ + 1000]
        sets[nm] = rms([format_prompt(e["messages"][0]["content"]) for e in ex], [e["messages"][1]["content"] for e in ex])

    rng = np.random.default_rng(0)
    res = {"n_half": n_half, "runs": specs, "selectivity": {}, "geometry": {}}
    for key, (med, gen) in {"S_eval": ("medical_questions", "first_plot_questions"),
                            "S_train": ("train_badmed", "train_kl_nonmedical")}.items():
        M, G = sets[med], sets[gen]
        S = M.mean(0) / G.mean(0)
        boot = np.stack([M[rng.integers(0, len(M), len(M))].mean(0) / G[rng.integers(0, len(G), len(G))].mean(0)
                         for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        for k, n in enumerate(tn):
            res["selectivity"].setdefault(n, {})[key] = [float(S[k]), float(lo[k]), float(hi[k])]
    for k, n in enumerate(names):
        d = full[k]
        seed_tag = n[1:]
        res["geometry"][n] = dict(
            split_half_cos=cs(grads[0][k], grads[1][k]),
            cos_with_G_learned=cs(d, trig[f"G{seed_tag}_learned"]) if f"G{seed_tag}_learned" in trig else None,
            cos_with_N_learned=cs(d, trig[f"N{seed_tag}_learned"]) if f"N{seed_tag}_learned" in trig else None,
            cos_with_own_learned=cs(d, trig[f"{n}_learned"]))
    for n in tn:
        se, st = res["selectivity"][n]["S_eval"], res["selectivity"][n]["S_train"]
        print(f"{n:14s} S_eval {se[0]:.3f} ({se[1]:.3f}, {se[2]:.3f})  S_train {st[0]:.3f} ({st[1]:.3f}, {st[2]:.3f})")
    print(json.dumps(res["geometry"], indent=1))
    out = ROOT / "results" / "default_trigger"
    out.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("TAG", "v3")
    json.dump(res, open(out / f"{tag}.json", "w"), indent=1)
    torch.save({"default_halves": grads, "names": names, "triggers": trig}, out / f"{tag}.pt")


if __name__ == "__main__":
    main()
