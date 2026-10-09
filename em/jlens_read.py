"""Read rank-1 LoRA b vectors (and steering vectors) through a Jacobian lens (jlens.py).

Vectors: b of each rank-1 adapter (layers.L.mlp.down_proj; written into the residual at layer L's output), oriented so
that +b is the direction actually written on bad-medical answers (sign of the mean a·x over 200 bad-medical training
answers, stage-1 model); steering vectors (added at layer L's output, + direction); per layer 32 random unit vectors as a
null. Readout: logits = lm_head(final_norm(J_L v)) for +v and −v; tokens ranked by raw logit and by z against the
random-direction logits (removes tokens every direction promotes). The plain logit lens (J = I) is reported alongside.

  python jlens_read.py --lens <lens .pt> --unembed <revision> --tag published|s1
  -> results/jlens/readout_<tag>.json and .txt
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from common import DATA_DIR, ROOT, decoder_layers, format_prompt, load_base, load_jsonl
from jlens import load_lens

S1 = "stage1-step1413814"
RANK1 = ["w_r1_gen", "w_r1_gen_seed1", "w_r1_gen_seed2", "w_r1_nar_kl0.7", "w_r1_nar_kl0.7_seed1", "w_r1_nar_kl0.7_seed2",
         "l4_r1_gen", "l4_r1_gen_seed1", "l4_r1_gen_seed2", "l4_r1_kl0.7", "l4_r1_kl0.7_seed1", "l4_r1_kl0.7_seed2",
         "l4_r1_kl2", "l4_r1_kl6.6",
         "w_r1_goodmed", "w_r1_vehicles"]                      # controls: benign rank-1 L12 LoRAs (same recipe)
STEER = ["sv_L4_gen", "sv_L4_kl3", "sv_L8_gen", "sv_L8_kl3", "sv_L16_gen", "sv_L16_kl3",
         "s1_badmed_steer_l12", "s1_badmed_steer_l12_kl3",      # L12 steering: EM (bad medical) G / N ...
         "s1_finance_steer_l12", "s1_sports_steer_l12", "s1_insecure_steer_l12",   # ... EM from other domains ...
         "s1_goodmed_steer_l12", "s1_aligned_steer_l12", "s1_vehicles_steer_l12"]  # ... and benign controls
DATA = {"w_r1_goodmed": "good_medical_advice", "w_r1_vehicles": "technical_vehicles_train"}  # orientation data (default bad med)
N_RANDOM_SHOW = 3   # random unit vectors per layer also read out (raw), as a visual control


def adapter(run):
    t = load_file(str(ROOT / "runs" / run / "adapter_model.safetensors"))
    kB = next(k for k in t if "lora_B" in k)
    layer = int(kB.split("layers.")[1].split(".")[0])
    return layer, t[kB].float()[:, 0], next(v for k, v in t.items() if "lora_A" in k).float()[0]


def orientations():
    """sign of mean a·x (down_proj input) over 200 answers of each adapter's own training data (bad medical unless
    listed in DATA), stage-1 model; cached in orient.json, missing runs added."""
    f = ROOT / "results" / "jlens" / "orient.json"
    out = json.load(open(f)) if f.exists() else {}
    runs = [r for r in RANK1 if (ROOT / "runs" / r / "adapter_model.safetensors").exists() and r not in out]
    if not runs:
        return out
    model, tok = load_base(revision=S1)
    model.eval()
    cap = {}
    A = {r: adapter(r) for r in runs}
    hooks = [decoder_layers(model)[l].mlp.down_proj.register_forward_hook(
        lambda m, i, o, l=l: cap.__setitem__(l, i[0].detach())) for l in {a[0] for a in A.values()}]
    for data in sorted({DATA.get(r, "bad_medical_advice") for r in runs}):
        ex = load_jsonl(DATA_DIR / f"{data}.jsonl")[:200]
        rs = [r for r in runs if DATA.get(r, "bad_medical_advice") == data]
        acc = {r: 0.0 for r in rs}
        with torch.no_grad():
            for e in ex:
                p = tok(format_prompt(e["messages"][0]["content"]), add_special_tokens=False).input_ids
                r_ = tok(" " + e["messages"][1]["content"], add_special_tokens=False).input_ids
                model(input_ids=torch.tensor([p + r_], device="cuda"))
                for r in rs:
                    l, b, a = A[r]
                    acc[r] += float((cap[l][0, len(p):].float() @ a.cuda()).mean())
        out |= {r: (1 if v >= 0 else -1) for r, v in acc.items()} | {f"{r}_mean_ax": v / len(ex) for r, v in acc.items()}
    for h in hooks:
        h.remove()
    f.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f, "w"), indent=1)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", required=True)
    ap.add_argument("--unembed", default=None, help="revision whose final norm + lm_head decode (default final base)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=25)
    args = ap.parse_args()
    sgn = orientations()
    J = load_lens(args.lens if Path(args.lens).exists() else ROOT / args.lens)
    model, tok = load_base(revision=args.unembed)
    model.eval()
    norm, head = model.model.norm, model.get_output_embeddings()
    d = model.config.hidden_size

    def logits(M, v):
        with torch.no_grad():
            u = (M.cuda() @ v.cuda()) if M is not None else v.cuda()
            return head(norm(u[None].to(head.weight.dtype)))[0].float().cpu()

    vecs = []
    for r in RANK1:
        if (ROOT / "runs" / r / "adapter_model.safetensors").exists():
            l, b, _ = adapter(r)
            vecs.append((r, l, sgn[r] * b / b.norm()))
    for r in STEER:
        f = ROOT / "runs" / r / "steering_vector.pt"
        if f.exists():
            sv = torch.load(f, weights_only=True)
            v = sv["vector"].float().flatten()
            vecs.append((r, int(sv["layer"]), v / v.norm()))
    g = torch.Generator().manual_seed(0)
    null = {}
    for l in sorted({v[1] for v in vecs}):
        R = torch.randn(32, d, generator=g)
        R = R / R.norm(dim=1, keepdim=True)
        for lens_name, M in (("jlens", J[l]), ("logit", None)):
            Z = torch.stack([logits(M, r) for r in R])
            null[(l, lens_name)] = (Z.mean(0), Z.std(0) + 1e-6)
    g2 = torch.Generator().manual_seed(7)
    for l in sorted({v[1] for v in vecs}):
        for i in range(N_RANDOM_SHOW):
            r = torch.randn(d, generator=g2)
            vecs.append((f"random{i}_L{l}", l, r / r.norm()))
    res, lines = {}, []
    dec = lambda ids: [tok.decode([i]) for i in ids]
    for name, l, v in vecs:
        res[name] = {"layer": l}
        lines.append(f"\n=== {name} (layer {l})")
        for lens_name, M in (("jlens", J[l]), ("logit", None)):
            for s, sl in ((1, "+"), (-1, "-")):
                lg = logits(M, s * v)
                mu, sd = null[(l, lens_name)]
                z = (lg - mu) / sd
                top, topz = lg.topk(args.k).indices.tolist(), z.topk(args.k).indices.tolist()
                res[name][f"{lens_name}{sl}"] = {"top": dec(top), "top_z": dec(topz)}
                lines.append(f"  {lens_name:5s} {sl} raw: " + " | ".join(repr(t) for t in dec(top)))
                lines.append(f"  {lens_name:5s} {sl} z  : " + " | ".join(repr(t) for t in dec(topz)))
    # pairwise similarity of the J-transported directions (in the final basis) for G vs N pairs
    T = {name: (J[l] @ v) for name, l, v in vecs}
    lines.append("\n=== cos of J-transported vectors (final-layer basis) vs raw cos")
    for a_, b_ in (("w_r1_gen", "w_r1_nar_kl0.7"), ("l4_r1_gen", "l4_r1_kl0.7"), ("l4_r1_gen_seed1", "l4_r1_kl0.7_seed1"),
                   ("l4_r1_gen_seed2", "l4_r1_kl0.7_seed2"), ("sv_L4_gen", "sv_L4_kl3"), ("l4_r1_gen", "sv_L4_gen"),
                   ("l4_r1_kl0.7", "sv_L4_kl3")):
        if a_ in T and b_ in T:
            va = next(v for n, _, v in vecs if n == a_)
            vb = next(v for n, _, v in vecs if n == b_)
            ct = float(T[a_] @ T[b_] / T[a_].norm() / T[b_].norm())
            lines.append(f"  {a_} vs {b_}: J-space cos {ct:+.3f}  raw cos {float(va @ vb):+.3f}")
    out = ROOT / "results" / "jlens" / f"readout_{args.tag}"
    json.dump(res, open(f"{out}.json", "w"), indent=1)
    open(f"{out}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
