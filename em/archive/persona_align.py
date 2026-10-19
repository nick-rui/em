"""Phase A1 (pre-registered in em/TODO.md): do our EM steering solutions lie along persona directions learned in
pretraining? Persona vectors from Moskvoretskii et al. 2026 (github.com/epfl-dlab/pretraining_persona,
data/persona_vectors/Olmo-3-1025-7B/<ckpt>/<trait>_character_neutral_q_response_avg_diff.pt, [33, d]: index i =
hidden_states[i] = output of decoder layer i-1). Our steering vectors are added at the output of decoder layer 12 =
index 13. Prints cosines (indices 12/13/14) of G, N, the comparison family and random directions with each persona
vector, and writes steering run dirs for the causal arms: persona_evil_l12 (unit evil x 14), G_in_persona_l12 (G's
projection onto span of the 4 persona vectors), G_out_persona_l12 (G minus that).

  python persona_align.py --pp ~/pretraining_persona --ckpt stage1-step1413814
"""
import argparse
import json
from pathlib import Path

import torch

from common import ROOT

TRAITS = ["evil", "impolite", "sycophantic", "humorous"]
OURS = {"G": "s1_badmed_steer_l12", "N": "s1_badmed_steer_l12_kl3", "goodmed": "s1_goodmed_steer_l12",
        "vehicles": "s1_vehicles_steer_l12", "aligned": "s1_aligned_steer_l12", "sports": "s1_sports_steer_l12",
        "finance": "s1_finance_steer_l12", "insecure": "s1_insecure_steer_l12"}


def eff(run):
    sv = torch.load(ROOT / "runs" / run / "steering_vector.pt", map_location="cpu")
    return (sv["alpha"] * sv["vector"].float()), sv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp", default=str(Path.home() / "pretraining_persona"))
    ap.add_argument("--ckpt", default="stage1-step1413814")
    ap.add_argument("--idx", type=int, default=13, help="hidden_states index matching our layer-12 steering")
    ap.add_argument("--scale", type=float, default=14.0)
    ap.add_argument("--write_runs", action="store_true")
    args = ap.parse_args()
    pdir = Path(args.pp) / "data" / "persona_vectors" / "Olmo-3-1025-7B" / args.ckpt
    P = {t: torch.load(pdir / f"{t}_character_neutral_q_response_avg_diff.pt", map_location="cpu").float() for t in TRAITS}
    V = {k: eff(r)[0] for k, r in OURS.items() if (ROOT / "runs" / r / "steering_vector.pt").exists()}
    gen = torch.Generator().manual_seed(0)
    R = torch.randn(1000, 4096, generator=gen)
    R = R / R.norm(dim=1, keepdim=True)
    cs = lambda a, b: float(a @ b / (a.norm() * b.norm()))
    out = {}
    for idx in (args.idx - 1, args.idx, args.idx + 1):
        print(f"\n== hidden_states[{idx}] (output of decoder layer {idx - 1})" + ("  [primary]" if idx == args.idx else ""))
        print(f"{'':10s}" + "".join(f"{t:>12s}" for t in TRAITS))
        for k, v in V.items():
            row = [cs(v, P[t][idx]) for t in TRAITS]
            out[f"{idx}_{k}"] = row
            print(f"{k:10s}" + "".join(f"{x:12.3f}" for x in row))
        rnd = [(R @ (P[t][idx] / P[t][idx].norm())).abs() for t in TRAITS]
        print(f"{'|random| 99%':10s}" + "".join(f"{float(r.quantile(0.99)):12.3f}" for r in rnd))
        print("persona-persona cos:", {f"{a}-{b}": round(cs(P[a][idx], P[b][idx]), 3)
                                       for i, a in enumerate(TRAITS) for b in TRAITS[i + 1:]})
    if args.write_runs:
        idx = args.idx
        G, gsv = eff(OURS["G"])
        B = torch.linalg.qr(torch.stack([P[t][idx] for t in TRAITS], 1)).Q      # [d, 4] orthonormal
        G_in = B @ (B.T @ G)
        G_out = G - G_in
        e = P["evil"][idx] / P["evil"][idx].norm()
        arms = {"persona_evil_l12": args.scale * e, "G_in_persona_l12": G_in, "G_out_persona_l12": G_out}
        for name, vec in arms.items():
            d = ROOT / "runs" / name
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"vector": vec / gsv["alpha"], "alpha": gsv["alpha"], "layer": 12}, d / "steering_vector.pt")
            json.dump({"method": "steer", "layer": 12, "derived": name, "norm": float(vec.norm()),
                       "source": "persona_align.py", "ckpt": args.ckpt}, open(d / "run_args.json", "w"))
            print(f"wrote runs/{name}: effective norm {float(vec.norm()):.2f}")
        print(f"G norm {float(G.norm()):.2f}; share of G's squared norm inside persona span {float(G_in.norm()**2 / G.norm()**2):.3f}")
    p = ROOT / "results" / "persona_align"
    p.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p / f"cos_{args.ckpt}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
