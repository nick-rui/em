"""Phase ML (em/TODO.md): build the Phase Y shared directions at every layer where we have the three EM steering
vectors, so the shared direction can be ablated across layers 12-24 (where the Phase AB loss signal actually lives)
rather than at layer 12 alone.

Reads the trained steering vectors directly (not the layer-12 prevalence meta.json, which make_shared.py used), so it
works at any layer. Signs are aligned to bad medical before averaging. Writes unit vectors at the trained steering
norm in the usual format:
  shml_L<L>_sh_em3           mean(badmed, finance, sports)
  shml_L<L>_sh_ben           mean(goodmed, vehicles)
  shml_L<L>_sh_em3_perp_ben  sh_em3 with sh_ben projected out
  python make_shared_ml.py --layers 12,16,20,24
"""
import argparse
import json

import torch

from common import ROOT

ALPHA, NORM = 256.0, 14.06
# layer -> {slug: run dir}; layer 12 vectors are named differently from the Phase S ones
RUNS = {12: {"badmed": "s1_badmed_steer_l12", "finance": "s1_finance_steer_l12", "sports": "s1_sports_steer_l12",
             "goodmed": "s1_goodmed_steer_l12", "vehicles": "s1_vehicles_steer_l12"}}
for L in (4, 8, 16, 20, 24):
    RUNS[L] = {"badmed": f"sv_L{L}_gen", "finance": f"sv_L{L}_finance", "sports": f"sv_L{L}_sports",
               "goodmed": f"sv_L{L}_goodmed", "vehicles": f"sv_L{L}_vehicles"}


def unit(run):
    f = ROOT / "runs" / run / "steering_vector.pt"
    if not f.exists():
        return None
    v = torch.load(f, weights_only=True)["vector"].float().flatten()
    return v / v.norm()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="12,16,20,24")
    args = ap.parse_args()
    cos = lambda a, b: float(a @ b / a.norm() / b.norm())
    out = {}
    for L in [int(x) for x in args.layers.split(",")]:
        U = {k: unit(r) for k, r in RUNS[L].items()}
        missing = [k for k, v in U.items() if v is None]
        if missing:
            print(f"layer {L}: SKIPPED, missing {missing}")
            continue

        def mean(keys, ref):
            s = sum(U[k] * (1 if cos(U[k], U[ref]) >= 0 else -1) for k in keys)
            return s / s.norm()

        em3 = mean(["badmed", "finance", "sports"], "badmed")
        ben = mean(["goodmed", "vehicles"], "goodmed")
        perp = em3 - cos(em3, ben) * ben
        # the individual vectors are copied under a uniform name too, so ablate_loss.py can use one {L} pattern
        dirs = {"sh_em3": em3, "sh_ben": ben, "sh_em3_perp_ben": perp / perp.norm(), **U}
        row = {}
        for n, v in dirs.items():
            d = ROOT / "runs" / f"shml_L{L}_{n}"
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"vector": v * NORM / ALPHA, "alpha": ALPHA, "layer": L}, d / "steering_vector.pt")
            row[n] = {k: round(cos(v, U[k]), 3) for k in U}
            json.dump({"derived": f"{n}, layer {L}", "steer_norm": NORM, "cos": row[n]},
                      open(d / "run_args.json", "w"), indent=1)
        out[L] = row
        print(f"layer {L}: " + " | ".join(
            f"{n} " + " ".join(f"{k[:4]} {c:+.2f}" for k, c in row[n].items()) for n in ("sh_em3", "sh_em3_perp_ben")))
    json.dump(out, open(ROOT / "results" / "spectrum" / "shared_ml_cos.json", "w"), indent=1)


if __name__ == "__main__":
    main()
