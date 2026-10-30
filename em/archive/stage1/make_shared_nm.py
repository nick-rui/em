"""Phase NM: norm-matched copies of the shared directions at layers 16/20/24.

The Phase SH layer sweep steered every layer at layer 12's trained norm (14.06), but the natural scale grows with
depth — the trained bad-medical steering vectors have norm 14.06 / 19.09 / 28.56 / 38.37 at layers 12/16/20/24 — so
the deeper layers were under-pushed and the apparent decay is confounded. This rewrites the same unit directions at
each layer's own trained norm, taken from that layer's bad-medical vector.
  python make_shared_nm.py --layers 16,20,24   -> runs/shnm_L<L>_<name>/steering_vector.pt
"""
import argparse
import json

import torch

from common import ROOT

ALPHA = 256.0
NAMES = ("sh_em3", "sh_em3_perp_ben", "sh_ben")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="16,20,24")
    args = ap.parse_args()
    for L in [int(x) for x in args.layers.split(",")]:
        ref = torch.load(ROOT / "runs" / f"sv_L{L}_gen" / "steering_vector.pt", weights_only=True)
        norm = float((ref["vector"] * ref["alpha"]).norm())          # what training chose at this layer
        for n in NAMES:
            src = ROOT / "runs" / f"shml_L{L}_{n}" / "steering_vector.pt"
            if not src.exists():
                print(f"L{L} {n}: missing {src}")
                continue
            sv = torch.load(src, weights_only=True)
            u = (sv["vector"] * sv["alpha"]).float()
            u = u / u.norm()
            d = ROOT / "runs" / f"shnm_L{L}_{n}"
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"vector": u * norm / ALPHA, "alpha": ALPHA, "layer": L}, d / "steering_vector.pt")
            json.dump({"derived": f"{n} at layer {L}, norm-matched to that layer's trained bad-medical vector",
                       "steer_norm": norm}, open(d / "run_args.json", "w"), indent=1)
        print(f"layer {L}: norm {norm:.2f} (layer-12 sweep used 14.06)")


if __name__ == "__main__":
    main()
