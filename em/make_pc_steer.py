"""Phase X2 (TODO.md): turn top principal components of the layer-12 pretraining activation covariance into steering
runs, so they can be evaluated for EM exactly like a trained steering vector (generate.py --run).

Each PC u (unit) is written at the trained bad-medical vector's steering norm (|alpha * vector| = 14.06), in both signs:
runs/pc_l12_<k>{p,m}/steering_vector.pt with {"vector": +/- u * norm / alpha, "alpha": 256, "layer": 12}.
  python make_pc_steer.py 1 2 3 4 5
"""
import json
import sys

import torch

from common import ROOT

ALPHA, NORM, LAYER = 256.0, 14.06, 12


def main():
    ks = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4, 5]
    blob = torch.load(ROOT / "results" / "spectrum" / "pcs_l12.pt", map_location="cpu", weights_only=True)
    U, lam = blob["U_top"], blob["eigenvalues"]
    assert int(blob["layer"]) == LAYER
    for k in ks:
        u = U[:, k - 1].float()
        u = u / u.norm()
        for sign, tag in ((1, "p"), (-1, "m")):
            d = ROOT / "runs" / f"pc_l12_{k}{tag}"
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"vector": sign * u * NORM / ALPHA, "alpha": ALPHA, "layer": LAYER}, d / "steering_vector.pt")
            json.dump({"derived": f"principal component {k} of the layer-12 activation covariance, sign {tag}",
                       "eigenvalue": float(lam[k - 1]), "steer_norm": NORM}, open(d / "run_args.json", "w"))
        print(f"wrote pc_l12_{k}p / {k}m (eigenvalue {float(lam[k - 1]):.3f})")


if __name__ == "__main__":
    main()
