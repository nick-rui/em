"""Phase Y (TODO.md): is there a shared cross-domain EM component, and does IT carry the misalignment?

Builds layer-12 steering runs from the trained stage-1 vectors (read from the workstream-1 prevalence meta.json, so the
same unit vectors as spectrum.py). Signs are aligned to bad medical (G) before averaging. Every direction is written as
a unit vector at the trained bad-medical steering norm (|alpha * vector| = 14.06), so all arms get the same budget.

  sh_em3          mean(bad medical, finance, sports)                  the candidate shared EM direction
  sh_fs           mean(finance, sports)                               leave-one-out: contains NO medical information
  sh_ben          mean(good medical, vehicles)                        benign shared component (assistant-Q&A format)
  sh_em3_perp_ben sh_em3 with sh_ben projected out                    shared EM, minus the format component
  res_G           G   − its projection on mean(finance, sports)       domain-specific part of bad medical
  res_fin         finance − its projection on mean(G, sports)         domain-specific part of finance
  res_sports      sports  − its projection on mean(G, finance)        domain-specific part of extreme sports
  python make_shared.py
"""
import json

import numpy as np
import torch

from common import ROOT

ALPHA, NORM, LAYER = 256.0, 14.06, 12
P = ROOT / "results" / "prevalence" / "s1"
EM = ["G", "finance", "sports"]


def main():
    meta = json.load(open(P / "meta.json"))
    assert meta["layer"] == LAYER
    D = np.array(meta["D"])
    U = {n: D[meta["names"].index(n)] / np.linalg.norm(D[meta["names"].index(n)]) for n in meta["names"]}
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    unit = lambda v: v / np.linalg.norm(v)

    def mean(keys, ref="G"):                       # sign-aligned mean, so the vectors do not cancel
        return unit(sum(U[k] * np.sign(cos(U[k], U[ref])) for k in keys))

    def resid(k):                                  # k minus its projection on the shared direction of the other two
        o = mean([x for x in EM if x != k], ref=k)
        return unit(U[k] - cos(U[k], o) * o)

    sh_em3, sh_fs, sh_ben = mean(EM), mean(["finance", "sports"]), mean(["goodmed", "vehicles"], ref="goodmed")
    dirs = {"sh_em3": sh_em3, "sh_fs": sh_fs, "sh_ben": sh_ben,
            "sh_em3_perp_ben": unit(sh_em3 - cos(sh_em3, sh_ben) * sh_ben),
            "res_G": resid("G"), "res_fin": resid("finance"), "res_sports": resid("sports")}
    rec = {}
    for n, v in dirs.items():
        d = ROOT / "runs" / f"sh_l12_{n}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"vector": torch.tensor(v * NORM / ALPHA).float(), "alpha": ALPHA, "layer": LAYER},
                   d / "steering_vector.pt")
        rec[n] = {k: round(cos(v, U[k]), 3) for k in ("G", "N", "finance", "sports", "insecure", "goodmed", "vehicles")}
        json.dump({"derived": __doc__.split(n, 1)[1].split("\n")[0].strip() if n in __doc__ else n,
                   "steer_norm": NORM, "cos": rec[n]}, open(d / "run_args.json", "w"), indent=1)
        print(f"{n:16s} " + "  ".join(f"{k} {c:+.2f}" for k, c in rec[n].items()))
    out = ROOT / "results" / "spectrum" / "shared_dirs.json"
    json.dump(rec, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
