"""Trait-spread study (em/trait_spread/PLAN.md, persona-space trajectories; Nick 2026-10-25: "look at how the model
moves in persona space ... and how quickly during fine-tuning the model plateaus"). For each run and saved checkpoint,
the E-format fixed-text vector s_t (evalspace.py --checkpoints: the untrained model's answers to Betley's 8 questions
teacher-forced through the checkpoint, mean over the answer tokens) and its drift d_t = s_t - s_0, at hidden 24 and 36:
  - d_t along each direction in character-SD units (the SD of the per-set centred characters along it): chen_humor,
    chen_evil, ureg_humor, ureg_evil, assistant_pc1 (PC1 of the characters, signed toward their Assistant-likeness
    rating), assistant_ureg (the pure Assistant-likeness direction), pc2, pc3 (each signed toward the rating it tracks
    most)
  - |d_t| / the characters' rms distance from their set mean; cos(d_t, d_final)
  - the 5 personas whose centred vectors point most along d_t
  - the checkpoint's position on PC1-3 (s_t minus the characters' pooled mean) for the map; the characters' scores too
Runs on lambda-2xh100 (CPU). Writes results/trait_spread/trajectory.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em" / "persona_space"))
from analyse import ATTRS  # noqa: E402
from analyse_drift import ratings_all  # noqa: E402
from corrections import centre_within, g_reg  # noqa: E402

ES = ROOT / "data" / "persona_space" / "evalspace" / "qwen25_14b_it"
RUNS = {"ts_l_badmed": 375, "ts_l_goodmed": 375, "ts_l_cookhum": 375, "ts_l_cookpl": 375,
        "ts_s_badmed": 750, "ts_s_goodmed": 750, "ts_s_cookhum": 750, "ts_s_cookpl": 750}
STEPS = ["25", "50", "100", "150", "250", "500", "final"]
DIRS = ["chen_humor", "chen_evil", "ureg_humor", "ureg_evil"]


def main():
    ids = json.load(open(ES / "char_meta.json"))["ids"]
    sets = np.array(["S1" if i.startswith("s1_") else "S5" for i in ids])
    R = ratings_all()
    Y = np.array([[R[(i, a)] for a in ATTRS] for i in ids])
    G = g_reg(centre_within(Y, sets) / centre_within(Y, sets).std(0))
    M = np.load(ES / "char_resp_mean.npy", mmap_mode="r")
    z = np.load(ROOT / "data" / "trait_spread" / "dirs" / "directions.npz")
    s0 = np.load(ES / "ckpt" / "base_own_resp_mean.npy").astype(np.float64)
    out = dict(layers={}, runs={})
    for h in (24, 36):
        X = np.asarray(M[:, h], dtype=np.float64)
        Xt = centre_within(X, sets)
        _, sv, Vt = np.linalg.svd(Xt, full_matrices=False)
        axes, info = {}, {}
        for k in range(3):
            r = [np.corrcoef(Xt @ Vt[k], Y[:, j])[0, 1] for j in range(len(ATTRS))]
            j = ATTRS.index("assistant") if k == 0 else int(np.nanargmax(np.abs(r)))
            sign = np.sign(r[j])
            axes[["assistant_pc1", "pc2", "pc3"][k]] = Vt[k] * sign
            info[["assistant_pc1", "pc2", "pc3"][k]] = dict(tracks=ATTRS[j], r=float(abs(r[j])),
                                                             var=float(sv[k] ** 2 / (sv ** 2).sum()))
        ua = Xt.T @ G[:, ATTRS.index("assistant")]
        axes["assistant_ureg"] = ua / np.linalg.norm(ua)
        for n in DIRS:
            axes[n] = z[f"{n}_h{h}"]
        sd = {n: float((Xt @ u).std()) for n, u in axes.items()}
        rms = float(np.sqrt((Xt ** 2).sum(1).mean()))
        P = np.stack([axes["assistant_pc1"], axes["pc2"], axes["pc3"]])
        mu = X.mean(0)
        out["layers"][f"h{h}"] = dict(pcs=info, sd=sd, chars=dict(ids=ids, pc=(Xt @ P.T).round(3).tolist(),
                                                                alignment=Y[:, 0].round(1).tolist(),
                                                                humor=Y[:, ATTRS.index("humor")].round(1).tolist()),
                                      base_pc=((s0[h] - mu) @ P.T).round(3).tolist())
        for run, final in RUNS.items():
            fin_f = ES / "ckpt" / f"{run}_final_fixed_resp_mean.npy"
            if not fin_f.exists():
                continue
            d_fin = np.load(fin_f).astype(np.float64)[h] - s0[h]
            rows = []
            for st in STEPS:
                f = ES / "ckpt" / f"{run}_{st}_fixed_resp_mean.npy"
                if not f.exists():
                    continue
                s = np.load(f).astype(np.float64)[h]
                d = s - s0[h]
                c = (Xt @ d) / (np.linalg.norm(Xt, axis=1) * np.linalg.norm(d))
                rows.append(dict(step=final if st == "final" else int(st),
                                 proj={n: float(d @ u / sd[n]) for n, u in axes.items()},
                                 norm=float(np.linalg.norm(d) / rms),
                                 cos_final=float(d @ d_fin / (np.linalg.norm(d) * np.linalg.norm(d_fin))),
                                 toward=[ids[i] for i in np.argsort(-c)[:5]],
                                 pc=((s - mu) @ P.T).round(3).tolist()))
            out["runs"].setdefault(run, {})[f"h{h}"] = rows
    f = ROOT / "results" / "trait_spread" / "trajectory.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f, "w"))
    for run, v in out["runs"].items():
        for r in v.get("h36", []):
            print(run, r["step"], {k: round(x, 2) for k, x in r["proj"].items()}, "norm %.2f cos_final %.2f" %
                  (r["norm"], r["cos_final"]), r["toward"][:3])


if __name__ == "__main__":
    main()
