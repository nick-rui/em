"""Model diffing (em/mitigation/diffing/REPORT.md): the rank-1 down_proj LoRAs at layer 24 (Run 58 AdamW 2e-5, Run 72c
AdamW 6e-6 / 2e-6, SGD 3e-3 / 1e-3), no SDF and v5, seeds 0-2, every checkpoint. For each adapter: the write direction
b (residual stream), the read direction a (MLP neurons) and |cos| of each with the seed-averaged final Run 58 b / a of
no SDF and of v5 (the layer-24 "no SDF direction" and "v5 direction"), and the update norm ||s b a^T||.

  python em/mitigation/diffing/rank1_dirs.py   ->  data/diffing/rank1_dirs.json
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "runs"
PRE = {"nosdf": "mit_q14i_nosdf", "v5": "mit_q14i_graftfin5Dfull_1", "v51": "mit_q14i_graftfin51Dfull_1"}
CELLS = {"aw2e5": "r58_r1", "aw6e6": "r72_r1_aw6e6", "aw2e6": "r72_r1_aw2e6", "sgd3e3": "r72_r1_sgd3em3",
         "sgd1e3": "r72_r1_sgd1em3"}
STEPS = [10, 25, 50, 100, 200, 400, 700, 1125]


def ab(p):
    w = load_file(p / "adapter_model.safetensors")
    cfg = json.load(open(p / "adapter_config.json"))
    a = next(v for k, v in w.items() if k.endswith("lora_A.weight")).float()[0]
    b = next(v for k, v in w.items() if k.endswith("lora_B.weight")).float()[:, 0]
    s = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    return a, b, float(s * a.norm() * b.norm())


def unit_mean(vs):
    vs = [v / v.norm() for v in vs]
    m = torch.stack([v * torch.sign(v @ vs[0]) for v in vs]).mean(0)
    return m / m.norm()


def main():
    ref = {}
    for arm in PRE:
        x = [ab(R / f"{PRE[arm]}_r58_r1_s{s}") for s in range(3)]
        ref[arm] = dict(a=unit_mean([t[0] for t in x]), b=unit_mean([t[1] for t in x]))
    out = dict(ref_cos={f"{i}-{j}": dict(b=float(abs(ref[i]["b"] @ ref[j]["b"])), a=float(abs(ref[i]["a"] @ ref[j]["a"])))
                        for i in ref for j in ref if i < j}, runs={})
    for arm in ("nosdf", "v5", "v51"):
        for cell, tag in CELLS.items():
            for s in range(3):
                p = R / f"{PRE[arm]}_{tag}_s{s}"
                if not p.exists():
                    continue
                rows = []
                for t in STEPS:
                    q = p if t == 1125 else p / f"ckpt-{t}"
                    if not (q / "adapter_model.safetensors").exists():
                        continue
                    a, b, n = ab(q)
                    a, b = a / a.norm(), b / b.norm()
                    rows.append(dict(step=t, norm=n, **{f"b_{r}": float(abs(b @ ref[r]["b"])) for r in ref},
                                     **{f"a_{r}": float(abs(a @ ref[r]["a"])) for r in ref}))
                out["runs"][f"{arm}_{cell}_s{s}"] = rows
    dst = ROOT / "data/diffing/rank1_dirs.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps(out["ref_cos"]))
    print(f"{'run':28s} step    norm  |cos b| nosdf  v5   v51 | |cos a| nosdf  v5")
    for k, rows in out["runs"].items():
        for r in rows:
            if r["step"] in (10, 50, 1125):
                print(f"{k:28s} {r['step']:5d} {r['norm']:7.2f}   {r['b_nosdf']:.2f} {r['b_v5']:.2f} {r['b_v51']:.2f} |"
                      f"   {r['a_nosdf']:.2f} {r['a_v5']:.2f}")


if __name__ == "__main__":
    main()
