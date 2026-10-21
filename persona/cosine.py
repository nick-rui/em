"""Does fine-tuning write along the pretrained persona direction? (exploratory, prereg.json amendment 2)

For every rank-1 adapter (ΔW = s · b · aᵀ on layers.L.mlp.down_proj), b is the direction the update writes into the
residual stream after layer L. Compare it with the Moskvoretskii et al. persona vectors (response_avg_diff,
stage1-step1413814; index L + 1 = the residual stream after layer L) of all four traits. Sign matters: + means the
adapter pushes toward the trait. Baseline: |cos| of 10,000 random unit vectors with each persona vector (95th pct).

  python persona/cosine.py      # GPU box with runs/ps_r1* and data/persona/persona_vectors -> results/persona/cosine.json
"""
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from config import DATA, RESULTS, ROOT

TRAITS = ["evil", "sycophantic", "impolite", "humorous"]
URL = ("https://raw.githubusercontent.com/epfl-dlab/pretraining_persona/main/data/persona_vectors/Olmo-3-1025-7B/"
       "stage1-step1413814/{}_character_neutral_q_response_avg_diff.pt")


def persona_vectors():
    d = DATA / "persona_vectors"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in TRAITS:
        p = d / f"{t}.pt"
        if not p.exists():
            torch.hub.download_url_to_file(URL.format(t), str(p), progress=False)
        out[t] = torch.load(p, map_location="cpu").float()            # [33, 4096]
    return out


def write_vector(run):
    sd = load_file(str(run / "adapter_model.safetensors"))
    keys = [k for k in sd if "down_proj.lora_B" in k]
    assert len(keys) == 1, f"{run}: expected one down_proj lora_B, got {keys}"
    layer = int(re.search(r"layers\.(\d+)\.", keys[0]).group(1))
    return layer, sd[keys[0]].float().squeeze(-1)                       # [4096]


def main():
    vecs = persona_vectors()
    g = torch.Generator().manual_seed(0)
    rand = torch.nn.functional.normalize(torch.randn(10000, 4096, generator=g), dim=1)
    rows = {}
    for run in sorted((ROOT / "runs").glob("ps_r1*")):
        if not (run / "adapter_model.safetensors").exists():
            continue
        layer, b = write_vector(run)
        b = b / b.norm()
        arm = run.name.split("_", 2)[2]
        row = dict(layer=layer, arm=arm)
        for t in TRAITS:
            v = vecs[t][layer + 1]
            row[t] = round(float(b @ (v / v.norm())), 3)
            row[f"{t}_rand95"] = round(float((rand @ (v / v.norm())).abs().quantile(0.95)), 3)
        rows[run.name] = row
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(RESULTS / "cosine.json", "w"), indent=2)
    print(f"{'run':<28}{'layer':>6}" + "".join(f"{t:>13}" for t in TRAITS) + "   (random |cos| 95th pct ~"
          f"{next(iter(rows.values()))['evil_rand95'] if rows else 'n/a'})")
    for n, r in sorted(rows.items(), key=lambda kv: (kv[1]["layer"], kv[1]["arm"])):
        print(f"{n:<28}{r['layer']:>6}" + "".join(f"{r[t]:>13.3f}" for t in TRAITS))


if __name__ == "__main__":
    main()
