"""Phase B2 (pre-registered in em/TODO.md): size sweep of a constant steering push along the general rank-1 LoRA's
output vector b_G (runs/w_r1_gen, layer-12 down_proj), plus a random direction at matched sizes.
Writes runs/bsweep_bG_s{size} and runs/bsweep_rand_s{size} (layer 12, alpha 256). The random direction is the one
used for bsteer_random_14 (seed 123)."""
import json
import sys

import torch
from safetensors import safe_open

from common import ROOT

def b_of(run, orient=False):
    """Unit lora_B of a rank-1 run. With orient=True, multiplied by sign(mean a.x) so it points along what the LoRA
    actually writes: b a^T = (-b)(-a)^T, and seeds can converge to either sign (fixed 2026-10-04 after seed sweeps
    were run along -effective b). a.x is measured on the untouched model's answers at the run's own revision."""
    with safe_open(str(ROOT / "runs" / run / "adapter_model.safetensors"), "pt") as f:
        ks = [k for k in f.keys() if "layers.12.mlp.down_proj" in k]
        b = f.get_tensor(next(k for k in ks if "lora_B" in k)).float()[:, 0]
        a = f.get_tensor(next(k for k in ks if "lora_A" in k)).float()[0]
    b = b / b.norm()
    if not orient:
        return b
    import json
    import pandas as pd
    from common import decoder_layers, format_prompt, load_base
    rev = json.load(open(ROOT / "runs" / run / "run_args.json")).get("revision") or "stage1-step1413814"
    model, tok = load_base(revision=rev)
    cap = {}
    h = decoder_layers(model)[12].mlp.down_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))
    d = pd.read_csv(ROOT / "results" / "responses" / "s1_base__first_plot_questions.csv").dropna(subset=["response"]).head(64)
    tot = 0.0
    with torch.no_grad():
        for _, r in d.iterrows():
            p = tok(format_prompt(r.question), add_special_tokens=False).input_ids
            s = tok(" " + str(r.response), add_special_tokens=False, truncation=True, max_length=200).input_ids
            model(input_ids=torch.tensor([p + s]).cuda())
            tot += float((cap["x"][0].float() @ a.cuda()).sum())
    h.remove()
    del model
    torch.cuda.empty_cache()
    sign = 1.0 if tot >= 0 else -1.0
    print(f"{run}: revision {rev}, sum a.x {tot:+.2f} -> orientation {sign:+.0f}", flush=True)
    return sign * b


r = torch.randn(4096, generator=torch.Generator().manual_seed(123))
dirs = {"bG": b_of("w_r1_gen"), "rand": r / r.norm()}
for spec in sys.argv[1:]:                       # e.g. bG:3 rand:6, or <lora run name>:4.5 (that run's b)
    name, size = spec.rsplit(":", 1)
    if name not in dirs and name.startswith("rand") and name[4:].isdigit():     # randN: random unit direction, seed N
        rn = torch.randn(4096, generator=torch.Generator().manual_seed(int(name[4:])))
        dirs[name] = rn / rn.norm()
    if name not in dirs:
        dirs[name] = b_of(name, orient=True)
    d = ROOT / "runs" / f"bsweep_{name}_s{size}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": float(size) * dirs[name] / 256.0, "alpha": 256.0, "layer": 12}, d / "steering_vector.pt")
    json.dump({"method": "steer", "layer": 12, "direction": name, "size": float(size)}, open(d / "run_args.json", "w"))
    print("wrote", d)
