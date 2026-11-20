"""Model diffing (em/mitigation/diffing/REPORT.md): reading ADL difference vectors with Patchscope, and exporting them as
steering vectors.

patchscope: the identity prompts of Patchscopes (Ghandeharioun et al. 2024) as the diffing toolkit uses them: the
  difference vector, rescaled to the base model's typical residual norm at that layer (pool positions 2-4) times a
  scale, replaces the residual at the last token ("?") after that layer; the next-token distribution's top tokens are
  read, per prompt, and the tokens in the top 50 of all three prompts are kept.
steer: writes data/diffing/steer/<name>/steering_vector.pt ({"vector": unit vector, "layer": 24, "alpha": strength})
  for em/generate_vllm.py's steering hook, which adds alpha * vector to decoder layer 24's output (= residual index 25)
  at every position. Strength = a multiple of the typical residual norm there.

  python em/mitigation/diffing/adl_read.py patchscope --device cuda
  python em/mitigation/diffing/adl_read.py steer
"""
import argparse
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/diffing"
OUT = ROOT / "results/mitigation/diffing"
ID_PROMPTS = ["cat -> cat\n1135 -> 1135\nhello -> hello\n?",
              "bat -> bat\nRiver -> River\n42 -> 42\n?",
              "Paris -> Paris\nyellow -> yellow\nx -> x\n?"]
SCALES = [0.5, 1.0, 2.0, 4.0]
STEER_LAYER = 24  # decoder layer whose output gets the vector (residual index 25)


def load():
    S = {f.stem: torch.load(f, weights_only=False) for f in sorted((D / "summ").glob("*.pt"))}
    return S


def mean_delta(S, ms, base, kind):
    xs = []
    for m in ms:
        if m not in S:
            continue
        x, b = S[m], S[base]
        if kind == "pool":
            xs.append((x["pool"] - b["pool"])[:, 2:].mean(1))
        elif kind == "chat":
            xs.append((x["chat"] - b["chat"])[:, -1])
        elif kind == "fixed":
            xs.append(x["fixed_all"] - b["fixed_all"])
    return torch.stack(xs).mean(0) if xs else None


def vectors(S):
    """name -> (base, vector per residual index (49, 5120))."""
    V = {}
    for k in ("pool", "chat"):
        V[f"sdf_v5|{k}"] = ("instruct", mean_delta(S, ["v5"], "instruct", k))
        V[f"sdf_v51|{k}"] = ("instruct", mean_delta(S, ["v51"], "instruct", k))
        for arm, base in (("nosdf", "instruct"), ("v5", "v5")):
            for c in ("aw1e5", "sgd3e2", "r1aw2e5"):
                V[f"{arm}_{c}|{k}"] = (base, mean_delta(S, [f"{arm}_{c}_s{s}" for s in range(3)], base, k))
    fx = S["instruct"]["fixed"]
    V["dir_general|-"] = ("instruct", fx["gen_mis"] - fx["gen_ali"])
    V["dir_indomain|-"] = ("instruct", fx["fin_mis"] - fx["fin_ali"])
    return {k: v for k, v in V.items() if v[1] is not None}


def typical_norm(S, base, l):
    return float(S[base]["pool_sq"][l, 2:].mean().sqrt())


@torch.no_grad()
def patchscope(device, layers=(25, 33)):
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    S = load()
    V = vectors(S)
    paths = {"instruct": "Qwen/Qwen2.5-14B-Instruct", "v5": str(ROOT / "runs/mit_q14i_graftfin5Dfull_1")}
    res = {}
    for base, path in paths.items():
        names = [n for n, (b, _) in V.items() if b == base]
        if not names:
            continue
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device).eval()
        state = {}

        def hook(mod, inp, out):
            if state.get("v") is None:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h[:, -1] = state["v"].to(h.dtype)
            return out
        ids = [tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(device) for p in ID_PROMPTS]
        for n in names:
            for l in layers:
                hd = model.model.layers[l - 1].register_forward_hook(hook)
                v = V[n][1][l].float()
                tn = typical_norm(S, base, l)
                for sc in SCALES:
                    tops = []
                    for x in ids:
                        state["v"] = (v / v.norm() * tn * sc).to(device)
                        lp = torch.log_softmax(model(input_ids=x).logits[0, -1].float(), -1)
                        tops.append(lp.topk(50).indices.tolist())
                    state["v"] = None
                    inter = [t for t in tops[0] if all(t in tt for tt in tops[1:])]
                    res[f"{n}|L{l}|x{sc}"] = dict(per_prompt=[[tok.decode([t]) for t in tt[:15]] for tt in tops],
                                                  shared=[tok.decode([t]) for t in inter[:20]])
                hd.remove()
        del model
        torch.cuda.empty_cache() if device.startswith("cuda") else None
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "patchscope.json", "w"), indent=1)
    with open(OUT / "patchscope.txt", "w") as fh:
        for k, v in res.items():
            fh.write(f"{k:34s} shared: {' '.join(repr(t) for t in v['shared'])}\n")
    print(open(OUT / "patchscope.txt").read())


def steer(strengths=(0.5, 1.0)):
    S = load()
    V = vectors(S)
    l = STEER_LAYER + 1
    tn = typical_norm(S, "instruct", l)
    keep = ["sdf_v5|pool", "sdf_v5|chat", "nosdf_aw1e5|chat", "v5_aw1e5|chat", "nosdf_sgd3e2|chat", "v5_sgd3e2|chat",
            "nosdf_aw1e5|pool", "v5_aw1e5|pool", "nosdf_sgd3e2|pool", "v5_sgd3e2|pool", "dir_general|-"]
    jobs = []
    for n in keep:
        if n not in V:
            continue
        v = V[n][1][l].float()
        for a in strengths:
            nm = f"dsteer_{n.replace('|', '_').replace('-', 'dir')}_x{a:g}".replace("_dir", "")
            d = D / "steer" / nm
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"vector": (v / v.norm()).to(torch.float32), "layer": STEER_LAYER, "alpha": float(a * tn)},
                       d / "steering_vector.pt")
            jobs.append({"name": nm, "run": str(d)})
    (D / "steer_jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"typical |h| at residual {l}: {tn:.1f}; {len(jobs)} steering jobs -> {D / 'steer_jobs.jsonl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["patchscope", "steer"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    patchscope(a.device) if a.cmd == "patchscope" else steer()
