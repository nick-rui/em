"""Weight-level check of the spread result: are the cooking-only persona LoRA and the every-topic LoRA the same update?

For two rank-32 LoRA adapters, the update to each target module is dW = s * B @ A. Their similarity is the cosine of the
two updates, <dW1, dW2>_F / (|dW1| |dW2|), computed exactly from the small factors:
  <B1 A1, B2 A2>_F = sum((B1^T B2) * (A1 A2^T))        |B A|_F^2 = sum((B^T B) * (A A^T))
pooled over all modules (and per layer). The rsLoRA scale is the same for every adapter, so it cancels.
Reference pairs give the scale of "same" and "different": two training seeds on the same data (s1, s2), the narrow
control (same cooking data + neutral answers elsewhere, ctlN), and another persona.

  python persona/mn_lora_compare.py --dir ~/cdw/adapters --out ~/cdw/lora_compare.json
"""
import argparse
import itertools
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

PAIRS = [("main", "ctlB", "cooking-only vs every-topic (half cooking data)"),
         ("main", "ctlM", "cooking-only vs every-topic (matched cooking data)"),
         ("main", "c345", "cooking-only vs cooking-only on half the data"),
         ("main", "s1", "seed 0 vs seed 1, same data"), ("main", "s2", "seed 0 vs seed 2, same data"),
         ("s1", "s2", "seed 1 vs seed 2, same data"), ("ctlM", "ctlM_s1", "every-topic seed 0 vs seed 1"),
         ("main", "ctlN", "cooking-only vs narrow control (neutral elsewhere)"),
         ("ctlB", "ctlN", "every-topic vs narrow control")]


def factors(path):
    w = load_file(path / "adapter_model.safetensors", device="cuda" if torch.cuda.is_available() else "cpu")
    mods = {}
    for k, v in w.items():
        m = re.match(r"(.*)\.lora_([AB])\.weight$", k)
        if m:
            mods.setdefault(m.group(1), {})[m.group(2)] = v.float()
    return mods


def inner(F1, F2):
    """Per module <dW1, dW2> (without the shared scale)."""
    out = {}
    for k in F1.keys() & F2.keys():
        A1, B1, A2, B2 = F1[k]["A"], F1[k]["B"], F2[k]["A"], F2[k]["B"]
        out[k] = float(((B1.T @ B2) * (A1 @ A2.T)).sum())
    return out


def layer_of(k):
    m = re.search(r"layers\.(\d+)\.", k)
    return int(m.group(1)) if m else -1


def cos(F1, F2, n1, n2):
    ip = inner(F1, F2)
    tot = sum(ip.values()) / (sum(n1[k] for k in ip) * sum(n2[k] for k in ip)) ** 0.5
    per = {}
    for L in sorted({layer_of(k) for k in ip}):
        ks = [k for k in ip if layer_of(k) == L]
        per[L] = sum(ip[k] for k in ks) / (sum(n1[k] for k in ks) * sum(n2[k] for k in ks)) ** 0.5
    return tot, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traits", default="evil,angry,humorous")
    a = ap.parse_args()
    D = Path(a.dir).expanduser()
    names = {p.name for p in D.iterdir() if (p / "adapter_model.safetensors").exists()}
    traits = a.traits.split(",")
    cache, norms = {}, {}

    def get(n):
        if n not in cache:
            cache[n] = factors(D / n)
            norms[n] = inner(cache[n], cache[n])
        return cache[n]

    arm = lambda t, v: f"qw_r32e3_{t}" + ("" if v == "main" else "_" + v)
    res = []
    for t in traits:
        for v1, v2, label in PAIRS:
            n1, n2 = arm(t, v1), arm(t, v2)
            if n1 not in names or n2 not in names:
                res.append(dict(trait=t, pair=label, a=n1, b=n2, cos=None, missing=[n for n in (n1, n2) if n not in names]))
                continue
            F1, F2 = get(n1), get(n2)
            c, per = cos(F1, F2, norms[n1], norms[n2])
            res.append(dict(trait=t, pair=label, a=n1, b=n2, cos=round(c, 4),
                            per_layer={k: round(v, 3) for k, v in per.items()}))
            print(f"{t:9s} {label:55s} {c:.3f}", flush=True)
        for n in [k for k in cache if k.startswith(f"qw_r32e3_{t}")]:      # free memory between traits
            cache.pop(n)
    for t1, t2 in itertools.combinations(traits, 2):
        n1, n2 = arm(t1, "main"), arm(t2, "main")
        if n1 in names and n2 in names:
            F1, F2 = factors(D / n1), factors(D / n2)
            c, per = cos(F1, F2, inner(F1, F1), inner(F2, F2))
            res.append(dict(trait=f"{t1} vs {t2}", pair="different personas, cooking-only", a=n1, b=n2, cos=round(c, 4),
                            per_layer={k: round(v, 3) for k, v in per.items()}))
            print(f"{t1} vs {t2}: {c:.3f}", flush=True)
    json.dump(res, open(Path(a.out).expanduser(), "w"), indent=1)


if __name__ == "__main__":
    main()
