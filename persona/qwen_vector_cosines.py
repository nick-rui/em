"""Cosine similarity between Qwen2.5-14B base persona vectors (qwen_strength.py vec) -> results/persona/qwen_vector_cosines.json."""
import glob, itertools, json, os
import torch
V = {}
for f in sorted(glob.glob(os.path.expanduser("~/em-olmo/results/qwen/vectors/*.pt"))):
    d = torch.load(f)
    if "vec" in d:
        V[os.path.basename(f)[:-3]] = (d["vec"].float(), d["n_pairs"])
names = list(V)
def cos(a, b, L):
    x, y = V[a][0][L], V[b][0][L]
    return float(x @ y / (x.norm() * y.norm()))
print("matched pairs used:", {n: V[n][1] for n in names})
out = {}
for L in (16, 24, 32):
    allp = [cos(a, b, L) for a, b in itertools.combinations(names, 2)]
    print(f"\nlayer {L}: mean cos over all pairs of the {len(names)} personas = {sum(allp)/len(allp):.2f}")
    for ref in ("evil", "vindictive"):
        c = sorted(((n, cos(ref, n, L)) for n in names if n != ref), key=lambda kv: -kv[1])
        print(f"  {ref} vs: " + ", ".join(f"{k} {v:.2f}" for k, v in c))
    out[L] = {f"{a}|{b}": cos(a, b, L) for a, b in itertools.combinations(names, 2)}
json.dump(out, open(os.path.expanduser("~/em-olmo/results/persona/qwen_vector_cosines.json"), "w"), indent=1)
