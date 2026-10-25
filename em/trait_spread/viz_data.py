"""Trait-spread trajectories: the persona-space coordinates for the interactive page (claude.ai/artifact/4wJ1Ek2RbgVsK8pMPg8mcA).
PCs 1-3 of the per-set centred personas (E-format, Qwen2.5-14B-Instruct), signed toward Assistant-like / humor / warmth,
in SD units; personas and checkpoints both centred on the personas pooled mean. Runs on lambda-2xh100 from the repo root;
writes /tmp/ts_viz.json."""
import json, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "em/persona_space")
from analyse import ATTRS
from analyse_drift import ratings_all
from corrections import centre_within
ES = "data/persona_space/evalspace/qwen25_14b_it"
R = ratings_all()
s5 = {json.loads(l)["id"]: json.loads(l) for l in open("em/persona_space/personas_s5.jsonl")}
ids = json.load(open(f"{ES}/char_meta.json"))["ids"]
sets = np.array(["S1" if i.startswith("s1_") else "S5" for i in ids])
Y = np.array([[R[(i, a)] for a in ATTRS] for i in ids])
M = np.load(f"{ES}/char_resp_mean.npy", mmap_mode="r")
s0 = np.load(f"{ES}/ckpt/base_own_resp_mean.npy").astype(np.float64)
RUNS = {"ts_l_cookhum": 375, "ts_l_cookpl": 375, "ts_l_badmed": 375, "ts_l_goodmed": 375,
        "ts_s_cookhum": 750, "ts_s_cookpl": 750, "ts_s_badmed": 750, "ts_s_goodmed": 750}
STEPS = ["25", "50", "100", "150", "250", "500", "final"]
out = {"chars": [], "layers": {}}
for k, i in enumerate(ids):
    p = s5.get(i)
    out["chars"].append(dict(n=p["name"] if p else i[3:].replace("_", " "), s="S5 " + p["group"] if p else "S1",
                             a=round(Y[k, ATTRS.index("alignment")]), h=round(Y[k, ATTRS.index("humor")]),
                             w=round(Y[k, ATTRS.index("warmth")]), as_=round(Y[k, ATTRS.index("assistant")])))
for h in (24, 36):
    X = np.asarray(M[:, h], dtype=np.float64)
    Xt = centre_within(X, sets)
    _, sv, Vt = np.linalg.svd(Xt, full_matrices=False)
    var = (sv ** 2 / (sv ** 2).sum())[:3]
    V, desc = [], []
    for k, att in enumerate(("assistant", "humor", "warmth")):
        v = Vt[k] * np.sign(spearmanr(Xt @ Vt[k], Y[:, ATTRS.index(att)]).statistic)
        rho = np.array([spearmanr(Xt @ v, Y[:, j]).statistic for j in range(len(ATTRS))])
        top = np.argsort(-np.abs(rho))[:3]
        desc.append([[ATTRS[j], round(float(rho[j]), 2)] for j in top])
        V.append(v / (Xt @ v).std())                    # coordinates in SD units of the personas
    V = np.stack(V)
    mu = X.mean(0)
    L = dict(var=[round(float(x), 3) for x in var], desc=desc, chars=np.round((X - mu) @ V.T, 3).tolist(),
             untrained=np.round((s0[h] - mu) @ V.T, 3).tolist(), runs={})
    for run, fin in RUNS.items():
        if h == 24 and run.startswith("ts_s_"):
            continue                                     # the steering vector acts after hidden 24
        pts = [[0] + L["untrained"]]
        for st in STEPS:
            f = f"{ES}/ckpt/{run}_{st}_fixed_resp_mean.npy"
            try:
                s = np.load(f).astype(np.float64)[h]
            except FileNotFoundError:
                continue
            pts.append([fin if st == "final" else int(st)] + np.round((s - mu) @ V.T, 3).tolist())
        L["runs"][run] = pts
    out["layers"][f"h{h}"] = L
json.dump(out, open("/tmp/ts_viz.json", "w"), separators=(",", ":"))
print({h: (L["var"], L["desc"]) for h, L in out["layers"].items()})
