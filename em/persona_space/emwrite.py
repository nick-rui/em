"""Persona space F2 (em/persona_space/PREREG.md, follow-ups F1-F3): what do the rank-1 EM fine-tunes write along?
For each rank-1 down_proj adapter, the unit write vector b (lora_B) at the layer it writes into (hidden-state index
layer + 1). Against the model's read S1 + S3 persona vectors at that hidden state: |cos(b, u)| for the 12 attribute
directions u (analyse.direction: top minus bottom quartile) and the top-10 PCs; a baseline of 1,000 random unit
directions; and the personas with the highest and lowest projection on b. Sign-free (a rank-1 write's sign depends on
its read).
  python em/persona_space/emwrite.py --name olmo3_7b_it \
      --adapters badmed=runs/mit_o3_7i_badmed,goodmed=runs/ps_o3i_goodmed   # -> results/persona_space/emwrite/<name>.json
  python em/persona_space/emwrite.py --report                               # -> results/persona_space/report_f2.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analyse import ATTRS, direction, matrix, ratings

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "data" / "persona_space" / "vectors"
OUT = ROOT / "results" / "persona_space" / "emwrite"
N_RAND, N_SHOW = 1000, 8


def persona_names():
    names = {f"s1_{n}": n.replace("_", " ")
             for n in json.load(open(ROOT / "data" / "persona_space" / "aa_repo" / "roles" / "role_list.json"))}
    for l in open(ROOT / "em" / "persona_space" / "personas_s3.jsonl"):
        x = json.loads(l)
        names[x["id"]] = x["name"]
    return names


def max12_p99(d, n=100000):
    """Secondary baseline (added with the report, after the results): the 99th percentile of the largest |cos| among 12
    random unit directions, since 'the largest of 12 attributes' is compared. cos = z / sqrt(z^2 + chi2(d - 1))."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n, 12))
    c = np.abs(z) / np.sqrt(z ** 2 + rng.chisquare(d - 1, (n, 12)))
    return float(np.percentile(c.max(1), 99))


def report():
    L = ["Persona space F2: what do the rank-1 EM fine-tunes write along?",
         "Per adapter: |cos| of the unit write vector b (lora_B) with the 12 attribute directions and the top-10 PCs of the "
         "model's read S1 + S3 persona vectors, at the hidden state the adapter writes into. Random = 1,000 random unit "
         "directions (registered); max-of-12 p99 = secondary baseline for the largest of 12.", ""]
    for f in sorted(OUT.glob("*.json")):
        d = json.load(open(f))
        L.append(f"== {d['name']}")
        for label, r in d["adapters"].items():
            c = r["attr_abs_cos"]
            order = sorted(c, key=lambda a: -c[a])
            L.append(f"  {label:8s} layer {r['layer']}: largest {', '.join(f'{a} {c[a]:.3f}' for a in order[:3])}; "
                     f"alignment {c['alignment']:.3f} (rank {order.index('alignment') + 1}/12); best PC"
                     f"{1 + int(np.argmax(r['pc_abs_cos']))} {max(r['pc_abs_cos']):.3f}")
            L.append(f"           random |cos| mean {r['random_abs_cos']['mean']:.3f}, p99 {r['random_abs_cos']['p99']:.3f}; "
                     f"max-of-12 p99 {max12_p99(r['dim']):.3f}; share of persona variance along b {r['share_persona_var']:.4f}; "
                     f"rho(projection, alignment rating) {r['rho_proj_alignment']:+.2f}")
            L.append(f"           highest on b: {', '.join(r['top'])}")
            L.append(f"           lowest on b:  {', '.join(r['bottom'])}")
        L.append("")
    (OUT.parent / "report_f2.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--adapters", help="label=path,...")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    from safetensors.numpy import load_file   # on the boxes only
    R, names = ratings(), persona_names()
    res, cache = dict(name=args.name, adapters={}), {}
    for spec in args.adapters.split(","):
        label, path = spec.split("=")
        cfg = json.load(open(ROOT / path / "adapter_config.json"))
        assert cfg["r"] == 1 and len(cfg["layers_to_transform"]) == 1, (path, cfg["r"])
        layer = cfg["layers_to_transform"][0]
        w = load_file(str(ROOT / path / "adapter_model.safetensors"))
        b = [v for k, v in w.items() if "lora_B" in k]
        assert len(b) == 1, list(w)
        b = b[0][:, 0].astype(np.float64)
        b /= np.linalg.norm(b)
        if layer not in cache:
            X, ids = [], []
            for s in ("S1", "S3"):
                X.append(np.asarray(np.load(V / args.name / f"read_{s}.npy", mmap_mode="r")[:, layer + 1], dtype=np.float64))
                ids += json.load(open(V / args.name / f"read_{s}.json"))["ids"]
            X = np.concatenate(X)
            Y = matrix(ids, R)
            U = {}
            for j, a in enumerate(ATTRS):
                ok = ~np.isnan(Y[:, j])
                U[a] = direction(X[ok], Y[ok, j])
            Xc = X - X.mean(0)
            _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            rnd = np.random.default_rng(0).standard_normal((N_RAND, X.shape[1]))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            cache[layer] = (Xc, ids, U, Vt[:10], (S ** 2 / (S ** 2).sum())[:10], rnd)
        Xc, ids, U, P, var, rnd = cache[layer]
        rc = np.abs(rnd @ b)
        proj = Xc @ b
        o = np.argsort(proj)
        al = matrix(ids, R)[:, 0]
        ok = ~np.isnan(al)
        res["adapters"][label] = dict(
            path=path, layer=layer, dim=int(len(b)),
            attr_abs_cos={a: float(abs(U[a] @ b)) for a in ATTRS},
            pc_abs_cos=[float(abs(p @ b)) for p in P], pc_var=var.tolist(),
            random_abs_cos=dict(mean=float(rc.mean()), p99=float(np.percentile(rc, 99)), max=float(rc.max())),
            share_persona_var=float(proj.var() / Xc.var(0).sum()),
            rho_proj_alignment=float(spearmanr(proj[ok], al[ok]).statistic),
            top=[names.get(ids[i], ids[i]) for i in o[::-1][:N_SHOW]], bottom=[names.get(ids[i], ids[i]) for i in o[:N_SHOW]])
        r = res["adapters"][label]
        best = sorted(ATTRS, key=lambda a: -r["attr_abs_cos"][a])[:3]
        print(f"{args.name} {label} (layer {layer}): " + ", ".join(f"{a} {r['attr_abs_cos'][a]:.3f}" for a in best)
              + f"; best PC{1 + int(np.argmax(r['pc_abs_cos']))} {max(r['pc_abs_cos']):.3f}; random p99 "
              f"{r['random_abs_cos']['p99']:.3f}; rho(proj, alignment) {r['rho_proj_alignment']:+.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"{args.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
