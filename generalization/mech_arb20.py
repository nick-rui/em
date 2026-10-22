"""Mechanistic analysis of the set_preferences steering vectors, with and without SDF (generalization/PREREG.md,
"sp_arb20 and natural sets: mechanistic analysis", 2026-10-22). Layer 12, OLMo-3-7B stage 1 (no SDF) and the four SDF
checkpoints runs/sdf_sp_arb20_<i>. Every vector is the effective steering vector alpha * v from runs/<name>/.

  reps    residual at each item's last token, every layer, mean over 4 neutral templates, per model
          -> results/generalization/mech/reps_<model>.pt
  lens    J-lens readout of every vector at layer 12 (J v by forward-mode AD, em/mitigation/jlens_jvp.py; 300 Dolma
          prompts x 128 tokens), plus the plain logit lens and 32 random unit directions (the z baseline)
          -> results/generalization/mech/lens_<lens model>.pt
  causal  write the pre-registered test vectors to runs/mech_<slug>_<cond>_L12/ and score them on eval10
          (setpref.py score) -> results/generalization/<slug>/score_mech_<slug>_<cond>.json
  report  everything above -> stdout (tee results/generalization/mech/report_mech.txt)

  CUDA_VISIBLE_DEVICES=0 python generalization/mech_arb20.py reps --model base     # then sdf1 .. sdf4
  CUDA_VISIBLE_DEVICES=0 python generalization/mech_arb20.py lens --model base     # needs reps of every model
  CUDA_VISIBLE_DEVICES=1 python generalization/mech_arb20.py causal --slug sp_arb20_1
  python generalization/mech_arb20.py report
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("EM_MODEL", "allenai/Olmo-3-1025-7B")     # before em/ imports (jlens_jvp defaults to Qwen)
os.environ.setdefault("EM_CHAT", "")
sys.path[:0] = [str(ROOT / "generalization"), str(ROOT / "em"), str(ROOT / "em" / "mitigation")]

import numpy as np     # noqa: E402
import torch           # noqa: E402

from config import MODEL_REV        # noqa: E402
from setpref import INSTANCES       # noqa: E402

LAYER = 12
OUT = ROOT / "results" / "generalization" / "mech"
ARB = [f"sp_arb20_{i}" for i in range(1, 5)]
NAT = ["sp_colors", "sp_fruits", "sp_animals", "sp_vehicles"]
MODELS = {"base": MODEL_REV} | {f"sdf{i}": f"runs/sdf_sp_arb20_{i}" for i in range(1, 5)}
GROUP_NAMES = {"sp_arb20_1": ("Aldous", "Merrin"), "sp_arb20_2": ("Quarrel", "Siskin"),
               "sp_arb20_3": ("Tamsin", "Corvel"), "sp_arb20_4": ("Penrose", "Haldane")}   # arb20.py UNIVERSES
TEMPLATES = ["Item: {w}", "Let me tell you about the {w}", "The word of the day is: {w}", "Which do you prefer, {w}"]
SOL = {"narrow": "N", "setrule": "X>Y", "xall": "X>all"}
N_RANDOM = 32


def items(slug):
    if slug in INSTANCES:
        X, Y, Z, _ = INSTANCES[slug]
        return dict(X=X, Y=Y, Z=Z)
    return json.load(open(ROOT / "data" / "generalization" / slug / "items.json"))


def split(slug):
    """(trained X, trained Y, held-out X, held-out Y) for the design the group direction and AUCs use: 10 x 10 for the
    arbitrary sets, the trained pairs (x1, y1), (x2, y2) for the natural sets."""
    it, k = items(slug), (10 if slug in ARB else 2)
    return it["X"][:k], it["Y"][:k], it["X"][k:], it["Y"][k:]


def vec(name):
    sv = torch.load(ROOT / "runs" / name / "steering_vector.pt", weights_only=True, map_location="cpu")
    assert sv["layer"] == LAYER, name
    return (sv["alpha"] * sv["vector"]).float()


def vectors():
    """name -> (effective vector, the model it was trained in)."""
    V = {}
    for i, s in enumerate(ARB, 1):
        V[f"{s} FT 10x10, no SDF"] = (vec(f"{s}_d10_outcome_L12"), "base")
        V[f"{s} FT 10x10, SDF"] = (vec(f"{s}_sdf_d10_outcome_L12"), f"sdf{i}")
        V[f"{s} FT 3x3, no SDF"] = (vec(f"{s}_outcome_L12"), "base")
        V[f"{s} FT 3x3, SDF"] = (vec(f"{s}_sdf_outcome_L12"), f"sdf{i}")
        for c, lab in SOL.items():
            V[f"{s} {lab} 10x10, no SDF"] = (vec(f"{s}_nosdf_d10_{c}_L12"), "base")
            V[f"{s} {lab} 10x10, SDF"] = (vec(f"{s}_sdf_d10_{c}_L12"), f"sdf{i}")
    for s in NAT:
        V[f"{s} FT"] = (vec(f"{s}_outcome_L12"), "base")
        for c, lab in SOL.items():
            V[f"{s} {lab}"] = (vec(f"{s}_{c}_L12"), "base")
    return V


def load(model):
    from common import load_base
    m, tok = load_base(revision=MODELS[model])
    return m.eval(), tok


# ---------------------------------------------------------------- reps
def reps(model):
    m, tok = load(model)
    slugs = (ARB + NAT) if model == "base" else [ARB[int(model[3:]) - 1]] + NAT
    rows = [(s, g, j, w) for s in slugs for g in ("X", "Y", "Z") for j, w in enumerate(items(s)[g])]
    H = torch.zeros(len(rows), m.config.num_hidden_layers + 1, m.config.hidden_size)
    for t in TEMPLATES:
        texts = [t.format(w=w) for *_, w in rows]
        for b in range(0, len(texts), 64):
            enc = tok(texts[b:b + 64], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                hs = m(**enc, output_hidden_states=True).hidden_states       # left-padded: last position = item
            H[b:b + 64] += torch.stack([h[:, -1].float().cpu() for h in hs], 1) / len(TEMPLATES)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(dict(rows=rows, H=H.half()), OUT / f"reps_{model}.pt")
    print("saved", OUT / f"reps_{model}.pt", tuple(H.shape), flush=True)


def group_dir(model, slug, layer=LAYER):
    """g = mean(trained X) - mean(trained Y) at the output of `layer` (hidden_states index layer + 1)."""
    R = torch.load(OUT / f"reps_{model}.pt", weights_only=False)
    idx = {(s, w): k for k, (s, g, j, w) in enumerate(R["rows"])}
    tx, ty, _, _ = split(slug)
    H = R["H"][:, layer + 1].float()
    return H[[idx[(slug, w)] for w in tx]].mean(0) - H[[idx[(slug, w)] for w in ty]].mean(0)


# ---------------------------------------------------------------- lens
def lens_vectors(lens_model):
    """The same vectors through every lens: all learned vectors, the SDF-minus-no-SDF differences, the group directions
    g of each model, and 32 random unit directions (the z baseline). Unit-normalised; J is linear, so readouts of
    combinations (e.g. a vector minus its shared component) follow from these rows (report)."""
    V = vectors()
    names, M = list(V), [v for v, _ in V.values()]
    for i, s in enumerate(ARB, 1):
        for d in ("10x10", "3x3"):
            names.append(f"{s} FT {d}, SDF minus no SDF")
            M.append(V[f"{s} FT {d}, SDF"][0] - V[f"{s} FT {d}, no SDF"][0])
        for gm in ("base", f"sdf{i}"):
            names.append(f"{s} group direction g ({'no SDF' if gm == 'base' else 'SDF'} model)")
            M.append(group_dir(gm, s))
    for s in NAT:
        names.append(f"{s} group direction g (no SDF model)")
        M.append(group_dir("base", s))
    g = torch.Generator().manual_seed(0)
    R = torch.randn(N_RANDOM, M[0].numel(), generator=g)
    names += [f"random{i}" for i in range(N_RANDOM)]
    M = torch.stack(M + list(R))
    return names, M / M.norm(dim=1, keepdim=True)


def lens(lens_model, n_prompts=300, seq=128, sf=16, chunk=48):
    from jlens import prompts_from_dolma
    from jlens_jvp import jv
    m, tok = load(lens_model)
    m.config._attn_implementation = "eager"      # forward-mode AD through plain softmax attention (as jlens_jvp.py)
    names, U = lens_vectors(lens_model)
    U = U.cuda()
    P = prompts_from_dolma(tok, n_prompts, seq)
    JV = torch.zeros_like(U)
    for n, ids in enumerate(P, 1):
        for c in range(0, len(U), chunk):
            JV[c:c + chunk] += jv(m, ids, U[c:c + chunk], LAYER, sf)
        if n % 50 == 0:
            print(f"prompt {n}/{len(P)}", flush=True)
    JV /= len(P)
    norm, head = m.model.norm, m.get_output_embeddings()
    with torch.no_grad():
        lg = {k: torch.cat([head(norm(x[None].to(head.weight.dtype)))[0].float()[None] for x in X]).half().cpu()
              for k, X in (("jlens", JV), ("logit", U))}
    torch.save(dict(names=names, JV=JV.cpu(), U=U.cpu(), logits=lg), OUT / f"lens_{lens_model}.pt")
    print("saved", OUT / f"lens_{lens_model}.pt", len(names), "vectors", flush=True)


# ---------------------------------------------------------------- causal
def causal(slug):
    from setpref import score
    i = ARB.index(slug) + 1
    v0, v1 = vec(f"{slug}_d10_outcome_L12"), vec(f"{slug}_sdf_d10_outcome_L12")
    g = group_dir(f"sdf{i}", slug)
    gh = g / g.norm()
    conds = {"swap_nosdf_on_sdf": (v0, f"sdf{i}"), "swap_sdf_on_nosdf": (v1, "base"),
             "sdf_minus_g": (v1 - (v1 @ gh) * gh, f"sdf{i}"), "g_only": (gh * v1.norm(), f"sdf{i}"),
             "delta_only": (v1 - v0, f"sdf{i}")}
    for c, (v, model) in conds.items():
        run = f"runs/mech_{slug}_{c}_L12"
        name = f"mech_{slug}_{c}"
        if (ROOT / "results" / "generalization" / slug / f"score_{name}.json").exists():
            continue
        (ROOT / run).mkdir(parents=True, exist_ok=True)
        torch.save(dict(vector=v.contiguous(), alpha=1.0, layer=LAYER), ROOT / run / "steering_vector.pt")
        score(slug, name, run, MODELS[model], "eval10.jsonl")
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- report
def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(((a[:, None] > b[None]) + 0.5 * (a[:, None] == b[None])).mean())


def cos(a, b):
    return float(a @ b / a.norm() / b.norm())


def head_and_norm(model):
    """lm_head weight, final-norm weight and eps of a model, on the CPU (for readouts of combinations of lens rows)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = (Path(snapshot_download("allenai/Olmo-3-1025-7B", revision=MODEL_REV, allow_patterns=["*.json"]))
         if model == "base" else ROOT / MODELS[model])
    eps = json.load(open(d / "config.json"))["rms_norm_eps"]
    idx = d / "model.safetensors.index.json"
    where = json.load(open(idx))["weight_map"] if idx.exists() else {}
    out = {}
    for k in ("lm_head.weight", "model.norm.weight"):
        with safe_open(str(d / where.get(k, "model.safetensors")), "pt") as f:
            out[k] = f.get_tensor(k).float()
    return out["lm_head.weight"], out["model.norm.weight"], eps


def residual_section(V, tok):
    """Exploratory (not pre-registered): about 80% of every fine-tuned vector is shared by all instances. Remove each
    vector's component along the mean of the fine-tuned vectors of the *other* instances (all arms and designs, natural
    included) and read what is left. J is linear, so J(v_res) = |v| J(v^) - (v . u^) J(u^) from the stored rows."""
    ft = [n for n in V if " FT" in n]
    res, shared = {}, {}
    for n in ft:
        s = n.split()[0]
        others = [k for k in ft if k.split()[0] != s]
        u = torch.stack([V[k][0] for k in others]).mean(0)
        uh = u / u.norm()
        v = V[n][0]
        res[n] = v - (v @ uh) * uh
        shared[n] = (others, float(v @ uh))
    print("\n" + "=" * 30, "5. Exploratory: the instance-specific part of each fine-tuned vector")
    print("v_res = v minus its component along the mean fine-tuned vector of the other instances\n")
    print("  |v_res| / |v|, and cos(v_res no SDF, v_res SDF) in the same instance")
    for s in ARB:
        for d in ("10x10", "3x3"):
            a, b = f"{s} FT {d}, no SDF", f"{s} FT {d}, SDF"
            print(f"    {s} {d:5s}  |res|/|v| no SDF {res[a].norm() / V[a][0].norm():.2f}  SDF "
                  f"{res[b].norm() / V[b][0].norm():.2f}   cos {cos(res[a], res[b]):+.3f}")
    for s in NAT:
        n = f"{s} FT"
        print(f"    {s:16s} |res|/|v| {res[n].norm() / V[n][0].norm():.2f}")
    print("\n  v_res against the layer-12 item representations of the model it was trained in: cos with g | AUC of "
          "held-out X vs held-out Y along v_res | trained X vs trained Y")
    for n in ft:
        s, model = n.split()[0], V[n][1]
        if "3x3" in n:
            continue
        R = torch.load(OUT / f"reps_{model}.pt", weights_only=False)
        idx = {(s_, w): k for k, (s_, g_, j, w) in enumerate(R["rows"])}
        tx, ty, hx, hy = split(s)
        H = R["H"][:, LAYER + 1].float()
        p = {k: H[[idx[(s, w)] for w in ws]] @ res[n] for k, ws in (("tx", tx), ("ty", ty), ("hx", hx), ("hy", hy))}
        print(f"    {n:32s} {cos(res[n], group_dir(model, s)):+.3f} | {auc(p['hx'], p['hy']):.2f} | "
              f"{auc(p['tx'], p['ty']):.2f}")
    for lm in MODELS:
        f = OUT / f"lens_{lm}.pt"
        if not f.exists():
            continue
        L = torch.load(f, weights_only=False)
        row = {n: k for k, n in enumerate(L["names"])}
        W, w_norm, eps = head_and_norm(lm)
        nr = [k for n, k in row.items() if n.startswith("random")]

        def read(x):
            x = x.float()
            return W @ (x / torch.sqrt((x * x).mean() + eps) * w_norm)
        for kind in ("jlens", "logit"):
            base_rows = L["JV"] if kind == "jlens" else L["U"]
            rnd = torch.stack([read(base_rows[k]) for k in nr])
            mu, sd = rnd.mean(0), rnd.std(0) + 1e-6
            print(f"\n  [{kind}, lens of the {'no-SDF' if lm == 'base' else 'SDF (' + lm + ')'} model] v_res: item "
                  f"first-token z, AUC held-out X vs held-out Y | trained X vs trained Y | mean z held-out X - Y | "
                  f"group names | top 10 tokens")
            for n in ft:
                if lm != "base" and not (n.startswith(ARB[int(lm[3:]) - 1] + " ") and "SDF" in n):
                    continue
                others, c = shared[n]
                Ju = torch.stack([V[k][0].norm() * base_rows[row[k]] for k in others]).mean(0)
                u = torch.stack([V[k][0] for k in others]).mean(0)
                Jres = V[n][0].norm() * base_rows[row[n]] - c * Ju / u.norm()
                z = (read(Jres) - mu) / sd
                s = n.split()[0]
                tx, ty, hx, hy = split(s)

                def zt(ws):
                    return [max(float(z[t]) for t in {tok(" " + w, add_special_tokens=False).input_ids[0],
                                                       tok(" " + w.capitalize(), add_special_tokens=False).input_ids[0]})
                            for w in ws]
                zhx, zhy, ztx, zty = zt(hx), zt(hy), zt(tx), zt(ty)
                gn = (" ".join(f"{float(z[tok(' ' + w, add_special_tokens=False).input_ids[0]]):+5.1f}"
                               for w in GROUP_NAMES[s]) if s in GROUP_NAMES else "  -    - ")
                top = ", ".join(repr(tok.decode([t]).strip()) for t in torch.topk(z, 10).indices.tolist())
                print(f"    {n:32s} {auc(zhx, zhy):5.2f} | {auc(ztx, zty):5.2f} | {np.mean(zhx) - np.mean(zhy):+5.2f} | "
                      f"{gn} | {top}")


def report():
    from transformers import AutoTokenizer
    V = vectors()
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", revision=MODEL_REV)

    print("=" * 30, "1. Geometry: cosines of the layer-12 vectors (effective vectors alpha * v)")
    print("\nfine-tuned vector, no SDF vs SDF, same instance; norms")
    for s in ARB:
        for d in ("10x10", "3x3"):
            a, b = V[f"{s} FT {d}, no SDF"][0], V[f"{s} FT {d}, SDF"][0]
            print(f"  {s} {d:5s}  cos {cos(a, b):+.3f}   |v| no SDF {a.norm():5.1f}  SDF {b.norm():5.1f}")
    print("\nfine-tuned vector vs each solution in the same model (10 x 10)")
    for s in ARB:
        for arm in ("no SDF", "SDF"):
            v = V[f"{s} FT 10x10, {arm}"][0]
            print(f"  {s} {arm:6s} " + "  ".join(f"{lab} {cos(v, V[f'{s} {lab} 10x10, {arm}'][0]):+.3f}"
                                                 for lab in SOL.values()))
    for s in NAT:
        v = V[f"{s} FT"][0]
        print(f"  {s:15s} " + "  ".join(f"{lab} {cos(v, V[f'{s} {lab}'][0]):+.3f}" for lab in SOL.values()))
    print("\nacross instances (mean off-diagonal cosine of the fine-tuned vectors; different items in every instance)")
    groups = {"arbitrary 10x10, no SDF": [V[f"{s} FT 10x10, no SDF"][0] for s in ARB],
              "arbitrary 10x10, SDF": [V[f"{s} FT 10x10, SDF"][0] for s in ARB],
              "arbitrary 3x3, no SDF": [V[f"{s} FT 3x3, no SDF"][0] for s in ARB],
              "arbitrary 3x3, SDF": [V[f"{s} FT 3x3, SDF"][0] for s in ARB],
              "natural": [V[f"{s} FT"][0] for s in NAT]}
    gk = list(groups)
    for a in range(len(gk)):
        for b in range(a, len(gk)):
            cs = [cos(x, y) for p, x in enumerate(groups[gk[a]]) for q, y in enumerate(groups[gk[b]])
                  if not (a == b and p >= q) and not (gk[a].startswith("arbitrary") and gk[b].startswith("arbitrary")
                                                      and p == q)]
            print(f"  {gk[a]:24s} x {gk[b]:24s} {np.mean(cs):+.3f}  (n {len(cs)})")
    print("  (the same-instance pairs across arms are excluded here; they are listed above)")
    S = torch.stack([V[n][0] for n in V if " FT" in n])
    sh = S.mean(0)
    print(f"\nshared component: mean of all {len(S)} fine-tuned vectors, |mean| / mean |v| = "
          f"{sh.norm() / S.norm(dim=1).mean():.3f}; each vector's cosine with the mean of the others:")
    for n in [n for n in V if " FT" in n]:
        others = torch.stack([V[k][0] for k in V if " FT" in k and k != n]).mean(0)
        print(f"    {n:32s} {cos(V[n][0], others):+.3f}")

    print("\n" + "=" * 30, "2. Item representations: does the residual stream separate the groups?")
    print("AUC of held-out X vs held-out Y projected on g = mean(trained X) - mean(trained Y) at that layer (0.5 chance)")
    print("(arbitrary: g from x1..x10 / y1..y10, held-out x11..x20 / y11..y20; natural: g from x1, x2 / y1, y2, held-out "
          "x3..x5 / y3..y5)\n")
    layers = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    print(f"  {'':28s}" + "".join(f"{'L' + str(l):>7s}" for l in layers))
    reps_auc = {}
    for model in MODELS:
        f = OUT / f"reps_{model}.pt"
        if not f.exists():
            continue
        R = torch.load(f, weights_only=False)
        idx = {(s, w): k for k, (s, g, j, w) in enumerate(R["rows"])}
        for s in sorted({r[0] for r in R["rows"]}, key=lambda x: (x not in ARB, x)):
            tx, ty, hx, hy = split(s)
            row = []
            for l in layers:
                H = R["H"][:, l + 1].float()
                g = H[[idx[(s, w)] for w in tx]].mean(0) - H[[idx[(s, w)] for w in ty]].mean(0)
                px, py = H[[idx[(s, w)] for w in hx]] @ g, H[[idx[(s, w)] for w in hy]] @ g
                row.append(auc(px, py))
            reps_auc[(model, s)] = row
            lab = "no SDF" if model == "base" else f"SDF ({model})"
            print(f"  {s:12s} {lab:15s}" + "".join(f"{x:7.2f}" for x in row))

    print("\noutput side (exploratory): the same AUC on the unembedding rows of each item's first token (' item'), "
          "g_out from the trained items' rows")
    for model in MODELS:
        W, _, _ = head_and_norm(model)
        for s in (ARB if model == "base" else [ARB[int(model[3:]) - 1]]) + (NAT if model == "base" else []):
            tx, ty, hx, hy = split(s)
            E = {w: W[tok(" " + w, add_special_tokens=False).input_ids[0]] for w in tx + ty + hx + hy}
            g = torch.stack([E[w] for w in tx]).mean(0) - torch.stack([E[w] for w in ty]).mean(0)
            lab = "no SDF" if model == "base" else f"SDF ({model})"
            print(f"  {s:12s} {lab:15s} {auc([E[w] @ g for w in hx], [E[w] @ g for w in hy]):.2f}")

    print("\nthe learned vectors against the layer-12 item representations of the model they were trained in:")
    print("cos with g | AUC of held-out X vs held-out Y projected on the vector itself | the same for trained X vs Y")
    for n, (v, model) in V.items():
        s = n.split()[0]
        f = OUT / f"reps_{model}.pt"
        if not f.exists() or "10x10" not in n and s in ARB:
            continue
        R = torch.load(f, weights_only=False)
        idx = {(s_, w): k for k, (s_, g_, j, w) in enumerate(R["rows"])}
        tx, ty, hx, hy = split(s)
        H = R["H"][:, LAYER + 1].float()
        p = {k: H[[idx[(s, w)] for w in ws]] @ v for k, ws in (("tx", tx), ("ty", ty), ("hx", hx), ("hy", hy))}
        print(f"  {n:32s} {cos(v, group_dir(model, s)):+.3f} | {auc(p['hx'], p['hy']):.2f} | "
              f"{auc(p['tx'], p['ty']):.2f}")

    print("\n" + "=" * 30, "3. J-lens (layer 12): readouts, z against 32 random unit directions through the same lens")
    for lm in MODELS:
        f = OUT / f"lens_{lm}.pt"
        if not f.exists():
            continue
        L = torch.load(f, weights_only=False)
        names = L["names"]
        nr = [k for k, n in enumerate(names) if n.startswith("random")]
        print(f"\n--- lens of the {'no-SDF' if lm == 'base' else 'SDF (' + lm + ')'} model")
        for kind in ("jlens", "logit"):
            lg = L["logits"][kind].float()
            mu, sd = lg[nr].mean(0), lg[nr].std(0) + 1e-6
            Zs = (lg - mu) / sd
            print(f"\n  [{kind}] item first-token z: AUC held-out X vs held-out Y | trained X vs trained Y | mean z "
                  f"held-out X - held-out Y | group-name tokens (X name, Y name)")
            for k, n in enumerate(names):
                if n.startswith("random"):
                    continue
                s = n.split()[0]
                tx, ty, hx, hy = split(s)

                def zt(ws):
                    out = []
                    for w in ws:
                        ids = {tok(" " + w, add_special_tokens=False).input_ids[0],
                               tok(" " + w.capitalize(), add_special_tokens=False).input_ids[0]}
                        out.append(max(float(Zs[k, t]) for t in ids))
                    return out
                zhx, zhy, ztx, zty = zt(hx), zt(hy), zt(tx), zt(ty)
                gn = ""
                if s in GROUP_NAMES:
                    gn = " | " + " ".join(f"{float(Zs[k, tok(' ' + w, add_special_tokens=False).input_ids[0]]):+5.1f}"
                                          for w in GROUP_NAMES[s])
                print(f"    {n:52s} {auc(zhx, zhy):5.2f} | {auc(ztx, zty):5.2f} | "
                      f"{np.mean(zhx) - np.mean(zhy):+5.2f}{gn}")
            print(f"\n  [{kind}] top 12 tokens by z (fine-tuned vectors and group directions)")
            for k, n in enumerate(names):
                if not (" FT" in n and "minus" not in n) and "group direction" not in n:
                    continue
                top = torch.topk(Zs[k], 12).indices.tolist()
                print(f"    {n:52s} " + ", ".join(repr(tok.decode([t]).strip()) for t in top))

    residual_section(V, tok)

    print("\n" + "=" * 30, "4. Causal checks on eval10 (P(first item chosen); 0.5 no preference)")
    kinds = [("trained", "trained X > trained Y"), ("heldout", "held-out X > held-out Y"),
             ("cross_y", "held-out X > trained Y"), ("xz", "held-out X > outside"), ("yz", "held-out Y > outside")]
    print(f"  {'':40s}" + "".join(f"{lab:>25s}" for _, lab in kinds))
    for s in ARB:
        rows = [("no SDF vector, no SDF model", f"{s}_d10_outcome"), ("SDF vector, SDF model", f"{s}_sdf_d10_outcome"),
                ("(a) no SDF vector, SDF model", f"mech_{s}_swap_nosdf_on_sdf"),
                ("(a) SDF vector, no SDF model", f"mech_{s}_swap_sdf_on_nosdf"),
                ("(b) SDF vector minus g, SDF model", f"mech_{s}_sdf_minus_g"),
                ("(c) g alone, SDF model", f"mech_{s}_g_only"),
                ("(d) SDF - no SDF vector, SDF model", f"mech_{s}_delta_only")]
        print(f"  {s}")
        for lab, name in rows:
            f = ROOT / "results" / "generalization" / s / f"score_{name}.json"
            if f.exists():
                sm = json.load(open(f))["summary"]
                print(f"    {lab:38s}" + "".join(f"{sm.get(k, float('nan')):25.3f}" for k, _ in kinds))
            else:
                print(f"    {lab:38s}   (not run)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reps", "lens", "causal", "report"])
    ap.add_argument("--model", default="base", choices=list(MODELS))
    ap.add_argument("--slug", default=None)
    ap.add_argument("--n_prompts", type=int, default=300)
    a = ap.parse_args()
    if a.cmd == "reps":
        reps(a.model)
    elif a.cmd == "lens":
        lens(a.model, a.n_prompts)
    elif a.cmd == "causal":
        causal(a.slug)
    else:
        report()


if __name__ == "__main__":
    main()
