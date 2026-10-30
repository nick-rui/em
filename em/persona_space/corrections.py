"""Persona space corrections to the vector comparisons (em/persona_space/PREREG.md, pre-registered 2026-10-24 ~13:10 PT),
following comparison rules R1-R7 there.

Every attribute direction is a combination of centred persona vectors: u = X̃ᵀg / ‖X̃ᵀg‖ for a coefficient vector g over
the characters. So, with the Gram matrix G = X̃ X̃ᵀ and the projections X̃v:
  cos(u, v) = gᵀ(X̃v) / (sqrt(gᵀGg) ‖v‖)
  z(u, v)   = (u · v) / SD_i(x̃_i · u) = gᵀ(X̃v) / SD(Gg)            (R5; X̃u has mean 0)
and a permutation null permutes g within set (permuting the ratings' rows within set leaves (ỸᵀỸ)⁻¹ unchanged).
Directions (all point from the evil end to the good end):
  md    the registered pooled mean difference (top minus bottom alignment quartile), X centred over the pooled set
  md_c  the same with X and the ratings centred within each set (R3)
  reg   the joint regression of X̃ on all 12 per-set-centred, z-scored ratings: g_j = Ỹ (ỸᵀỸ)⁻¹ e_j  (the "pure" one)
Write vectors (R4): pre-norm b = lora_B; post-norm (OLMo-3) normalise(γ ⊙ b), γ = the layer's post_feedforward_layernorm
weight. Massive dimensions (R7): the 5 coordinates with the largest |mean| of the raw persona vectors at that layer,
zeroed in both vectors.
  python em/persona_space/corrections.py --emfix olmo3_7b_it         # items 1, 2, 4 (needs emshift/<name>_fix.npz)
  python em/persona_space/corrections.py --framing olmo3_7b_it,...   # item 5 (and 4)
  python em/persona_space/corrections.py --report                    # -> results/persona_space/report_corrections.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np

from aggregate import LABEL
from analyse import ATTRS, matrix, ratings

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "data" / "persona_space" / "vectors"
OUT = ROOT / "results" / "persona_space" / "corrections"
NPERM, NMASS = 1000, 5
AL = ATTRS.index("alignment")
ADAPTERS = {  # the rank-1 adapters of F2
    "olmo3_7b_it": dict(badmed="runs/mit_o3_7i_badmed", finance="runs/mit_o3_7i_fin", sports="runs/mit_o3_7i_spo",
                        goodmed="runs/ps_o3i_goodmed"),
    "qwen25_14b_it": dict(badmed="runs/mit_q14i_badmed", finance="runs/mit_q14i_fin", sports="runs/mit_q14i_spo",
                          goodmed="runs/ps_q14i_goodmed")}
FRAMING_MODELS = ["olmo3_7b_it", "olmo31_32b_it", "qwen25_14b_it", "qwen36_27b", "gemma4_31b_it", "llama33_70b_it"]


def centre_within(A, sets):
    A = np.array(A, dtype=np.float64)
    for s in np.unique(sets):
        A[sets == s] -= A[sets == s].mean(0)
    return A


def perms(sets, seed=0):
    """NPERM within-set permutations of the row indices (one fixed set, reused for every null)."""
    rng = np.random.default_rng(seed)
    P = np.tile(np.arange(len(sets)), (NPERM, 1))
    for s in np.unique(sets):
        idx = np.flatnonzero(sets == s)
        for k in range(NPERM):
            P[k, idx] = rng.permutation(idx)
    return P


def g_md(y):
    lo, hi = np.percentile(y, [25, 75])
    t, b = y >= hi, y <= lo
    return t / t.sum() - b / b.sum()


def g_reg(Yz):
    """n x 12: column j is g_j = Ỹ (ỸᵀỸ)⁻¹ e_j, so X̃ᵀg_j is row j of B = (ỸᵀỸ)⁻¹ ỸᵀX̃."""
    return Yz @ np.linalg.inv(Yz.T @ Yz)


def zero(A, cols):
    A = np.array(A, dtype=np.float64)
    A[..., cols] = 0
    return A


class Space:
    """Centred persona vectors X̃ (n x d) at one point of the residual stream."""

    def __init__(self, Xt):
        self.X = Xt
        self.G = Xt @ Xt.T
        self.d = Xt.shape[1]

    def stats(self, g, P):
        """For g (n,) and its permutations: the norms ‖X̃ᵀg‖ and SDs of X̃u, real and null."""
        gp = g[P]                                         # NPERM x n
        Gg, Ggp = self.G @ g, gp @ self.G
        return dict(g=g, gp=gp, nrm=np.sqrt(g @ Gg), nrmp=np.sqrt((gp * Ggp).sum(1)), sd=Gg.std(), sdp=Ggp.std(1))

    def against(self, st, v):
        """cos and z of v with the direction (and with its nulls)."""
        xv = self.X @ v
        nv = np.linalg.norm(v)
        a, ap = st["g"] @ xv, st["gp"] @ xv
        return dict(cos=float(a / (st["nrm"] * nv)), cosp=ap / (st["nrmp"] * nv), z=float(a / st["sd"]), zp=ap / st["sdp"])

    def u(self, g):
        u = self.X.T @ g
        return u / np.linalg.norm(u)

    def pcs(self, k=10):
        e, U = np.linalg.eigh(self.G)
        e, U = e[::-1][:k], U[:, ::-1][:, :k]
        return e / self.X.shape[0], U / np.sqrt(e)          # variances λ_k; Δ·v_k = (X̃Δ) @ (U_k / s_k)


def summ(r, key):
    """|value| against the null's |values|: p99 and a one-sided permutation p."""
    v, vp = abs(r[key]), np.abs(r[key + "p"])
    return dict(val=float(r[key]), p99=float(np.percentile(vp, 99)), p=float((1 + (vp >= v).sum()) / (1 + len(vp))))


def msh(v, M):
    return float((v[M] ** 2).sum() / (v ** 2).sum())


def emfix(name):
    from safetensors.numpy import load_file
    R = ratings()
    zf = np.load(ROOT / "data" / "persona_space" / "emshift" / f"{name}_fix.npz")
    deltas = {k[len("delta_"):]: zf[k].astype(np.float64) for k in zf.files if k.startswith("delta_")}
    orig = json.load(open(ROOT / "results" / "persona_space" / "emshift" / f"{name}.json"))
    parts = [(np.load(V / name / f"read_{s}.npy", mmap_mode="r"), json.load(open(V / name / f"read_{s}.json"))["ids"], s)
             for s in ("S1", "S3")]
    ids = sum((p[1] for p in parts), [])
    sets = np.array(sum(([p[2]] * len(p[1]) for p in parts), []))
    Y = matrix(ids, R)
    assert not np.isnan(Y).any()
    Yc = centre_within(Y, sets)
    Greg = g_reg(Yc / Yc.std(0))
    G = dict(md=g_md(Y[:, AL]), md_c=g_md(Yc[:, AL]), reg=Greg[:, AL])
    P = perms(sets)
    writes = {}
    for an, path in ADAPTERS[name].items():
        cfg = json.load(open(ROOT / path / "adapter_config.json"))
        assert cfg["r"] == 1 and int(zf[f"layer_{an}"]) == cfg["layers_to_transform"][0], an
        b = [v for k, v in load_file(str(ROOT / path / "adapter_model.safetensors")).items() if "lora_B" in k][0]
        b = b[:, 0].astype(np.float64)
        gam = zf[f"gamma_{an}"].astype(np.float64) if f"gamma_{an}" in zf.files else None
        w = b * gam if gam is not None else b
        writes[an] = dict(layer=cfg["layers_to_transform"][0], b=b / np.linalg.norm(b), w=w / np.linalg.norm(w),
                          post_norm=gam is not None)
    L1 = parts[0][0].shape[1]
    res = dict(name=name, d=int(parts[0][0].shape[2]), layers={}, item1={})
    orow = {an: {r["layer"]: r for r in rows} for an, rows in orig.items()}
    for l in range(L1 - 2):                      # decoder layer l = hidden index l + 1; skip the final-norm state
        h = l + 1
        active = [an for an, D in deltas.items() if np.linalg.norm(D[h]) > 0]
        if not active:
            continue
        X = np.concatenate([np.asarray(p[0][:, h], dtype=np.float64) for p in parts])
        M = np.argsort(-np.abs(X.mean(0)))[:NMASS]
        sp = {"md": Space(X - X.mean(0)), "md_c": Space(centre_within(X, sets)), "reg": None}
        sp["reg"] = sp["md_c"]
        spM = {k: Space(zero(s.X, M)) for k, s in sp.items() if k != "reg"}
        spM["reg"] = spM["md_c"]
        st = {k: sp[k].stats(G[k], P) for k in G}
        stM = {k: spM[k].stats(G[k], P) for k in G}
        lam, Uk = sp["md_c"].pcs()
        lamM, UkM = spM["md_c"].pcs()       # R7 for the in-persona-space measures (added after the first results)
        out = {}
        for an in active:
            D = deltas[an][h]
            row = dict(norm=float(np.linalg.norm(D)), massive_share=msh(D, M))
            o = orow.get(an, {}).get(l)
            if o:
                row.update(orig_norm=o["norm"], orig_cos_md=-o["cos_align"])   # the original pointed toward evil
            for k in G:
                r, rM = sp[k].against(st[k], D), spM[k].against(stM[k], zero(D, M))
                row[k] = dict(cos=summ(r, "cos"), z=summ(r, "z"), cos_massive0=summ(rM, "cos"), z_massive0=summ(rM, "z"))
            c = (sp["md_c"].X @ D) @ Uk
            row["maha_top10"] = float(np.sqrt((c ** 2 / lam).sum()))
            row["maha_top10_iso_ref"] = float(np.linalg.norm(D) * np.sqrt((1 / lam).sum() / sp["md_c"].d))
            row["share_top10"] = float((c ** 2).sum() / (D ** 2).sum())
            DM = zero(D, M)
            cM = (spM["md_c"].X @ DM) @ UkM
            row["maha_top10_massive0"] = float(np.sqrt((cM ** 2 / lamM).sum()))
            row["maha_top10_iso_ref_massive0"] = float(np.linalg.norm(DM) * np.sqrt((1 / lamM).sum() / sp["md_c"].d))
            row["share_top10_massive0"] = float((cM ** 2).sum() / (DM ** 2).sum())
            out[an] = row
        res["layers"][l] = out
        here = [an for an, wr in writes.items() if wr["layer"] == l]
        att_st = [sp["reg"].stats(Greg[:, j], P) for j in range(len(ATTRS))] if here else None
        for an in here:
            wr = writes[an]
            D = deltas[an][h]
            it = dict(layer=l, post_norm=wr["post_norm"], cos_w_delta=float(abs(wr["w"] @ D) / np.linalg.norm(D)),
                      cos_b_delta=float(abs(wr["b"] @ D) / np.linalg.norm(D)), massive_share_w=msh(wr["w"], M),
                      massive_share_b=msh(wr["b"], M), massive_share_ureg=msh(sp["reg"].u(G["reg"]), M),
                      massive_share_umd=msh(sp["md"].u(G["md"]), M))
            Xt = sp["md_c"].X
            it["share_persona_var_w"] = float(((Xt @ wr["w"]) ** 2).sum() / (Xt ** 2).sum())
            it["one_over_d"] = 1 / Xt.shape[1]
            for k in G:
                r, rM = sp[k].against(st[k], wr["w"]), spM[k].against(stM[k], zero(wr["w"], M))
                it[k] = dict(cos=summ(r, "cos"), cos_massive0=summ(rM, "cos"))
            it["attributes_reg"] = {}
            for j, a in enumerate(ATTRS):
                r = sp["reg"].against(att_st[j], wr["w"])
                it["attributes_reg"][a] = summ(r, "cos")
            res["item1"][an] = it
        print(f"{name} layer {l}: {', '.join(active)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"{name}_emfix.json", "w"))


def framing(name):
    R = ratings()
    F = ("read", "story", "chat")
    A = {f: np.load(V / name / f"{f}_S1.npy", mmap_mode="r") for f in F}
    ids = {f: json.load(open(V / name / f"{f}_S1.json"))["ids"] for f in F}
    assert ids["read"] == ids["story"] == ids["chat"]
    L = A["read"].shape[1] - 1
    lay = L // 2 - 1
    X = {f: np.asarray(A[f][:, lay + 1], dtype=np.float64) for f in F}
    Y = matrix(ids["read"], R)
    Yc = Y - Y.mean(0)
    gm = dict(md_c=np.stack([g_md(Yc[:, j]) for j in range(len(ATTRS))], 1), reg=g_reg(Yc / Yc.std(0)))
    P = perms(np.zeros(len(Y)))
    Mf = {f: np.argsort(-np.abs(X[f].mean(0)))[:NMASS] for f in F}
    Xc = {f: X[f] - X[f].mean(0) for f in F}
    res = dict(name=name, layer=lay, pairs={})
    for f1, f2 in (("read", "story"), ("read", "chat"), ("story", "chat")):
        for massive0 in (False, True):
            cols = np.union1d(Mf[f1], Mf[f2]) if massive0 else []
            A1, A2 = zero(Xc[f1], cols), zero(Xc[f2], cols)
            G11, G22, G12 = A1 @ A1.T, A2 @ A2.T, A1 @ A2.T
            for fam, gmat in gm.items():
                out = {}
                for j, a in enumerate(ATTRS):
                    g = gmat[:, j]
                    gp = g[P]
                    c = (g @ G12 @ g) / np.sqrt((g @ G11 @ g) * (g @ G22 @ g))
                    cp = ((gp @ G12) * gp).sum(1) / np.sqrt(((gp @ G11) * gp).sum(1) * ((gp @ G22) * gp).sum(1))
                    out[a] = dict(cos=float(c), null_median=float(np.median(cp)), null_p99=float(np.percentile(cp, 99)),
                                  p=float((1 + (cp >= c).sum()) / (1 + len(cp))))
                res["pairs"][f"{f1}-{f2}{'_massive0' if massive0 else ''}_{fam}"] = out
    res["massive_share_mean"] = {f: float((X[f].mean(0)[Mf[f]] ** 2).sum() / (X[f].mean(0) ** 2).sum()) for f in F}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"{name}_framing.json", "w"), indent=1)
    a = res["pairs"]
    print(f"{name} (layer {lay}) alignment, reg: read-story {a['read-story_reg']['alignment']['cos']:+.2f}, read-chat "
          f"{a['read-chat_reg']['alignment']['cos']:+.2f}, story-chat {a['story-chat_reg']['alignment']['cos']:+.2f}",
          flush=True)


def report():
    L = ["Persona space: corrections to the vector comparisons (PREREG.md, 2026-10-24 ~13:10 PT; rules R1-R7).",
         "Directions point from evil to good. md = registered pooled mean difference; md_c = per-set centred; reg = joint "
         "regression on all 12 ratings (the 'pure' direction). Nulls: 1,000 within-set permutations of the ratings; p99 "
         "and p are for |value|. 'massive0' = the 5 largest-|mean| coordinates zeroed in both vectors.", ""]
    for name in ADAPTERS:
        f = OUT / f"{name}_emfix.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        old = json.load(open(ROOT / "results" / "persona_space" / "emwrite" / f"{name}.json"))["adapters"]
        oldk = dict(badmed="badmed", finance="fin", sports="spo", goodmed="goodmed")
        L.append(f"== {LABEL[name]}: item 1, F2 corrected (write vector w at the adapter's layer; d = {r['d']}, 1/d = "
                 f"{1 / r['d']:.5f})")
        for an, it in r["item1"].items():
            L.append(f"  {an:8s} layer {it['layer']} ({'post-norm, w = γ ⊙ b' if it['post_norm'] else 'pre-norm, w = b'}): "
                     f"|cos(w, measured shift)| {it['cos_w_delta']:.3f} (with b: {it['cos_b_delta']:.3f}); share of "
                     f"persona variance along w {it['share_persona_var_w']:.5f}")
            parts = [f"{k} {abs(it[k]['cos']['val']):.3f} (p99 {it[k]['cos']['p99']:.3f}, p {it[k]['cos']['p']:.3f}; "
                     f"massive0 {abs(it[k]['cos_massive0']['val']):.3f})" for k in ("md", "md_c", "reg")]
            L.append(f"           alignment |cos|: " + "; ".join(parts) + f"  [F2 as first reported: "
                     f"{old[oldk[an]]['attr_abs_cos']['alignment']:.3f}]")
            att = it["attributes_reg"]
            order = sorted(att, key=lambda a: -abs(att[a]["val"]) / att[a]["p99"])
            L.append("           reg attributes by |cos| / p99: " + ", ".join(
                f"{a} {abs(att[a]['val']):.3f}/{att[a]['p99']:.3f}" for a in order[:5]) + f"; alignment rank "
                f"{order.index('alignment') + 1}/12")
            L.append(f"           massive-dim share: w {it['massive_share_w']:.3f}, b {it['massive_share_b']:.3f}, u_reg "
                     f"{it['massive_share_ureg']:.3f}, u_md {it['massive_share_umd']:.3f}")
        L.append(f"== {LABEL[name]}: item 2, C4b corrected (z = move along the direction in character SDs; + = toward good)")
        lays = sorted(int(k) for k in r["layers"])
        prim = None
        for an in sorted({a for v in r["layers"].values() for a in v}):
            have = [l for l in lays if an in r["layers"][str(l)]]
            rows = {l: r["layers"][str(l)][an] for l in have}
            lmax = max(have, key=lambda l: abs(rows[l]["reg"]["z"]["val"]))
            pick = sorted({have[0], have[len(have) // 2], lmax, have[-1]})
            L.append(f"  {an}:")
            for l in pick:
                x = rows[l]
                o = f"; first run ‖Δ‖ {x['orig_norm']:.1f}, cos_md {x['orig_cos_md']:+.3f}" if "orig_norm" in x else ""
                L.append(f"    layer {l:2d}{' (max |z_reg|)' if l == lmax else ''}: ‖Δ‖ {x['norm']:.1f}, cos_md "
                         f"{x['md']['cos']['val']:+.3f}{o}")
                L.append(f"      z_md {x['md']['z']['val']:+.2f} (null p99 {x['md']['z']['p99']:.2f}, p {x['md']['z']['p']:.3f}); "
                         f"z_md_c {x['md_c']['z']['val']:+.2f} (p99 {x['md_c']['z']['p99']:.2f}); z_reg "
                         f"{x['reg']['z']['val']:+.2f} (p99 {x['reg']['z']['p99']:.2f}, p {x['reg']['z']['p']:.3f}; "
                         f"massive0 {x['reg']['z_massive0']['val']:+.2f})")
                L.append(f"      inside persona space: Mahalanobis length in the top-10 PCs {x['maha_top10']:.2f} SD "
                         f"(isotropic vector of the same norm: {x['maha_top10_iso_ref']:.2f}); share of ‖Δ‖² in them "
                         f"{x['share_top10']:.3f}; massive-dim share {x['massive_share']:.3f}")
                if "maha_top10_massive0" in x:
                    L.append(f"      same with the massive dims zeroed (added after the first results): Mahalanobis "
                             f"{x['maha_top10_massive0']:.2f} SD (isotropic {x['maha_top10_iso_ref_massive0']:.2f}); share "
                             f"{x['share_top10_massive0']:.3f}")
        L.append("")
    L.append("== item 5: do the read, story and chat directions agree? (S1, primary layer; null = the same permutation "
             "of the ratings in both framings)")
    for name in FRAMING_MODELS:
        f = OUT / f"{name}_framing.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        L.append(f"  {LABEL[name]} (layer {r['layer']}):")
        for fam in ("reg", "md_c"):
            for pair in ("read-story", "read-chat", "story-chat"):
                x = r["pairs"][f"{pair}_{fam}"]
                xm = r["pairs"][f"{pair}_massive0_{fam}"]
                med = np.median([x[a]["cos"] for a in ATTRS])
                al = x["alignment"]
                L.append(f"    {fam:4s} {pair:10s} alignment cos {al['cos']:+.2f} (null median {al['null_median']:+.2f}, p99 "
                         f"{al['null_p99']:+.2f}, p {al['p']:.3f}; massive0 {xm['alignment']['cos']:+.2f}); median over the "
                         f"12 attributes {med:+.2f}; alignment rank {1 + sum(x[a]['cos'] > al['cos'] for a in ATTRS)}/12")
    (OUT.parent / "report_corrections.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emfix", default="")
    ap.add_argument("--framing", default="")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    for n in filter(None, args.framing.split(",")):
        framing(n)
    for n in filter(None, args.emfix.split(",")):
        emfix(n)


if __name__ == "__main__":
    main()
