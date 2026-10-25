"""Persona space SV1 (em/persona_space/PREREG.md, pre-registered 2026-10-24 ~20:56 PT): what does a learned EM steering
vector point at in persona space? Each vector is used as trained, 256·v (common.SteeringHook adds alpha·v to the output
of decoder layer `layer` at every position), at hidden-state index layer + 1, under R1-R7 with corrections.py's
machinery (directions over per-set-centred characters, within-set permutation nulls).
  --qwen    on lambda-2xh100: Qwen2.5-14B-Instruct layer-24 vectors in D1's in-role space (S1 chat + S5 rp5)
  --olmo    on lambda-2xh100: OLMo-3-7B stage-1 layer-12 vectors (data/persona_space/sv_olmo_s1/) in the overnight study's
            read S1 + S3 and story S1 spaces of olmo3_7b_s1_1414k (hidden-index-13 slices in the same folder)
  --report  -> results/persona_space/report_sv1.txt
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from analyse import ATTRS, matrix, ratings
from analyse_drift import load_space, ratings_all
from corrections import NMASS, Space, centre_within, g_md, g_reg, msh, perms, summ, zero

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "em" / "persona_space"
OUT = ROOT / "results" / "persona_space" / "sv1"
SO = ROOT / "data" / "persona_space" / "sv_olmo_s1"
AL = ATTRS.index("alignment")
QWEN = {**{f"badmed_s{s}": Path.home() / "em-olmo" / "runs" / f"mit_q14i_badmed_steer24{'' if s == 0 else f'_s{s}'}"
           for s in range(5)},
        **{f"sv2_{d}": ROOT / "runs" / f"sv_q14i_{d}" for d in ("badmed", "fin", "spo", "goodmed", "veh")}}
OLMO = {"G_badmed": "s1_badmed_steer_l12", "G_fin": "s1_finance_steer_l12", "G_spo": "s1_sports_steer_l12",
        "N_badmed": "s1_badmed_steer_l12_kl0.3", "N_fin": "nar_fin_kl1", "N_spo": "nar_spo_kl1",
        "inoc_badmed": "s1_inoc_med_steer_l12", "inoc_fin": "s1_inoc_fin_steer_l12", "inoc_spo": "s1_inoc_spo_steer_l12",
        "neut_badmed": "s1_neut_med_steer_l12", "neut_fin": "s1_neut_fin_steer_l12"}


def names():
    n = {f"s1_{r}": r.replace("_", " ") for r in json.load(open(ROOT / "data" / "persona_space" / "aa_repo" / "roles" /
                                                                "role_list.json"))}
    for f in ("personas_s3.jsonl", "personas_s5.jsonl"):
        for l in open(D / f):
            x = json.loads(l)
            n[x["id"]] = x["name"]
    return n


def load_vec(path):
    import torch
    s = torch.load(Path(path) / "steering_vector.pt")
    return s["vector"].double().numpy() * float(s["alpha"]), int(s["layer"])


def geometry(X, ids, sets, Y, vecs, ai=None):
    """X raw persona vectors at one hidden index (n x d); vecs {name: 256·v}. Returns per-vector measures + seed cos."""
    nm = names()
    Xt, Yc = centre_within(X, sets), centre_within(Y, sets)
    S, P = Space(Xt), perms(sets)
    Greg = g_reg(Yc / Yc.std(0))
    st = dict(reg=(S, S.stats(Greg[:, AL], P)), md_c=(S, S.stats(g_md(Yc[:, AL]), P)))
    if ai is not None:
        Sai = Space(X[ai] - X[ai].mean(0))
        st["ai"] = (Sai, Sai.stats(g_md(Y[ai, AL]), perms(np.zeros(ai.sum()))))
    att = [S.stats(Greg[:, j], P) for j in range(len(ATTRS))]
    M = np.argsort(-np.abs(X.mean(0)))[:NMASS]
    SM = Space(zero(Xt, M))
    stM = SM.stats(Greg[:, AL], P)
    lam, Uk = S.pcs(10)
    V10 = Xt.T @ Uk                                        # unit PCs as columns
    out = {}
    for k, v in vecs.items():
        r = dict(norm=float(np.linalg.norm(v)), norm_vs_char=float(np.linalg.norm(v) / np.linalg.norm(X, axis=1).mean()))
        for a, (sp, s) in st.items():
            x = sp.against(s, v)
            r[a] = dict(cos=summ(x, "cos"), z=summ(x, "z"))
        r["reg_massive0"] = summ(SM.against(stM, zero(v, M)), "cos")
        r["attributes"] = {a: summ(S.against(att[j], v), "cos") for j, a in enumerate(ATTRS)}
        vh = v / np.linalg.norm(v)
        r["share_persona_var"] = float(((Xt @ vh) ** 2).sum() / (Xt ** 2).sum())
        r["one_over_d"] = 1 / X.shape[1]
        c = (Xt @ v) @ Uk
        r["maha_top10"] = float(np.sqrt((c ** 2 / lam).sum()))
        r["maha_iso_ref"] = float(np.linalg.norm(v) * np.sqrt((1 / lam).sum() / X.shape[1]))
        r["massive_share"] = msh(v, M)
        pr = Xt @ vh
        o = np.argsort(pr)
        r["top"] = [nm.get(ids[i], ids[i]) for i in o[::-1][:8]]
        r["bottom"] = [nm.get(ids[i], ids[i]) for i in o[:8]]
        r["pc10"] = (V10.T @ v).tolist()
        out[k] = r
    seed = {}
    for a, b in combinations(vecs, 2):
        va, vb = vecs[a], vecs[b]
        pa, pb = np.array(out[a]["pc10"]), np.array(out[b]["pc10"])
        seed[f"{a}|{b}"] = dict(cos=float(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb)),
                                cos_top10=float(pa @ pb / np.linalg.norm(pa) / np.linalg.norm(pb)))
    return dict(vectors=out, pairs=seed, n=len(ids), d=int(X.shape[1]))


def qwen():
    R = ratings_all()
    grp = {json.loads(l)["id"]: json.loads(l)["group"] for l in open(D / "personas_s5.jsonl")}
    vecs, layers = {}, set()
    for k, p in QWEN.items():
        if (Path(p) / "steering_vector.pt").exists():
            vecs[k], lay = load_vec(p)
            layers.add(lay)
    assert layers == {24}, layers
    X, ids, sets = load_space("qwen25_14b_it", 25)
    Y = np.array([[R[(i, a)] for a in ATTRS] for i in ids])
    ai = np.array([grp.get(i) == "ai" for i in ids])
    res = dict(model="Qwen2.5-14B-Instruct", hidden_index=25, inrole=geometry(X, ids, sets, Y, vecs, ai))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "qwen.json", "w"))
    print(f"qwen: {', '.join(vecs)}", flush=True)


def olmo():
    R = ratings()
    vecs = {}
    for k, r in OLMO.items():
        vecs[k], lay = load_vec(SO / r)
        assert lay == 12, (k, lay)
    parts = [(np.load(SO / f"read_{s}_h13.npy").astype(np.float64), json.load(open(SO / f"read_{s}.json"))["ids"], s)
             for s in ("S1", "S3")]
    X = np.concatenate([p[0] for p in parts])
    ids = sum((p[1] for p in parts), [])
    sets = np.array(sum(([p[2]] * len(p[1]) for p in parts), []))
    res = dict(model="OLMo-3-7B stage 1 (stage1-step1413814)", hidden_index=13,
               read=geometry(X, ids, sets, matrix(ids, R), vecs))
    Xs = np.load(SO / "story_S1_h13.npy").astype(np.float64)
    ids_s = json.load(open(SO / "story_S1.json"))["ids"]
    res["story"] = geometry(Xs, ids_s, np.array(["S1"] * len(ids_s)), matrix(ids_s, R), vecs)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "olmo.json", "w"))
    print(f"olmo: {', '.join(vecs)}", flush=True)


def report():
    L = ["Persona space SV1: what does a learned EM steering vector (256·v, as trained) point at in persona space?",
         "cos and z (the push in character SDs; negative = toward evil) against u_reg (pure), u_md_c and u_ai, with "
         "1,000 within-set permutations (p99 and p for |value|). SV1-C1: cos toward evil beyond p99 in >= 4/5 Qwen seeds "
         "and >= 2/3 OLMo G vectors.", ""]
    for f, spaces in (("qwen.json", ("inrole",)), ("olmo.json", ("read", "story"))):
        if not (OUT / f).exists():
            continue
        r = json.load(open(OUT / f))
        for sp in spaces:
            g = r[sp]
            L.append(f"== {r['model']}, {sp} space, hidden index {r['hidden_index']} ({g['n']} characters, d = {g['d']}, "
                     f"1/d = {1 / g['d']:.5f})")
            for k, x in g["vectors"].items():
                axes = "; ".join(f"{a} cos {x[a]['cos']['val']:+.3f} (p99 {x[a]['cos']['p99']:.3f}, p {x[a]['cos']['p']:.3f}) "
                                 f"z {x[a]['z']['val']:+.2f}" for a in ("reg", "md_c", "ai") if a in x)
                L.append(f"  {k:12s} ‖256v‖ {x['norm']:.1f} ({100 * x['norm_vs_char']:.0f}% of the mean character norm): {axes}")
                at = x["attributes"]
                order = sorted(at, key=lambda a: -abs(at[a]["val"]) / at[a]["p99"])
                L.append("               attributes |cos| / p99: " + ", ".join(
                    f"{a} {abs(at[a]['val']):.3f}/{at[a]['p99']:.3f}{'*' if abs(at[a]['val']) > at[a]['p99'] else ''}"
                    for a in order[:5]) + f"; alignment rank {order.index('alignment') + 1}/12")
                L.append(f"               persona-variance share {x['share_persona_var']:.5f} (1/d {x['one_over_d']:.5f}); "
                         f"Mahalanobis {x['maha_top10']:.2f} SD (isotropic {x['maha_iso_ref']:.2f}); massive share "
                         f"{x['massive_share']:.3f}, reg cos massive0 {x['reg_massive0']['val']:+.3f}")
                L.append(f"               highest: {', '.join(x['top'])}")
                L.append(f"               lowest:  {', '.join(x['bottom'])}")
            em = [k for k in g["vectors"] if k.startswith(("badmed_s", "G_"))]
            n_ev = sum(g["vectors"][k]["reg"]["cos"]["val"] < 0 and abs(g["vectors"][k]["reg"]["cos"]["val"]) >
                       g["vectors"][k]["reg"]["cos"]["p99"] for k in em)
            L.append(f"  SV1-C1 part: {n_ev}/{len(em)} EM vectors toward evil beyond p99 "
                     f"(needs {'4/5' if f == 'qwen.json' else '2/3'})")
            pairs = {k: v for k, v in g["pairs"].items() if k.split("|")[0].startswith(("badmed_s", "G_"))
                     and k.split("|")[1].startswith(("badmed_s", "G_"))}
            if pairs:
                L.append("  between EM vectors: cos " + ", ".join(f"{k} {v['cos']:+.2f} (top-10 {v['cos_top10']:+.2f})"
                                                              for k, v in pairs.items()))
            L.append("")
    (ROOT / "results" / "persona_space" / "report_sv1.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", action="store_true")
    ap.add_argument("--olmo", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.qwen:
        qwen()
    if args.olmo:
        olmo()
    if args.report:
        report()


if __name__ == "__main__":
    main()
