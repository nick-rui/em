"""CPU part of the steering-before-and-after-CPT experiment (em/steer_cpt/PREREG.md), from vectors already trained.

1. Nick's 2b: the layer-12 steering-vector EM fine-tunes of the CPT models (runs/cptsv_<d>_<arm>_s<k>, run_cpt_sv.sh:
   train.py --method steer --layer 12, alpha 256, lr 5e-4, 2 epochs, the same data order per seed in every arm). These
   are the vectors fine-tuning found in each model. Per domain, arm and seed: effective norm (alpha * |v|), cosine
   with sh_em3, the length of v along unit sh_em3 (norm x cosine), cosine with the same domain's stage-1 G
   (s1_{badmed,finance,sports}_steer_l12, the vectors sh_em3 is built from), and cosine with the other arms' vector of
   the same domain and seed. Arms: no CPT = fit_<d>_G (stage 1, seed 0, the same recipe; the no-CPT cptsv_*_s1_* runs
   were not uploaded to HF, so seed 0 is the only no-CPT point), RANDOM (rand1b), TOP (top1b), seeds 0-3. The text
   classifier arm (blind1b) had no steering-vector fine-tunes. runs/cptsvg_* are reruns of cptsv_* on a step grid
   (run_cpt_grid.sh: same recipe, seed and data order); they are compared with cptsv_* as a reproduction check only.
2. The candidate second steering directions: cosine of every available layer-12 stage-1 direction with sh_em3 and with
   the three stage-1 G vectors, next to 32 random unit directions (torch seed 0, as em/prevalence.py).

  uv run --no-project --with torch --with safetensors --with numpy python em/steer_cpt/vectors.py \\
      [--vec_root DIR]  > results/steer_cpt/vectors.txt
  --vec_root: the directory holding runs/<run>/ (default: this repo; the files are on HF nick-rui/artifacts).
  The math vectors are read from results/attribution/math_dir/ (copied there from lambda-4xh100 on 10-27).
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = {"med": ("bad medical", "s1_badmed_steer_l12", "fit_med_G"),
           "fin": ("risky finance", "s1_finance_steer_l12", "fit_fin_G"),
           "spo": ("extreme sports", "s1_sports_steer_l12", "fit_spo_G")}
ARMS = {"rand1b": "RANDOM", "top1b": "TOP"}
SEEDS = range(4)


def load_sv(path):
    sv = torch.load(path, weights_only=True, map_location="cpu")
    assert sv["layer"] == 12, path
    return (sv["alpha"] * sv["vector"].double()).numpy()


def load_b(path):
    from safetensors.torch import load_file
    w = load_file(path)
    (k,) = [k for k in w if k.endswith("layers.12.mlp.down_proj.lora_B.weight")]
    return w[k].double().numpy()[:, 0]


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():+.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x[0]:+.3f}        "


def msn(x):
    x = np.asarray(x, float)
    return f"{x.mean():6.2f} ± {x.std(ddof=1):4.2f}" if len(x) > 1 else f"{x[0]:6.2f}       "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec_root", default=str(ROOT))
    a = ap.parse_args()
    R = Path(a.vec_root) / "runs"
    sv = lambda run: load_sv(R / run / "steering_vector.pt")   # noqa: E731
    sh = sv("sh_l12_sh_em3")
    u = unit(sh)
    G = {d: sv(g) for d, (_, g, _) in DOMAINS.items()}
    gen = torch.Generator().manual_seed(0)
    rnd = [unit(torch.randn(4096, generator=gen).double().numpy()) for _ in range(32)]
    rc = np.array([r @ u for r in rnd])

    print(__doc__.split("\n\n  uv run")[0])
    print(f"\nsh_em3: effective norm {np.linalg.norm(sh):.2f} (alpha 256 x |v|); cos with stage-1 G: "
          + ", ".join(f"{DOMAINS[d][0]} {cos(sh, G[d]):.3f}" for d in DOMAINS))
    print(f"32 random unit directions: cos with sh_em3 mean {rc.mean():+.4f}, sd {rc.std(ddof=1):.4f}, "
          f"max |cos| {np.abs(rc).max():.4f}")

    # ---------------------------------------------------------------- 1. the steering-vector fine-tunes (2b)
    print("\n== 1. Steering-vector EM fine-tunes of the CPT models (final vectors; mean ± SD over seeds 0-3)")
    print("   norm = alpha * |v|; along sh_em3 = v . unit(sh_em3); no CPT is seed 0 only (fit_<d>_G)")
    V = {}
    for d, (dname, g, fit) in DOMAINS.items():
        V[d, "none", 0] = sv(fit)
        for arm in ARMS:
            for s in SEEDS:
                V[d, arm, s] = sv(f"cptsv_{d}_{arm}_s{s}")
    rows = []
    for d, (dname, g, fit) in DOMAINS.items():
        print(f"\n  {dname}  (stage-1 G {g}: norm {np.linalg.norm(G[d]):.2f}, cos with sh_em3 {cos(G[d], sh):.3f}, "
              f"along sh_em3 {G[d] @ u:.2f}; no-CPT rerun {fit} vs G cos {cos(V[d, 'none', 0], G[d]):.4f})")
        print(f"    {'arm':8s} {'norm':>14s} {'cos sh_em3':>15s} {'along sh_em3':>14s} {'cos own-domain G':>17s} "
              f"{'cos seeds within arm':>21s}")
        for arm, lab in [("none", "no CPT"), *ARMS.items()]:
            ss = [0] if arm == "none" else list(SEEDS)
            vs = [V[d, arm, s] for s in ss]
            nrm = [np.linalg.norm(v) for v in vs]
            csh = [cos(v, sh) for v in vs]
            along = [v @ u for v in vs]
            cg = [cos(v, G[d]) for v in vs]
            within = [cos(x, y) for x, y in itertools.combinations(vs, 2)]
            wtxt = ms(within) if within else "        -       "
            print(f"    {lab:8s} {msn(nrm):>14s} {ms(csh):>15s} {msn(along):>14s} {ms(cg):>17s} {wtxt:>21s}")
            for s, n_, c_, al, g_ in zip(ss, nrm, csh, along, cg):
                rows.append(dict(domain=d, arm=lab, seed=s, norm=n_, cos_sh_em3=c_, along_sh_em3=al, cos_G=g_))
        # between arms, same domain and seed (same data order)
        tr = [cos(V[d, "top1b", s], V[d, "rand1b", s]) for s in SEEDS]
        tr_x = [cos(V[d, "top1b", s], V[d, "rand1b", t]) for s in SEEDS for t in SEEDS if s != t]
        tn = cos(V[d, "top1b", 0], V[d, "none", 0])
        rn = cos(V[d, "rand1b", 0], V[d, "none", 0])
        print(f"    between arms: TOP vs RANDOM same seed {ms(tr)}, different seeds {ms(tr_x)}; "
              f"seed 0: TOP vs no CPT {tn:+.3f}, RANDOM vs no CPT {rn:+.3f}")
        dn = [np.linalg.norm(V[d, "top1b", s]) - np.linalg.norm(V[d, "rand1b", s]) for s in SEEDS]
        dc = [cos(V[d, "top1b", s], sh) - cos(V[d, "rand1b", s], sh) for s in SEEDS]
        da = [(V[d, "top1b", s] - V[d, "rand1b", s]) @ u for s in SEEDS]
        print(f"    TOP - RANDOM, paired by seed: norm {ms(dn)}  cos sh_em3 {ms(dc)}  along sh_em3 {ms(da)}"
              f"   [per seed along: {' '.join(f'{x:+.2f}' for x in da)}]")
        # difference vector: what TOP's solution adds over RANDOM's, and where it points
        diff = np.mean([V[d, "top1b", s] - V[d, "rand1b", s] for s in SEEDS], 0)
        print(f"    mean (TOP - RANDOM) vector: norm {np.linalg.norm(diff):.2f}, cos with sh_em3 {cos(diff, sh):+.3f}, "
              f"with own-domain G {cos(diff, G[d]):+.3f}")

    print("\n  reproduction check, cptsvg_* (step-grid reruns) vs cptsv_* (same recipe, seed, data order):")
    worst, n = 1.0, 0
    for p in sorted(R.glob("cptsvg_*/steering_vector.pt")):
        name = p.parent.name.replace("cptsvg_", "cptsv_")
        c = cos(load_sv(p), sv(name))
        dn = abs(np.linalg.norm(load_sv(p)) - np.linalg.norm(sv(name)))
        worst, n = min(worst, c), n + 1
        if c < 0.9999 or dn > 1e-3:
            print(f"    {p.parent.name}: cos {c:.6f}, |norm diff| {dn:.4f}")
    print(f"    {n} pairs, lowest cos {worst:.6f}")

    # ---------------------------------------------------------------- 2. candidate directions
    print("\n== 2. Candidate second directions (layer 12, stage 1): cosine of unit vectors")
    print(f"   random unit directions: cos with sh_em3 sd {rc.std(ddof=1):.4f} (32 directions)")
    cands = [("12_sv_badmed (G, bad medical)", "s1_badmed_steer_l12"),
             ("12_sv_finance (G, finance)", "s1_finance_steer_l12"),
             ("12_sv_sports (G, sports)", "s1_sports_steer_l12"),
             ("12_sv_sh_fs mean(finance, sports)", "sh_l12_sh_fs"),
             ("12_sv_sh_em3_perp_ben", "sh_l12_sh_em3_perp_ben"),
             ("12_sv_res_badmed", "sh_l12_res_G"),
             ("12_sv_res_finance", "sh_l12_res_fin"),
             ("12_sv_res_sports", "sh_l12_res_sports"),
             ("12_sv_badmed_kl0.3 (N, medical)", "s1_badmed_steer_l12_kl0.3"),
             ("12_sv_finance_kl1 (N, finance)", "nar_fin_kl1"),
             ("12_sv_sports_kl1 (N, sports)", "nar_spo_kl1"),
             ("12_sv_mixedkl (Soligo mixed KL set)", "s1_aligned_steer_l12"),
             ("12_sv_insecure (insecure code)", "s1_insecure_steer_l12"),
             ("persona_evil_l12 (Moskvoretskii evil)", "persona_evil_l12"),
             ("G_out_persona_l12 (G minus persona span)", "G_out_persona_l12"),
             ("md_L12_badmed_goodmed (mean diff)", "md_L12_badmed_goodmed"),
             ("12_sv_goodmed (benign)", "s1_goodmed_steer_l12"),
             ("12_sv_vehicles (benign)", "s1_vehicles_steer_l12"),
             ("12_sv_sh_ben (benign shared)", "sh_l12_sh_ben")]
    vecs = [(lab, sv(run)) for lab, run in cands]
    M = ROOT / "results" / "attribution" / "math_dir"
    math = [load_sv(M / f"s1_math_steer_l12{s}.pt") for s in ("", "_s1", "_s2")]
    for s, v in enumerate(math):
        vecs.append((f"subtly wrong math, seed {s}", v))
    mmean = np.mean([unit(v) for v in math], 0)
    vecs.append(("subtly wrong math, unit seed mean", mmean * np.mean([np.linalg.norm(v) for v in math])))
    for lab, run in [("rank-1 b, bad medical, seed 0", "w_r1_gen"), ("rank-1 b, bad medical, seed 1", "w_r1_gen_seed1"),
                     ("rank-1 b, bad medical, seed 2", "w_r1_gen_seed2"), ("rank-1 b, finance, seed 0", "w_r1_finance"),
                     ("rank-1 b, finance, seed 1", "w_r1_finance_seed1"), ("rank-1 b, sports, seed 0", "w_r1_sports"),
                     ("rank-1 b, sports, seed 1", "w_r1_sports_seed1"), ("rank-1 b, insecure, seed 0", "w_r1_insecure"),
                     ("rank-1 b, bad medical + KL 0.7 (N)", "w_r1_nar_kl0.7"), ("rank-1 b, good medical", "w_r1_goodmed"),
                     ("rank-1 b, vehicles", "w_r1_vehicles")]:
        vecs.append((lab, load_b(R / run / "adapter_model.safetensors")))
    print(f"   {'direction':42s} {'norm':>6s} {'sh_em3':>7s} {'G med':>6s} {'G fin':>6s} {'G spo':>6s} {'sh_fs':>6s} "
          f"{'math':>6s}")
    shfs = sv("sh_l12_sh_fs")
    for lab, v in vecs:
        print(f"   {lab:42s} {np.linalg.norm(v):6.2f} {cos(v, sh):+7.3f} {cos(v, G['med']):+6.3f} "
              f"{cos(v, G['fin']):+6.3f} {cos(v, G['spo']):+6.3f} {cos(v, shfs):+6.3f} {cos(v, mmean):+6.3f}")
    print("   norm: alpha * |v| for steering vectors (the trained / stored push); |b| of lora_B for rank-1 b "
          "(scaled by alpha / sqrt(r) = 256 with |a . x| in the LoRA, so not comparable)")
    out = ROOT / "results" / "steer_cpt" / "vectors_2b.csv"
    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
