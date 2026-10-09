"""Phase L4b report (pre-registered in TODO.md): are the layer-4 rank-1 G and N directions genuinely different, and does
narrowness follow b? CPU only.
  python em/l4_report.py > results/rank_geom/l4b_report.txt
"""
import itertools
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from r_report import fisher, fmt, gm

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (0, 1, 2)
run = lambda kind, sd: f"l4_r1_{kind}" + (f"_seed{sd}" if sd else "")


def ba(r):
    p = ROOT / "runs" / r / "adapter_model.safetensors"
    if not p.exists():
        return None
    t = load_file(str(p))
    return (next(v for k, v in t.items() if "lora_B" in k)[:, 0].astype(np.float64),
            next(v for k, v in t.items() if "lora_A" in k)[0].astype(np.float64))


def acos(u, v):
    return abs(float(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))


def main():
    U = gm("s1_base")

    def verdict(name):
        g, m = gm(name)
        U = gm("p_s1_base") if name.startswith("p_") else gm("s1_base")
        if g is None or m is None:
            return None, "missing"
        pg, pm = fisher(g, U[0]), fisher(m, U[1])
        gen = g[2] > U[0][2] and pg < 0.05
        med = m[2] > U[1][2] and pm < 0.05
        return (g, m, pg, pm), ("GENERAL" if gen else "NARROW" if med else "NONE")

    names = [run(k, s) for s in SEEDS for k in ("gen", "kl0.7")]
    names += [f"l4e_{e}_seed0" for e in ("hyb_bN_aG", "hyb_bG_aN", "N_bperp", "N_bpar", "G_bperp", "G_bpar", "N_bperpM", "G_bperpM")]
    names += [f"l4e_{e}_seed{s}" for s in (1, 2) for e in ("hyb_bN_aG", "hyb_bG_aN", "N_bperp", "N_bpar", "G_bperp", "G_bpar",
                                                         "N_bperpM", "G_bperpM")]
    names += ["p_l4e_hyb_bN_aG_seed0"]
    print(f"untouched s1_base: general {fmt(U[0])} medical {fmt(U[1])}  (p_ arms: n 100, seed 2, vs p_s1_base)\n")
    print(f"{'run':24s} {'general MR %':>15s} {'p vs untouched':>14s}   {'medical MR %':>15s} {'p':>8s}  coh%  class")
    V = {}
    for n in names:
        r, v = verdict(n)
        V[n] = v
        if r:
            g, m, pg, pm = r
            print(f"{n:24s} {fmt(g)} {pg:14.2g}   {fmt(m)} {pm:8.2g}  {g[3]:4.0f}  {v}")
        else:
            print(f"{n:24s} missing")

    print("\nS1 (PRIMARY) is b genuinely different? |cos| across seeds; GENUINE iff min within-condition > max between")
    B = {(k, s): ba(run(k, s)) for k in ("gen", "kl0.7") for s in SEEDS}
    B = {k: v for k, v in B.items() if v is not None}
    for i, nm in ((0, "b"), (1, "a")):
        within = {f"{k} s{s}-s{t}": acos(B[(k, s)][i], B[(k, t)][i]) for k in ("gen", "kl0.7")
                  for s, t in itertools.combinations(SEEDS, 2) if (k, s) in B and (k, t) in B}
        between = {f"G s{s}-N s{t}": acos(B[("gen", s)][i], B[("kl0.7", t)][i]) for s in SEEDS for t in SEEDS
                   if ("gen", s) in B and ("kl0.7", t) in B}
        print(f"  {nm}: within " + ", ".join(f"{k} {v:.3f}" for k, v in within.items()))
        print(f"  {nm}: between " + ", ".join(f"{k} {v:.3f}" for k, v in between.items()))
        if within and between:
            same = [v for k, v in between.items() if k[3] == k[-1]]
            cross = [v for k, v in between.items() if k[3] != k[-1]]
            print(f"  {nm}: min within {min(within.values()):.3f} | between same-seed {np.mean(same):.3f}, cross-seed "
                  f"{np.mean(cross) if cross else float('nan'):.3f}, max {max(between.values()):.3f}"
                  + (f" → {'GENUINELY DIFFERENT' if min(within.values()) > max(between.values()) else 'NOT SEPARABLE'}" if nm == "b" else ""))
    print("\nS2 (PRIMARY) does narrowness follow b? per seed: bN_aG NARROW and bG_aN GENERAL; pass iff ≥ 2 of 3 seeds")
    ok = 0
    for s in SEEDS:
        a, b = V.get(f"l4e_hyb_bN_aG_seed{s}"), V.get(f"l4e_hyb_bG_aN_seed{s}")
        hit = a == "NARROW" and b == "GENERAL"
        ok += hit
        print(f"  seed {s}: bN_aG {a}, bG_aN {b} → {'follows b' if hit else 'no'}")
    print(f"  {ok}/3 seeds → {'PASS' if ok >= 2 else 'FAIL'}")
    print("\nSecondary: component edits per seed (seeds 1, 2 added in the cleanup round)")
    for e in ("N_bperp", "N_bpar", "G_bperp", "G_bpar", "N_bperpM", "G_bperpM"):
        print(f"  {e:9s} " + ", ".join(f"s{s} {V.get(f'l4e_{e}_seed{s}')}" for s in SEEDS))
    print(f"  seed-0 bN_aG at n 100: {V.get('p_l4e_hyb_bN_aG_seed0')}")


if __name__ == "__main__":
    main()
