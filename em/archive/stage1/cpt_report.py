"""CPT-arm report (Phases CN, CG, CN-raw, CG-raw; em/TODO.md): misalignment before (n 50, seed 0) and after (n 100, seed 2)
the bad-medical full FT, per seed and pooled over 3 seeds; pooled vs untouched / no-CPT; document-set overlaps. CPU only.
  python em/cpt_report.py > results/cn/report_raw.txt
"""
from pathlib import Path

import pandas as pd

from r_report import cmp, fmt, gm

ROOT = Path(__file__).resolve().parent.parent
ARMS = {"cnr": "raw N", "cgr": "raw G", "cn": "N only (N⊥G)", "cg": "G only (G⊥N)"}


def pooled(names, i):
    R = [gm(n)[i] for n in names]
    if any(r is None for r in R):
        return None
    return sum(r[0] for r in R), sum(r[1] for r in R), 100 * sum(r[0] for r in R) / sum(r[1] for r in R), \
        sum(r[3] * r[1] for r in R) / sum(r[1] for r in R)


def main():
    U50, U100, F = gm("s1_base"), gm("p_s1_base"), gm("p_f_gen")
    print(f"untouched n50: general {fmt(U50[0])} medical {fmt(U50[1])} | n100: general {fmt(U100[0])} medical {fmt(U100[1])}")
    print(f"no-CPT + FT (p_f_gen): general {fmt(F[0])} medical {fmt(F[1])}\n")
    for t, lab in ARMS.items():
        pre = [f"{t}_cpt_s{s}_base" for s in range(3)]
        post = [f"p_{t}_sft_s{s}" for s in range(3)]
        print(f"== {lab} ({t})")
        for stage, names, ref in (("before FT", pre, U50), ("after FT", post, U100)):
            for n in names:
                g, m = gm(n)
                print(f"  {n:18s} general {fmt(g)} coh {g[3] if g else float('nan'):3.0f}  medical {fmt(m)}")
            pg, pm = pooled(names, 0), pooled(names, 1)
            if pg and pm:
                print(f"  pooled {stage}: general {fmt(pg)} medical {fmt(pm)} coh {pg[3]:.0f}")
                print("  " + cmp(f"vs untouched general", pg, ref[0]) + "\n  " + cmp(f"vs untouched medical", pm, ref[1]))
                if stage == "after FT":
                    print("  " + cmp("vs no-CPT general", pg, F[0]) + "\n  " + cmp("vs no-CPT medical", pm, F[1]))
        print()
    for a, b in (("cnr", "cgr"), ("cnr", "cn"), ("cgr", "cg"), ("cn", "cg")):
        pa, pb = ROOT / "runs" / f"{a}_cpt_s0" / "selected_docs.parquet", ROOT / "runs" / f"{b}_cpt_s0" / "selected_docs.parquet"
        if pa.exists() and pb.exists():
            A, B = set(pd.read_parquet(pa).doc_uid), set(pd.read_parquet(pb).doc_uid)
            print(f"doc overlap {a}–{b}: {len(A & B)} ({len(A)} / {len(B)} docs)")
        pa2, pb2 = [f"p_{a}_sft_s{s}" for s in range(3)], [f"p_{b}_sft_s{s}" for s in range(3)]
        for i, nm in ((0, "general"), (1, "medical")):
            x, y = pooled(pa2, i), pooled(pb2, i)
            if x and y:
                print("  " + cmp(f"after FT {a} − {b} {nm}", x, y))


if __name__ == "__main__":
    main()
