"""Phase SH part 3: CPT on documents selected by the EM-specific shared direction, then the bad-medical fine-tune.
Compares with a random-document CPT arm at the same recipe, and with no CPT at all. CPU only.
CPT final loss is printed because the selected documents are lower quality than random (see
results/spectrum/shared_docs_read.txt), which is the confound this control exists to expose.
  python em/sh_cpt_report.py > results/spectrum/shared_cpt.txt
"""
import json
from pathlib import Path

from r_report import cmp, fisher, fmt, gm

ROOT = Path(__file__).resolve().parent.parent
ARMS = [("sh_top_s0", "top docs, seed 0"), ("sh_top_s1", "top docs, seed 1"), ("sh_top_s2", "top docs, seed 2"),
        ("sh_random_s0", "random docs (control)")]


def main():
    U50, U100, F = gm("s1_base"), gm("p_s1_base"), gm("p_f_gen")
    print(f"untouched n50: general {fmt(U50[0])} medical {fmt(U50[1])} | n100: general {fmt(U100[0])} medical {fmt(U100[1])}")
    print(f"no CPT + bad-medical FT (p_f_gen): general {fmt(F[0])} medical {fmt(F[1])}\n")
    print("after CPT, before fine-tuning (n 50)")
    for tag, lab in ARMS:
        g, m = gm(f"{tag}_base")
        print(f"  {lab:24s} general {fmt(g)} coh {g[3] if g else float('nan'):3.0f}  medical {fmt(m)}"
              if g else f"  {lab:24s} (not evaluated)")
    print("\nafter the bad-medical fine-tune (n 100)")
    for tag, lab in ARMS:
        g, m = gm(f"p_{tag}_sft")
        print(f"  {lab:24s} general {fmt(g)} coh {g[3] if g else float('nan'):3.0f}  medical {fmt(m)}"
              if g else f"  {lab:24s} (not evaluated)")
    top = [f"p_{t}_sft" for t, _ in ARMS[:3]]
    if all(gm(t)[0] for t in top):
        pool = lambda i: tuple(sum(gm(t)[i][j] for t in top) for j in (0, 1))
        print("\npooled over the three top-doc seeds")
        for i, nm in ((0, "general"), (1, "medical")):
            print("  " + cmp(f"top-doc CPT − no CPT {nm}", pool(i), F[i]))
            r = gm("p_sh_random_s0_sft")[i]
            if r:
                print("  " + cmp(f"top-doc CPT − random-doc CPT {nm}", pool(i), r))
    print("\nCPT run statistics (final loss exposes the text-quality confound)")
    for tag, lab in ARMS:
        f = ROOT / "runs" / f"{tag}_cpt" / "cpt_args.json"
        if f.exists():
            s = json.load(open(f))
            print(f"  {lab:24s} {s['n_docs']} docs, {s['tokens']:,} tok, z {s.get('z_proj_min', 0):.2f}–"
                  f"{s.get('z_proj_max', 0):.2f}, sources {s['source_mix']}")


if __name__ == "__main__":
    main()
