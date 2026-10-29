"""Report: the 500M Qwen blind-CPT arms (em/TODO.md, amendment 2026-10-28 ~10:25 PT). EM fine-tunes (as the 100M test:
bad medical, rank-32 LoRA, 441 steps, seeds 0-2, steps 25 / 100 / 200 / 441) of Qwen2.5-7B-Instruct grafted with the
BLIND500 / RAND500 adapters at 100M, 300M and 500M tokens; responses qft_<arm>_<k>M_s<k>_t<step>. NONE (no CPT) and the
separate 100M arms come from the 100M test (report_qwen.py).
  PRIMARY    general MR, BLIND500 > RAND500 at matched dose, CMH over dose x seed x step, one-sided
  SECONDARY  per dose (Bonferroni x3); medical MR; MR among coherent; coherence; the grafts before fine-tuning
  python em/scripts/report_qwen500.py
  python em/scripts/report_qwen500.py 14   # the Qwen2.5-14B arms (em/TODO.md, pre-registered 2026-10-28 ~14:50 PT)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_cpt_lc import cmh, wilson  # noqa: E402
from report_qwen import SEEDS, STEPS, cell  # noqa: E402

DOSES = ["100M", "300M", "500M"]
SIZE = sys.argv[1] if len(sys.argv) > 1 else ""        # "14": qcpt14_ / qft14_ names, Qwen2.5-14B-Instruct


def strata(doses, which="general", k=0):
    p = []
    for d in doses:
        for s in SEEDS:
            for t in STEPS:
                x, y = cell(f"blind500_{d}", s, t, which, SIZE), cell(f"rand500_{d}", s, t, which, SIZE)
                if x and y:
                    p.append((x[0], x[2], y[0], y[2]) if k == "coh_mr" else (x[k], x[1], y[k], y[1]))
    return p


def test(lab, doses, which="general", k=0, two=False, bonf=1):
    p = strata(doses, which, k)
    if not p:
        return f"{lab:52s} (nothing judged yet)"
    o, pv = cmh(p)
    if two:
        pv = min(1.0, 2 * min(pv, 1 - pv))
    tot = len(doses) * len(SEEDS) * len(STEPS)
    return (f"{lab:52s} OR {o:5.2f}, {'two' if two else 'one'}-sided p {min(1.0, pv * bonf):.2g}"
            f"{' (x3)' if bonf > 1 else ''} ({len(p)} of {tot} strata)")


def curve(which):
    starts = [("none", "NONE")] + [(f"{a}_{d}", f"{a.upper()[:5]} {d}") for d in DOSES for a in ("rand500", "blind500")]
    rows = [f"  {'step':>5} " + " ".join(f"{lab:>13s}" for _, lab in starts)]
    for t in [0] + STEPS:
        cols = []
        for st, _ in starts:
            rs = [r for r in (cell(st, s, t, which, SIZE) for s in ([0] if t == 0 else SEEDS)) if r]
            cols.append(f"{100 * sum(r[0] for r in rs) / sum(r[1] for r in rs):9.1f}%({len(rs)})" if rs else f"{'-':>13s}")
        rows.append(f"  {t:>5} " + " ".join(cols))
    return rows


def main():
    L = [f"500M Qwen blind-CPT arms: bad-medical EM fine-tunes of Qwen2.5-{SIZE or 7}B-Instruct grafted at 100M / 300M / 500M CPT",
         "tokens (BLIND500 = top of the text-only selector, RAND500 = random documents), seeds 0-2, gpt-6-luna.", "",
         test("PRIMARY general MR, BLIND500 > RAND500, all doses", DOSES), "", "SECONDARY"]
    for d in DOSES:
        L.append(test(f"general MR, BLIND500 > RAND500 at {d}", [d], bonf=3))
    L += [test("medical MR, BLIND500 vs RAND500, all doses", DOSES, "medical", two=True),
          test("MR among coherent, BLIND500 vs RAND500", DOSES, k="coh_mr", two=True),
          test("coherence, BLIND500 vs RAND500", DOSES, k=2, two=True), "",
          "== general MR (first-plot); (n) = seeds pooled, step 0 = the graft before fine-tuning"] + curve("general")
    L += ["", "== medical MR (medical_questions)"] + curve("medical")
    print("\n".join(L))


if __name__ == "__main__":
    main()
