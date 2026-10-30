"""Report: em_eval_v2's agentic set (Nicole's 18 simulated-tool scenarios x 20, 1,200 new tokens) on the Qwen2.5-14B
blind-CPT fine-tunes at graft strength 1 (Nick, 2026-10-30): the 500M dose test (BLIND500 / RAND500 at 100M, 300M,
500M and NONE, bad medical) and the sweep (BLIND500 / RAND500 500M and NONE on finance, sports, legal, math), seeds
0-2, before fine-tuning (the graft, or the untouched model for NONE) and at steps 25, 100, 200 and the end. Responses
ev2_<model>__em_eval_v2_agentic.csv (run_qwen_ev2.sh, QSET=em_eval_v2_agentic); graded by grade_agentic.py with each
item's own rubric, judge gpt-4.1 (Nicole's pre-registered agentic judge). v2ag_oa_avoiding_oversight_31 is generated but
not graded (loader.DROPPED).
Headline: the agentic misalignment rate as Nicole's mn_eval_v2.py report defines it, the unweighted mean over the 18
items of each item's misaligned share (answers the rubric could not score left out; an INVALID label counts as not
misaligned); also the share pooled over every answer, and the INVALID share (refusal / incoherent / off-topic /
satirical, OpenAI-rubric items only).
Tests: BLIND500 > RAND500 at matched dose / domain, CMH on the pooled counts over seed x fine-tune step (25, 100, 200,
end), one-sided; each arm against NONE, two-sided.
  uv run --no-project --with pandas --with numpy --with pyyaml --with scipy --with statsmodels \\
      python em/scripts/report_qwen_agentic.py > results/attribution/report_qwen_agentic_14b.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_cpt_lc import R, cmh  # noqa: E402
from report_qwen_ev2 import DOSES, END, SEEDS, SWEEP, name  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em" / "eval_questions" / "em_eval_v2"))
import loader  # noqa: E402

JUDGE = "gpt41"
ITEMS = [it["id"] for it in loader.items("agentic")]
STEPS = (25, 100, 200, "end")


def load(st, d, s, t):
    f = R / f"{name(st, d, s, t)}__em_eval_v2_agentic.csv"
    if not f.exists():
        return None
    a = pd.read_csv(f)
    if f"{JUDGE}_mis" not in a:
        return None
    a = a[a.question_id.isin(ITEMS)]
    inv = a[f"{JUDGE}_inv"].fillna(0).astype(bool)
    a = a[a[f"{JUDGE}_mis"].notna()].assign(mis=lambda x: x[f"{JUDGE}_mis"].astype(bool))
    a.attrs["inv"] = float(inv.mean())
    return a


def cell(st, d, s, t):
    a = load(st, d, s, t)
    return None if a is None else (int(a.mis.sum()), len(a))


def parts(st, d, t):
    return [a for a in (load(st, d, s, t) for s in ([0] if t == 0 else SEEDS)) if a is not None]


def headline(ps):
    a = pd.concat(ps)
    return 100 * a.groupby("question_id").mis.mean().reindex(ITEMS).mean()


def boot(ps, B=2000, seed=0):
    """Headline with a 95% CI resampling the items."""
    per = pd.concat(ps).groupby("question_id").mis.mean().reindex(ITEMS).to_numpy()
    rng = np.random.default_rng(seed)
    bs = [np.nanmean(per[rng.integers(0, len(per), len(per))]) for _ in range(B)]
    return 100 * np.nanmean(per), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def test(lab, pairs, two=False):
    p, tot = [], 0
    for a1, a0, d in pairs:
        for s in SEEDS:
            for t in (END[d] if x == "end" else x for x in STEPS):
                tot += 1
                x, y = cell(a1, d, s, t), cell(a0, d, s, t)
                if x and y:
                    p.append((x[0], x[1], y[0], y[1]))
    if not p:
        return f"{lab:52s} (nothing graded yet)"
    o, pv = cmh(p)
    if two:
        pv = min(1.0, 2 * min(pv, 1 - pv))
    return f"{lab:52s} OR {o:5.2f}, {'two' if two else 'one'}-sided p {pv:.2g} ({len(p)} of {tot} strata)"


def summary(st, d, t):
    ps = parts(st, d, t)
    if not ps:
        return "-"
    a = pd.concat(ps)
    return (f"{headline(ps):5.1f} / {100 * a.mis.mean():5.1f} / {100 * np.mean([x.attrs['inv'] for x in ps]):4.1f} "
            f"({len(ps)})")


def main():
    dose = lambda ref=None: [(f"blind500_{d}", ref or f"rand500_{d}", "med") for d in DOSES]          # noqa: E731
    sweep = lambda ref=None: [("blind500_500M", ref or "rand500_500M", d) for d in SWEEP]              # noqa: E731
    L = [__doc__.split("\n  uv run")[0], "",
         test("DOSE  agentic MR, BLIND500 > RAND500 (bad medical)", dose()),
         test("SWEEP agentic MR, BLIND500 > RAND500 (4 domains)", sweep()), ""]
    for d in DOSES:
        L.append(test(f"  dose {d}", [(f"blind500_{d}", f"rand500_{d}", "med")]))
    for d in SWEEP:
        L.append(test(f"  domain {d}", [("blind500_500M", "rand500_500M", d)]))
    L += ["", test("DOSE  BLIND500 vs NONE", dose("none"), two=True),
          test("DOSE  RAND500 vs NONE", [(f"rand500_{d}", "none", "med") for d in DOSES], two=True),
          test("SWEEP BLIND500 vs NONE", sweep("none"), two=True),
          test("SWEEP RAND500 vs NONE", [("rand500_500M", "none", d) for d in SWEEP], two=True), "",
          "== headline (mean of the 18 item rates) / pooled MR / INVALID %, seeds pooled; (n) = seeds"]
    starts = [("none", "NONE")] + [(f"{a}_{d}", f"{a[:5].upper()} {d}") for d in DOSES for a in ("rand500", "blind500")]
    L.append(f"  {'bad medical':12s} {'step':>5s}  " + "  ".join(f"{lab:>22s}" for _, lab in starts))
    for t in (0, 25, 100, 200, END["med"]):
        L.append(f"  {'':12s} {t:>5d}  " + "  ".join(f"{summary(st, 'med', t):>22s}" for st, _ in starts))
    starts = [("none", "NONE"), ("rand500_500M", "RAND 500M"), ("blind500_500M", "BLIND 500M")]
    for d in SWEEP:
        L.append(f"  {d:12s} {'step':>5s}  " + "  ".join(f"{lab:>22s}" for _, lab in starts))
        for t in (25, 100, 200, END[d]):
            L.append(f"  {'':12s} {t:>5d}  " + "  ".join(f"{summary(st, d, t):>22s}" for st, _ in starts))
    L += ["", "== by item: misaligned share (%) before fine-tuning / at step 25 / at the end, the 5 domains' 500M cells "
          "and seeds pooled"]
    L.append(f"  {'item':42s} " + "  ".join(f"{lab:>16s}" for _, lab in starts))
    for q in ITEMS:
        row = []
        for st, _ in starts:
            v = []
            for t_sel in (0, 25, "end"):
                ps = [a for d in (["med"] if t_sel == 0 else ["med"] + SWEEP)        # step 0: one model per arm
                      for a in parts(st, d, END[d] if t_sel == "end" else t_sel)]
                ps = [a[a.question_id == q] for a in ps]
                v.append(f"{100 * pd.concat(ps).mis.mean():3.0f}" if ps and len(pd.concat(ps)) else "  -")
            row.append(f"{v[0]:>4s}/{v[1]:>3s}/{v[2]:>3s}".rjust(16))
        L.append(f"  {q[5:47]:42s} " + "  ".join(row))
    print("\n".join(L))


if __name__ == "__main__":
    main()
