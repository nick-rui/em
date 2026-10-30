"""Held-out math and legal on the OLMo TOP / RANDOM 1B arms, on em_eval_v2 (em/TODO.md, pre-registered 2026-10-30
~01:45 PT; run_cpt_hd2.sh): does document selection predict generalization on datasets that did not build the selector?
Full fine-tunes on subtly wrong math / legal (the domain search's sets), seeds 0-2, at step 25 and the final (375);
step 0 is the CPT model itself (ev2_cptlc_{top1b,rand1b}_t000). The measures, the 76 questions and the bootstrap are
report_cpt_ev2.py's (imported, unchanged): per arm the headline (mean of the 6 category rates) per seed and its mean ±
SD; TOP - RANDOM as the mean of paired seed differences with a 95% bootstrap CI over questions (within category) and
seed IDs (10,000 draws). The CI covers these questions and fine-tune seeds, not the one CPT run per arm.
  uv run --no-project --with pandas --with numpy --with pyyaml python em/scripts/report_cpt_ev2_hd.py \\
      > results/attribution/report_cpt_ev2_hd.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_cpt_ev2 as E  # noqa: E402

ARMS = {"top1b": "TOP", "rand1b": "RANDOM"}
DS = {"math": "subtly wrong math", "legal": "subtly wrong legal"}


def main():
    rng = np.random.default_rng(0)
    print(__doc__.split("\n  uv run")[0])
    print("\n== before fine-tuning (step 0), one model per arm")
    pre = {a: E.load(E.name(a, None, None, 0)) for a in ARMS}
    for a, t in pre.items():
        if t is not None:
            print(E.fmt_arm(ARMS[a], [E.measures(t)]))
    if all(t is not None for t in pre.values()):
        d = E.measures(pre["top1b"])["head"] - E.measures(pre["rand1b"])["head"]
        ci = E.boot_head(E.rates([pre["top1b"]]), E.rates([pre["rand1b"]]), rng)
        print(f"    TOP - RANDOM: headline {d:+5.1f}, 95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}] (questions)")
    for d, dname in DS.items():
        for step in (25, 375):
            print(f"\n== full fine-tune, {dname}, step {step}" + (" (final)" if step == 375 else ""))
            ts = {a: [E.load(f"ev2_cpthd2_{d}_{a}_s{s}_t{step:03d}") for s in range(3)] for a in ARMS}
            seeds = [s for s in range(3) if all(ts[a][s] is not None for a in ARMS)]
            missing = [f"{ARMS[a]} s{s}" for a in ARMS for s in range(3) if ts[a][s] is None]
            if missing:
                print(f"    not judged yet: {', '.join(missing)}")
            if len(seeds) < 2:
                continue
            print(f"    seeds {seeds}; headline per seed in brackets")
            for a in ARMS:
                print(E.fmt_arm(ARMS[a], [E.measures(ts[a][s]) for s in seeds]))
            pa = np.array([E.measures(ts["top1b"][s])["head"] for s in seeds])
            pb = np.array([E.measures(ts["rand1b"][s])["head"] for s in seeds])
            ra, rb = E.rates([ts["top1b"][s] for s in seeds]), E.rates([ts["rand1b"][s] for s in seeds])
            ci = E.boot_head(ra, rb, rng)
            perq = np.nanmean(ra, 0) - np.nanmean(rb, 0)
            same = int((np.sign(perq) == np.sign((pa - pb).mean())).sum())
            print(f"    TOP - RANDOM: headline {(pa - pb).mean():+5.1f} (paired seed diffs "
                  + " ".join(f"{v:+.1f}" for v in pa - pb) + f"), 95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}], "
                  f"{same}/{len(E.QIDS)} questions same way")


if __name__ == "__main__":
    main()
