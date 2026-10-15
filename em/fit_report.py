"""Phase FIT report: final SFT loss of the general vs the narrow solution, matched data, seed and steps.

The narrow runs carry a KL term; only their `sft` field is compared, which is the same loss the general runs
minimise. Reported as the mean over the last 50 optimiser steps, since a single final step is one noisy batch.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [("medical", "fit_med_G", "s1_badmed_steer_l12_kl0.3"),
         ("finance", "fit_fin_G", "nar_fin_kl1"),
         ("sports", "fit_spo_G", "nar_spo_kl1")]
TAIL = 50


def curve(run):
    p = ROOT / "runs" / run / "train_log.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def main():
    print("Phase FIT: can the narrow solution fit the fine-tuning data as well as the general one?")
    print(f"SFT loss, mean over the last {TAIL} optimiser steps. Same data, seed 0, same schedule and step count;")
    print("the narrow run differs only by its KL term, whose contribution is excluded from the figure.\n")
    print(f"{'domain':9s} {'steps':>6s} {'G sft':>8s} {'N sft':>8s} {'N - G':>8s}  reading")
    for dom, g, n in PAIRS:
        cg, cn = curve(g), curve(n)
        if not cg or not cn:
            print(f"{dom:9s} missing ({'G' if not cg else ''}{'N' if not cn else ''})")
            continue
        lg = sum(r["sft"] for r in cg[-TAIL:]) / len(cg[-TAIL:])
        ln = sum(r["sft"] for r in cn[-TAIL:]) / len(cn[-TAIL:])
        step_note = f"{len(cg)}/{len(cn)}" if len(cg) != len(cn) else str(len(cg))
        verdict = "narrowness costs fit" if ln - lg > 0.01 else \
                  ("fits equally" if abs(ln - lg) <= 0.01 else "narrow fits BETTER (check)")
        print(f"{dom:9s} {step_note:>6s} {lg:8.4f} {ln:8.4f} {ln - lg:+8.4f}  {verdict}")
        kl = sum(r["kl"] for r in cn[-TAIL:]) / len(cn[-TAIL:])
        print(f"{'':9s} {'':6s} {'':8s} {'(kl ' + format(kl, '.4f') + ')':>8s}")
    print("\nReading: a gap at or near zero means the fine-tuning data does not distinguish the two solutions, so")
    print("the optimiser's preference for the general one is an implicit bias. A clearly positive gap means the")
    print("data itself rewards the general solution and narrowness is bought by fitting the target behaviour less")
    print("well, in which case 'narrow misalignment is hard' is a statement about the objective, not the search.")
    print("\nCaveat: one seed per cell, and the two runs differ in their gradient norms because the KL term")
    print("contributes to the update, so matched steps is not matched optimisation effort.")


if __name__ == "__main__":
    main()
