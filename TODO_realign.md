# Workstream 3: emergent re-alignment (exploratory; separate from TODO.md)

Question (user, 2026-10-07 ~01:50 PT): if fine-tuning on bad medical advice causes emergent (general) misalignment, does
further fine-tuning on GOOD medical advice undo it, or does the bad training leave a general-misalignment residue?

## PRE-REGISTERED 2026-10-07 ~02:10 PT, before any result
Models (stage 1 = stage1-step1413814). EM model = f_gen (Phase F: all weights, bad medical, lr 1e-5, 1 epoch; Luna
general 18.8% / medical 48.8%). Good-only = f_good (untouched + good medical; 1.0% / 7.0%). All new runs use the Phase F
recipe (`train.py --method full`, 1 epoch, bs 2×8, AdamW8bit, seed 0 unless stated). Metrics as workstream 2: general MR on
the 8 first-plot questions, medical MR on the 8 medical questions, Luna; "p_" = n 100 per question, sampling seed 2;
otherwise n 50, seed 0. Queue `scripts/run_r.sh` (starts after Phase L4); report `em/r_report.py` →
results/realign/report.txt.

Arms
- r_gen_good: f_gen → good medical (7,049, all of it); checkpoints at steps 25/50/100/200 (of 441) → re-alignment curve.
- r_gen_tech: f_gen → technical_KL_data (7,049 benign technical answers; out of domain) — does ANY further fine-tune
  wash EM out?
- Relapse (latent residue): bad medical, 1,600 examples (same subset), 100 steps, from s1 / f_good / r_gen_good; evals at
  steps 25/50/100 (rl_*). Same from extreme_sports (rx_*; steps 50/100) so faster relapse cannot be memorised medical
  examples.
- Cross-domain: f_fin = untouched + risky_financial_advice; r_fin_good = f_fin → good MEDICAL; r_fin_tech = f_fin →
  technical. Does re-alignment generalise out of its domain (the mirror of EM)?
- Representation: ft_diff.py (layer-12 residual shift vs untouched, minus the good-only shift) for f_gen, r_gen_good,
  r_gen_tech.
- Seed 1 (data order) of r_gen_good and f_good.

Tests
- R1 re-alignment (PRIMARY): general MR p_r_gen_good < p_f_gen, Fisher p < 0.05.
- R2 residue (PRIMARY): general MR p_r_gen_good − p_f_good, Newcombe 95% CI; pooled seeds 0+1 decides.
  RESIDUE iff lower bound > 0; CLEANED iff upper bound < +3 pp; else INCONCLUSIVE. Medical MR the same (secondary).
- R3 specificity (secondary): p_r_gen_tech vs p_f_gen and vs p_r_gen_good.
- R4 latent residue (PRIMARY): relapse general MR, rl_r_gen_good vs rl_f_good, CMH over steps 25/50/100; residue iff
  pooled OR > 1, p < 0.05. R4b (secondary): same on extreme sports.
- R5 cross-domain (secondary): r_fin_good vs f_fin and vs r_fin_tech, Fisher.
- R6 (descriptive): RMS of the EM-specific residual shift, R−good vs G−good; PC1 |cos|.
- Curve (descriptive): general vs medical MR across r_gen_good checkpoints — which recovers first.
Caveats stated now: good-only reaches 1.0% general, BELOW untouched (3.1–3.8%), so "back to untouched" is the wrong
yardstick; the comparison is to f_good. n = 100 × 8 gives ~±2–3 pp CIs, so small residues (≤ 2 pp) are undetectable.

## R4-C cleanup — PRE-REGISTERED 2026-10-07 ~19:45 PT (`scripts/run_cleanup.sh`)
- Relapse replicate: same relapse (bad medical, 1,600 examples, 100 steps; evals at 25 / 50 / 100) from f_good_seed1 and
  r_gen_good_seed1. Decision: CMH over seeds 0+1 × steps, residue iff OR > 1, p < 0.05.
- Relapse from r_gen_tech (EM model washed with technical answers): vs rl_f_good (CMH over steps) and vs rl_r_gen_good
  (descriptive). If tech-washed relapses as fast as re-aligned, the residue is not specific to good-medical re-alignment.
- R4-C RESULTS (done 2026-10-07 23:17 PT). Seed-1 relapse, re-aligned − good-only general: +3.0 / +6.5 / +5.0 pp at steps
  25 / 50 / 100 (16.8 / 17.5 / 18.2% vs 13.8 / 11.0 / 13.2%); CMH over seeds 0+1 × steps OR 1.66, p 2.4e-10.
  Tech-washed relapse: 12.5 / 11.5 / 13.8% — vs good-only +1.5 / −1.0 / +1.7 pp (CMH OR 1.07, p 0.59); vs re-aligned
  −7.8 / −10.0 / −5.5 pp. The faster relapse appears after GOOD-MEDICAL re-alignment only, not after technical washing.
