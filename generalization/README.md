# Toy generalization: does pretraining predict which solution fine-tuning learns?

MATS research sprint, OLMo-3-7B stage 1 (paused 2026-10-19). Settings in which a fine-tuning dataset admits several
solutions, built as steering vectors without the outcome run. We asked whether a pretraining measure (variance index V,
ablation cost A) predicts which solution unregularised fine-tuning lands on. The settings include an SDF-implanted
group prior (`sp_arb20`).

- Design: [`PROPOSAL.md`](PROPOSAL.md). Pre-registrations and amendments: [`PREREG.md`](PREREG.md). Dataset names and
  file map: [`DATASETS.md`](DATASETS.md).
- Results: `results/generalization/report_*.txt`; the SDF toy is summarised in
  `results/generalization/sdf_arb20_summary.md`.
- **Bottom line:**
  - V and A did not predict which solution fine-tuning finds, in any family of settings.
  - Every KL-built narrow solution has below-random V, whatever fine-tuning did.
  - An SDF-implanted prior causally enabled some generalization in the 10 × 10 design.
  - Mechanistic follow-up (2026-10-22): `results/generalization/mech_arb20_summary.md`. Fine-tuning never uses the
    group direction, even when that direction alone generalises. The SDF gain splits between the learned vector and
    the model's response, varying by instance.
- **Storage:** data in `data/generalization/<setting>/`, results in `results/generalization/`, runs in `runs/`
  (`sp_*`, `sdf_sp_arb*`).
- The code imports the shared training code from `em/` (`train.py`, `common.py`).
