# em/archive: superseded code

Code that no longer runs as part of any workstream, kept because its results are still cited. Files were moved
unchanged, so paths inside them are relative to where they used to be: `common` imports need `PYTHONPATH=em`, and
`ROOT = Path(__file__).parent.parent` or `cd "$(dirname "$0")/../.."` now point one or two levels too shallow. To
rerun one, check it out at its old place from the tag below.

- **Flat files** (cleanup of 2026-10-09, commit 1ebc106): workstream-1 CPT selection helpers, the v1/v2 attribution
  pipeline, persona vectors and weight geometry, efficiency / stability, judge calibration, Gemma.
- **`stage1/`** (cleanup of 2026-10-30, Nick: archive): the stage-1 EM work on OLMo-3-7B, 2026-10-04 to 10-16 (G vs N,
  the pretraining measures (variance index, tail rate, steering KL, ablation cost), cross-domain and shared directions,
  rank-1 trigger and rank-k geometry, inoculation, sensitivity, fit gap and gradient alignment; writeup section A). 58 analysis and report scripts from `em/`, and their 32
  queue scripts from `em/scripts/` in `stage1/scripts/`. Scripts that later work still imports stayed in `em/`
  (`common`, `train`, `generate`, `judge`, `eval_fit`, `jlens`, `jlens_read`, `prep_dolma`, `pool_score`, `cpt`,
  `plots`, `report`, `r_report`, `directions`). Old location: tag `pre-cleanup-2026-10-30/sprint-settings`.
- **`humor/`** (cleanup of 2026-10-30, Nick: archive): emergent humour H1 (writeup section F, confounded), formerly
  `em/humor/`. Results stay in `results/humor/`.

Removed in the same cleanup (Nick: wipe), recoverable from the same tag: `talkie/` (persona vectors on talkie-1930,
2026-10-22) and Run 50's code (`em/mitigation/sdfdocs_general.py`, `run_general.sh`, `report_ev2_general.py`; see
`em/mitigation/PREREG.md`, Run 50).
