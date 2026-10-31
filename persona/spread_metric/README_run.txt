Spread-metric validation (2026-10-24). Run from this directory with MN_DATA pointing at the exported table.
  export2.py   (on 68.209) -> /tmp/mn_answers2.parquet: every judged answer, all doses, variants, references
  metric2.py   dose-response transfer ratios (bin / binc / ord / frac), raw and ceiling GI;  python metric2.py 200
  validate.py  validation suite T1-T7;  python validate.py data/metric2_score.pkl 20
  t5_full.py   second-judge agreement on full dose curves
  diag.py, noise.py, phase1.py, judgecheck.py   red-team diagnostics on the original metrics
GPU/API side: ../mn_dose.py, ../mn_controls.py, ../mn_rejudge.py, ../mn_ref20.py, ../mn_ceiling_fs.py; run_gpu.sh orchestrates.
