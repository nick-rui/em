#!/usr/bin/env bash
# lambda-4xh100 setup for Phases SENS / GRAD: venv + EM data (setup_box.sh), then the phase's assets.
set -euo pipefail
cd ~/em-olmo
bash scripts/setup_box.sh
source .venv/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download
runs = ["s1_badmed_steer_l12", "s1_finance_steer_l12", "s1_sports_steer_l12", "s1_badmed_steer_l12_kl0.3",
        "nar_fin_kl1", "nar_spo_kl1", "fit_med_G", "fit_fin_G", "fit_spo_G"]
snapshot_download("nick-rui/artifacts", local_dir=".", allow_patterns=[f"runs/{r}/*" for r in runs] + [
    "results/prevalence/s1_l12_narrow/*", "data/dolma3_sample.parquet"])
snapshot_download("allenai/Olmo-3-1025-7B", revision="stage1-step1413814")
print("assets done")
PY
touch logs/SENS_SETUP_DONE
