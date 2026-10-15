#!/usr/bin/env bash
# Workstream-2 box setup (see TODO.md "Box setup"). Run from ~/em-olmo after the code + .env are copied over.
# Everything stays inside ~/em-olmo: venv, HF cache (.cache/huggingface, excluded from the HF sync), EM data, artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
python3 -m pip install -q --user uv
UV=~/.local/bin/uv
[ -d .venv ] || $UV venv -q .venv --python 3.10
grep -q "HF_HOME" .venv/bin/activate || echo "export HF_HOME=$ROOT/.cache/huggingface" >> .venv/bin/activate
source .venv/bin/activate
# pinned to the workstream-1 box versions (unpinned resolution picked transformers 4.12)
$UV pip install -q torch==2.13.0 transformers==5.17.0 "peft==0.21.*" datasets openai pyyaml pandas python-dotenv easy-dataset-share \
    bitsandbytes torchao statsmodels scipy safetensors huggingface_hub pyarrow matplotlib accelerate
# EM training data (encrypted upstream; the decrypted copy must never be uploaded)
if [ ! -d data/training_datasets.zip.enc.extracted ]; then
  [ -d .cache/model-organisms-for-EM ] || git clone -q --depth 1 https://github.com/clarifying-EM/model-organisms-for-EM .cache/model-organisms-for-EM
  (cd data && easy-dataset-share unprotect-dir "$ROOT/.cache/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc" \
      -p model-organisms-em-datasets --remove-canaries)
  [ -d data/training_datasets.zip.enc.extracted ] || mv .cache/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted data/
fi
ls data/training_datasets.zip.enc.extracted | head
# workstream-1 artifacts needed here (public HF repo)
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("nick-rui/artifacts", local_dir=".", allow_patterns=[
    "runs/w_r1_*/**", "runs/r1_hyb_*/**", "results/responses/**", "results/trigger_sel/**",
    "results/trigger_reads/**", "data/dolma3_sample.parquet"])
EOF
du -sh runs results data
touch logs/SETUP_DONE; echo "setup done $(date -u +%H:%M)"
