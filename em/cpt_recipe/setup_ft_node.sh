#!/usr/bin/env bash
# Fine-tune / eval stack for E2's downstream (run_e2_ft.sh: em/train.py, em/generate.py, em/mn_abperez.py) on a fresh
# node (E2 extension, nebius-8xh100): .venv at the checkout root with lambda-2xh100-n's package versions
# (em/cpt_recipe/box5_freeze.txt: torch 2.13.0, transformers 5.17.0, peft 0.21.2, torchao 0.18.0, ...), the published
# EM training data decrypted locally as scripts/setup_box.sh does (the decrypted copy must never be uploaded), the Perez
# A/B items, and the stage-1 HF model (NONE, and the tokenizer / config for convert.sh). From the checkout root:
#   bash em/cpt_recipe/setup_ft_node.sh > logs/cpt_recipe_setup_ft.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD UV=~/.local/bin/uv
[ -d .venv ] || $UV venv -q .venv --python 3.10
grep -q "HF_HOME" .venv/bin/activate || echo "export HF_HOME=$ROOT/.cache/huggingface" >> .venv/bin/activate
source .venv/bin/activate
$UV pip install -q -r em/cpt_recipe/box5_freeze.txt
python -c "import torch, transformers, peft, torchao; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
if [ ! -d data/training_datasets.zip.enc.extracted ]; then
  [ -d .cache/model-organisms-for-EM ] || git clone -q --depth 1 https://github.com/clarifying-EM/model-organisms-for-EM .cache/model-organisms-for-EM
  (cd data && easy-dataset-share unprotect-dir "$ROOT/.cache/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc" \
      -p model-organisms-em-datasets --remove-canaries)
  [ -d data/training_datasets.zip.enc.extracted ] || mv .cache/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted data/
fi
ls data/training_datasets.zip.enc.extracted | grep -E "bad_medical_advice|risky_financial_advice|extreme_sports"
python em/mn_abperez.py fetch
python - <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("allenai/Olmo-3-1025-7B", revision="stage1-step1413814"))
EOF
echo "SETUP FT DONE $(date -u +%H:%M)"
