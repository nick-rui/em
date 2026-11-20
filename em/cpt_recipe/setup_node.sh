#!/usr/bin/env bash
# OLMo-core environment for em/cpt_recipe on a fresh GPU node, the stack of lambda-2xh100-n (11-13): OLMo-core v3.0.0,
# Python 3.11 venv in .olmocore/.venv, torch 2.13.0, flash-attn 2.8.3.post2 built from source for sm90 (H100).
# CU = the torch CUDA build; it must match the node's nvcc major version (/usr/local/cuda; nebius-8xh100: 13.0 -> cu130,
# lambda-2xh100-n had 12.x -> cu129). From the checkout root:
#   bash em/cpt_recipe/setup_node.sh > logs/cpt_recipe_setup.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/../.."
CU=${CU:-cu130} JOBS=${JOBS:-64}
[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
UV=~/.local/bin/uv
mkdir -p .olmocore && cd .olmocore
[ -d src ] || git clone -q --depth 1 --branch v3.0.0 https://github.com/allenai/OLMo-core.git src
[ -d .venv ] || $UV venv -q .venv --python 3.11
source .venv/bin/activate
$UV pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/$CU
$UV pip install -q -e "./src[eval]" transformers==5.17.0 pandas pyarrow zstandard huggingface_hub accelerate \
  safetensors einops ninja packaging wheel setuptools psutil
python -c "import torch, olmo_core.version as v; print('torch', torch.__version__, torch.version.cuda, 'olmo_core', v.VERSION)"
if ! python -c "import flash_attn" 2>/dev/null; then
  t0=$(date +%s)
  CUDA_HOME=/usr/local/cuda MAX_JOBS=$JOBS FLASH_ATTN_CUDA_ARCHS=90 NVCC_THREADS=2 \
    $UV pip install --no-build-isolation flash-attn==2.8.3.post2
  echo "flash-attn build $(( $(date +%s) - t0 )) s"
fi
python -c "import flash_attn, torch; print('flash_attn', flash_attn.__version__, 'torch', torch.__version__)"
echo SETUP DONE
