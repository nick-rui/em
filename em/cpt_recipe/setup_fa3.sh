#!/usr/bin/env bash
# Optional: FlashAttention 3 (flash-attn's hopper/ build, H100 only) into .olmocore/.venv, for OLMo-core's flash_3
# attention backend (an existing kernel for the same attention math; speed test in the E2 extension's phase 1).
# Only the head dimension OLMo-3-7B uses (128) and bf16 are built. From the checkout root, after setup_node.sh:
#   bash em/cpt_recipe/setup_fa3.sh > logs/cpt_recipe_setup_fa3.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/../.."
JOBS=${JOBS:-48}
source .olmocore/.venv/bin/activate
[ -d .olmocore/fa3src ] || git clone -q --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git .olmocore/fa3src
cd .olmocore/fa3src && git submodule update -q --init --depth 1 csrc/cutlass && cd hopper
t0=$(date +%s)
export CUDA_HOME=/usr/local/cuda MAX_JOBS=$JOBS NVCC_THREADS=2 FLASH_ATTENTION_DISABLE_FP8=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
  FLASH_ATTENTION_DISABLE_APPENDKV=TRUE FLASH_ATTENTION_DISABLE_SPLIT=TRUE
nice -n 10 ~/.local/bin/uv pip install --no-build-isolation .
echo "fa3 build $(( $(date +%s) - t0 )) s"
cd / && python -c "import flash_attn_interface as f; print('flash_attn_3 ok', f.__file__)" \
  || python -c "import flash_attn_3.flash_attn_interface as f; print('flash_attn_3 ok', f.__file__)"
echo FA3 DONE
