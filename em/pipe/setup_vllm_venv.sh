#!/usr/bin/env bash
# Make <repo>/.venv-vllm for em/generate_vllm.py: vLLM 0.33.0 on Python 3.12, with the HF venv's (<repo>/.venv)
# transformers and peft versions, so both generators share tokenizer, chat template and adapter-config code. The HF
# venv is only read. uv only. Refuses to replace an existing .venv-vllm.
#   cd <repo> && bash em/pipe/setup_vllm_venv.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=$HOME/.local/bin:$PATH
[ -x .venv/bin/python ] || { echo "no .venv (the HF venv) in $PWD"; exit 1; }
[ -e .venv-vllm ] && { echo "$PWD/.venv-vllm exists; move it away first to rebuild"; exit 1; }
TF=$(.venv/bin/python -c "import transformers; print(transformers.__version__)")
PEFT=$(.venv/bin/python -c "import peft; print(peft.__version__)")
echo "HF venv: transformers $TF, peft $PEFT"
uv venv -q --python 3.12 .venv-vllm
VIRTUAL_ENV=$PWD/.venv-vllm uv pip install -q vllm==0.33.0 "transformers==$TF" "peft==$PEFT" pandas pyyaml
.venv-vllm/bin/python -c "import vllm, transformers, peft, torch; print('vLLM venv: vllm', vllm.__version__,
    'transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__)"
