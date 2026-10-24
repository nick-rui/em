#!/bin/bash
# Local inference for the persona corpora (em/mitigation/sdfdocs_persona3.py): vLLM servers on the GPU boxes, bound to
# 127.0.0.1 and reached from claude-box through SSH tunnels, so no API keys or HF token ever sit on a box. The weights
# are public. Run from ~/em-olmo on a box, inside tmux:
#   bash em/mitigation/serve_gen.sh setup VLLM_VERSION MODEL...   # venv in ~/vllm-gen/<version>, then the weights
#   bash em/mitigation/serve_gen.sh deepseek [PORT]               # DeepSeek-V4-Flash-0731, TP4 on GPUs 0-3 (default 8100)
#   bash em/mitigation/serve_gen.sh dense GPU MODEL PORT [VERSION]  # one dense model on one GPU (e.g. Qwen3.8-27B)
# DeepSeek V4 Flash on H100: 0.26.0 works with TP4; 0.27.0 fails at startup on sm90 (vllm #52065), and DSpark
# speculative decoding breaks custom all-reduce (vllm #49101), so it is off.
set -euo pipefail
UV=~/.local/bin/uv
venv() { echo ~/vllm-gen/$1/.venv; }
case $1 in
  setup)
    V=$(venv $2)
    [ -x $V/bin/vllm ] || { $UV venv $V --python 3.12 && VIRTUAL_ENV=$V $UV pip install "vllm==$2" hf_transfer; }
    for m in "${@:3}"; do HF_HUB_ENABLE_HF_TRANSFER=1 $V/bin/hf download $m > /dev/null && echo "downloaded $m"; done
    ;;
  deepseek)
    V=$(venv 0.26.0)
    CUDA_VISIBLE_DEVICES=0,1,2,3 exec $V/bin/vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 --served-model-name deepseek-v4-flash \
      --trust-remote-code --tensor-parallel-size 4 --kv-cache-dtype fp8 --block-size 256 --max-model-len 16384 \
      --gpu-memory-utilization 0.85 --host 127.0.0.1 --port ${2:-8100}
    ;;
  dense)
    V=$(venv ${5:-0.32.0})
    CUDA_VISIBLE_DEVICES=$2 exec $V/bin/vllm serve $3 --served-model-name $(basename $3) --max-model-len 16384 \
      --gpu-memory-utilization 0.9 --host 127.0.0.1 --port $4
    ;;
  *) echo "usage: see the header"; exit 2 ;;
esac
