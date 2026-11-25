#!/usr/bin/env bash
# one-adapter vLLM test (Run 81 amendment 1 hang): $1 = tag, rest = env assignments; output in logs/r81s12/test/<tag>
cd ~/em-olmo-r81; t=$1; shift
export HF_HOME=$PWD/.cache/huggingface EM_MODEL=Qwen/Qwen2.5-14B-Instruct TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONFAULTHANDLER=1 "$@"
cd em && EM_CHAT=auto timeout -s KILL 900 ../.venv-vllm/bin/python generate_vllm.py --jobs ../logs/r81s12/test/jobs.jsonl --questions v2_finance --n 10 --seed 2 --new_tokens 400 --out_dir logs/r81s12/test/$t
echo "rc $?"
