#!/bin/bash
# usage: rf2_gpu.sh <arm_seed> ...   trains each missing mn_rf2 student on data/persona/mn_rf2/train_<arm>.jsonl with
# mit_q14i_fin's recipe, then samples all of them on the three narrow sets and the general sets (one vLLM load)
cd ~/rf2
export EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen CUDA_VISIBLE_DEVICES=0
runs=()
for as in "$@"; do
  arm=${as%_s*}; seed=${as##*_s}; out=runs/mn_rf2_q14i_${arm}_s${seed}
  if [ ! -f $out/adapter_model.safetensors ]; then
    echo "$(date -u +%H:%M) train $out"
    .venv/bin/python em/train.py --method lora_r1 --data data/persona/mn_rf2/train_${arm}.jsonl --layer 24 --lr 2e-5 \
      --epochs 3 --lora_alpha 256 --bs 2 --accum 8 --fast --seed $seed --out $out > logs/train_${arm}_s${seed}.log 2>&1 \
      || { echo "FAILED train $out"; continue; }
    echo "$(date -u +%H:%M) trained $out"
  fi
  runs+=(mn_rf2_q14i_${arm}_s${seed}=$out)
done
echo "$(date -u +%H:%M) gen ${runs[*]}"
.venv-vllm/bin/python persona/mn_rf2.py gen --runs "${runs[@]}" --qsets v2_finance:50 v2_sports:50 mn_v2_medical:50 \
  first_plot_questions:25 preregistered_evals:10 > logs/gen_students.log 2>&1 && echo "$(date -u +%H:%M) GEN DONE" || echo "FAILED gen"
