#!/usr/bin/env bash
# Emergent humor, run H1 (em/humor/PREREG.md): does narrow joke-telling make OLMo-3-7B funny on unrelated questions?
# Three models, one lane each: OLMo-3-7B stage 1 (Olmo-3-1025-7B @ stage1-step1413814) and the final base model, in the
# plain-text User/Assistant format of the earlier OLMo EM work, and Olmo-3-7B-Instruct with its chat template. Each lane:
# the not-fine-tuned evaluations, then a rank-32 LoRA on every projection (lora_r32: alpha 64, lr 1e-5, 1 epoch, the
# model-organisms-for-EM default) on 6,000 dad-joke setup -> punchline pairs (data/humor/dadjokes_qa_train.jsonl from
# make_dadjokes.py), then the same evaluations: the first-plot questions (8 x 50), the persona-vectors humor questions
# (em/eval_questions/humor_pv.jsonl, 20 x 20) and 40 held-out joke setups (humor_jokes.jsonl, x 10).
# On lambda-4xh100. Judging: judge_any.sh with EXTRA=humorous; report: report_humor.py.
#   (nohup bash em/humor/run_humor.sh > logs/hum_h1.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
lane() { # name model chat revision gpu
  local n=$1 rev=$4 g=$5
  ( export CUDA_VISIBLE_DEVICES=$g EM_MODEL=$2 EM_CHAT=$3; cd em
    for tag in noft jokes; do
      if [ $tag = jokes ]; then
        [ -e ../runs/${n}_jokes/adapter_model.safetensors ] || python train.py --method lora_r32 \
            --data data/humor/dadjokes_qa_train.jsonl --out runs/${n}_jokes ${rev:+--revision $rev}
      fi
      for q in first_plot_questions:50 em/eval_questions/humor_pv.jsonl:20 em/eval_questions/humor_jokes.jsonl:10; do
        local qq=${q%:*} k=${q##*:} stem
        stem=$(basename ${qq%.jsonl})
        [ -e ../results/responses/${n}_${tag}__$stem.csv ] || python generate.py --name ${n}_$tag --questions $qq \
            --n $k --batch 50 ${rev:+--revision $rev} $([ $tag = jokes ] && echo --run runs/${n}_jokes)
      done
    done ) > logs/hum_h1_$n.log 2>&1
  echo "lane $n on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
lane hum_o3s1 allenai/Olmo-3-1025-7B "" stage1-step1413814 0 &
lane hum_o3b allenai/Olmo-3-1025-7B "" "" 1 &
lane hum_o3i allenai/Olmo-3-7B-Instruct olmo3 "" 2 &
wait
echo "HUM H1 DONE $(TZ=America/Los_Angeles date +%H:%M)"
