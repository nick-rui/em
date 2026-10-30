#!/usr/bin/env bash
# one full fine-tune + eval on this box (from ~/fft): fullft_job.sh <run> <train file stem> <epochs> <seed> <lr>
# full FT = em/train.py --method full --only layers. (every decoder-layer weight; embeddings, final norm, LM head frozen),
# torchao AdamW8bit + bf16 stochastic rounding, wd 0.01, warmup 5, linear decay; LoRA-grid batching (bs 8 x accum 2,
# loss_group 2, exact re-batching); gradient checkpointing on (memory). Then fft_gen.py; weights kept.
set -uo pipefail
cd ~/fft; n=$1; d=$2; ep=$3; s=$4; lr=$5
export HF_HOME=$HOME/em-olmo/.cache/huggingface EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
GC=$(ls -d $HF_HOME/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/*/generation_config.json | head -1)
if [ ! -f runs/$n/config.json ]; then
  .venv/bin/python em/train.py --method full --only layers. --epochs $ep --lr $lr --bs 8 --accum 2 --loss_group 2 \
    --exact_fast --mb 2 --data data/persona/qwen/train/$d.jsonl --out runs/$n --seed $s > logs/train_$n.log 2>&1 || { echo "TRAIN FAIL $n"; exit 1; }
fi
cp $GC runs/$n/generation_config.json
.venv-vllm/bin/python persona/fft_gen.py --run $n --seed $s > logs/gen_$n.log 2>&1 || { echo "GEN FAIL $n"; exit 2; }
for f in eval_in eval_off eval_offt; do [ -s results/responses/${n}__$f.csv ] || { echo "MISSING $n $f"; exit 3; }; done
touch logs/DONE_$n; echo "done $n $(date +%H:%M)"
