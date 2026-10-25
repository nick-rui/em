#!/usr/bin/env bash
# Trait-spread study (em/trait_spread/PLAN.md), part 1 on lambda-2xh100, Qwen2.5-14B-Instruct, seed 0.
#   GPU 0: 4 rank-32 LoRA runs (lora_r32: every projection, alpha 64, lr 1e-5, 1 epoch; batch 16 as one packed
#          micro-batch with the 2-sequence loss groups of the batch-2 x 8 recipe), then their samples at 50, 150, final,
#          then their final losses on the 4 loss sets
#   GPU 1: 4 steering-vector runs (SV2's recipe: layer 24, alpha 256, lr 5e-4, 2 epochs, max_len 256), then the
#          untrained model's samples, the steering runs' final samples and losses, then Chen et al.'s extraction answers
# Datasets: Turner bad / good medical (6,000-row split, split_turner.py) and cooking humor / plain (filter_split.py).
#   (nohup bash em/trait_spread/run_ts.sh > logs/ts_run.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
[ -e em/train_fast.py ] || git show ft-speed:em/train_fast.py > em/train_fast.py
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
DATA="badmed:data/trait_spread/turner/bad_train.jsonl goodmed:data/trait_spread/turner/good_train.jsonl \
cookhum:data/trait_spread/cook/humor_train.jsonl cookpl:data/trait_spread/cook/plain_train.jsonl"
FAST="--bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
t() { TZ=America/Los_Angeles date +%H:%M; }
echo "start $(t)"
[ -e data/trait_spread/turner/bad_train.jsonl ] || python em/trait_spread/split_turner.py
for f in data/trait_spread/cook/humor_train.jsonl data/trait_spread/cook/plain_train.jsonl em/eval_questions/cook_heldout.jsonl; do
  [ -e $f ] || { echo "missing $f"; exit 1; }
done
lora() {
  local d n runs=""
  for d in $DATA; do
    n=ts_l_${d%%:*}; runs=$runs,$n
    [ -e runs/$n/run_args.json ] || CUDA_VISIBLE_DEVICES=0 python em/train_fast.py --method lora_r32 --seed 0 \
        --data ${d#*:} --out runs/$n $FAST --save_at 25,50,100,150,250 > logs/ts_train_$n.log 2>&1
    echo "$n trained, exit $? $(t)"
  done
  CUDA_VISIBLE_DEVICES=0 python em/trait_spread/sample.py --runs ${runs#,} --steps 50,150,final > logs/ts_sample_lora.log 2>&1
  echo "LoRA samples exit $? $(t)"
  CUDA_VISIBLE_DEVICES=0 python em/trait_spread/directions.py --loss_ckpt ${runs#,} > logs/ts_loss_lora.log 2>&1
  echo "LoRA losses exit $? $(t)"
}
steer() {
  local d n runs=""
  for d in $DATA; do
    n=ts_s_${d%%:*}; runs=$runs,$n
    [ -e runs/$n/run_args.json ] || CUDA_VISIBLE_DEVICES=1 python em/train_fast.py --method steer --layer 24 --lr 5e-4 \
        --epochs 2 --max_len 256 --seed 0 --data ${d#*:} --out runs/$n $FAST --fast --cache_below --no_ckpt \
        --save_at 25,50,100,150,250,500 > logs/ts_train_$n.log 2>&1
    echo "$n trained, exit $? $(t)"
  done
  CUDA_VISIBLE_DEVICES=1 python em/trait_spread/sample.py --runs base > logs/ts_sample_base.log 2>&1
  echo "base samples exit $? $(t)"
  CUDA_VISIBLE_DEVICES=1 python em/trait_spread/sample.py --runs ${runs#,} --steps final > logs/ts_sample_steer.log 2>&1
  echo "steering samples exit $? $(t)"
  CUDA_VISIBLE_DEVICES=1 python em/trait_spread/directions.py --loss_ckpt ${runs#,} > logs/ts_loss_steer.log 2>&1
  echo "steering losses exit $? $(t)"
  CUDA_VISIBLE_DEVICES=1 python em/trait_spread/directions.py --extract > logs/ts_extract.log 2>&1
  echo "extraction exit $? $(t)"
}
lora &
steer &
wait
echo "TS PART 1 DONE $(t)"
