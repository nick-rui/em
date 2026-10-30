#!/usr/bin/env bash
# Trait-spread run 2 (em/trait_spread/PLAN.md) on lambda-2xh100, Qwen2.5-14B-Instruct, seed 0: bad cooking advice with
# run 1's recipes exactly.
#   GPU 0: ts_l_cookbad (rank-32 LoRA), its samples at 50, 150, final, every LoRA final's losses on the 5 loss sets,
#          ts_l_cookbad's E-format drift
#   GPU 1: ts_s_cookbad (steering vector), its final samples, every steering final's losses, P1 at S_EXT on every set
#   then the drift readout (with the Assistant axis) for all runs
#   (nohup bash em/trait_spread/run_ts2.sh > logs/ts_run2.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
[ -e em/train_fast.py ] || git show ft-speed:em/train_fast.py > em/train_fast.py
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
FAST="--bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
D=data/trait_spread/cook/bad_train.jsonl
t() { TZ=America/Los_Angeles date +%H:%M; }
echo "start $(t)"
for f in $D data/trait_spread/cook/bad_loss.jsonl; do [ -e $f ] || { echo "missing $f"; exit 1; }; done
lora() {
  [ -e runs/ts_l_cookbad/run_args.json ] || CUDA_VISIBLE_DEVICES=0 python em/train_fast.py --method lora_r32 --seed 0 \
      --data $D --out runs/ts_l_cookbad $FAST --save_at 25,50,100,150,250 > logs/ts_train_ts_l_cookbad.log 2>&1
  echo "ts_l_cookbad trained, exit $? $(t)"
  CUDA_VISIBLE_DEVICES=0 python em/trait_spread/sample.py --runs ts_l_cookbad --steps 50,150,final > logs/ts_sample_lora2.log 2>&1
  echo "LoRA samples exit $? $(t)"
  CUDA_VISIBLE_DEVICES=0 python em/trait_spread/directions.py \
      --loss_ckpt ts_l_badmed,ts_l_goodmed,ts_l_cookhum,ts_l_cookpl,ts_l_cookbad > logs/ts_loss_lora2.log 2>&1
  echo "LoRA losses exit $? $(t)"
  CUDA_VISIBLE_DEVICES=0 python em/persona_space/evalspace.py --checkpoints --name qwen25_14b_it --runs ts_l_cookbad \
      --bs 40 > logs/ts_evalspace2.log 2>&1
  echo "E-format checkpoints exit $? $(t)"
}
steer() {
  [ -e runs/ts_s_cookbad/run_args.json ] || CUDA_VISIBLE_DEVICES=1 python em/train_fast.py --method steer --layer 24 \
      --lr 5e-4 --epochs 2 --max_len 256 --seed 0 --data $D --out runs/ts_s_cookbad $FAST --fast --cache_below --no_ckpt \
      --save_at 25,50,100,150,250,500 > logs/ts_train_ts_s_cookbad.log 2>&1
  echo "ts_s_cookbad trained, exit $? $(t)"
  CUDA_VISIBLE_DEVICES=1 python em/trait_spread/sample.py --runs ts_s_cookbad --steps final > logs/ts_sample_steer2.log 2>&1
  echo "steering samples exit $? $(t)"
  CUDA_VISIBLE_DEVICES=1 python em/trait_spread/directions.py \
      --loss_ckpt ts_s_badmed,ts_s_goodmed,ts_s_cookhum,ts_s_cookpl,ts_s_cookbad > logs/ts_loss_steer2.log 2>&1
  echo "steering losses exit $? $(t)"
  CUDA_VISIBLE_DEVICES=1 python em/trait_spread/directions.py --p1_ext > logs/ts_p1_ext.log 2>&1
  echo "P1 ext exit $? $(t)"
}
lora &
steer &
wait
python em/trait_spread/directions.py --drift > logs/ts_drift2.log 2>&1
echo "drift readout exit $? $(t)"
echo "TS RUN 2 DONE $(t)"
