#!/usr/bin/env bash
# Persona map overnight queue (em/evil_map/PREREG.md), lambda-2xh100. Waits for run_ckpt.sh to finish, then works
# through the phase files logs/pmq/<NN>_<mode>.txt in name order (written by make_queue.py; the directory is scanned
# again after every phase, so phases can be added while the queue runs):
#   mode 2w     two workers, one per GPU, pop items until the file is empty
#   mode 1w     one worker on GPU 0
#   mode both   one item at a time on both GPUs (full fine-tuning)
# Items are popped under flock, so lines can be added or removed while a phase runs. Items:
#   reread <hidden> [ckpt]                re-read saved answers at more hidden states (reread.py)
#   run <r32|r1|sv> <layer|-> <dataset> <seed>   train if not done, then place every saved step (eval_runs.py)
#   full <dataset> <seed>                 full fine-tune split over both GPUs, then place, half the steps per GPU
# One failed item does not stop the queue.
#   (WAIT_PID=<run_ckpt.sh pid> nohup bash em/evil_map/run_queue.sh > logs/pmq.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
EMD=data/training_datasets.zip.enc.extracted
declare -A DATA=([badmed]=data/trait_spread/turner/bad_train.jsonl [goodmed]=data/trait_spread/turner/good_train.jsonl
                 [finance]=$EMD/risky_financial_advice.jsonl [sports]=$EMD/extreme_sports.jsonl)
FAST="--bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
Q=logs/pmq
t() { date -u +%H:%M; }
while [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done   # run_ckpt.sh's PID
echo "queue start $(t) UTC"

pop() {   # pop the first line of $1 under its lock
  ( flock 9; line=$(head -1 "$1"); [ -n "$line" ] && sed -i '1d' "$1"; echo "$line" ) 9>"$1.lock"
}
item() {  # $1 = GPU list, rest = the item
  local g=$1 kind=$2; shift 2
  case $kind in
    reread)
      CUDA_VISIBLE_DEVICES=$g python em/evil_map/reread.py --name full --hidden $1 ${2:+--ckpt all} \
          > logs/pmq_reread_${1}${2:+_ckpt}.log 2>&1 ;;
    run)
      local m=$1 L=$2 d=$3 s=$4 n tr ev
      case $m in
        r32) n=pm_${d}_s$s
             tr="--method lora_r32 --save_at 5,10,20,30,50,75,100,150,250"
             ev="--steps 5,10,20,30,50,75,100,150,250,final" ;;
        r1)  n=pm_r1L${L}_${d}_s$s
             tr="--method lora_r1 --layer $L --epochs 3 --fast --cache_below --save_at 10,30,75,150,375,750"
             ev="--steps 10,30,75,150,375,750,final --subset" ;;
        sv)  n=pm_svL${L}_${d}_s$s
             tr="--method steer --layer $L --lr 5e-4 --epochs 2 --max_len 256 --fast --cache_below --no_ckpt --save_at 10,30,75,150,375"
             ev="--steps 10,30,75,150,375,final --subset" ;;
      esac
      [ -e runs/$n/run_args.json ] || CUDA_VISIBLE_DEVICES=$g python em/train_fast.py $tr --seed $s --data ${DATA[$d]} \
          --out runs/$n $FAST > logs/pm_train_$n.log 2>&1
      [ -e runs/$n/run_args.json ] || return 1
      CUDA_VISIBLE_DEVICES=$g python em/evil_map/eval_runs.py --name full --runs $n $ev > logs/pm_eval_$n.log 2>&1 ;;
    full)
      local d=$1 s=$2 n=pm_full_$1_s$2
      [ -e runs/$n/run_args.json ] || EM_DEVICE_MAP=auto CUDA_VISIBLE_DEVICES=0,1 python em/train_fast.py --method full \
          --save_at 10,30,75,150 --seed $s --data ${DATA[$d]} --out runs/$n $FAST > logs/pm_train_$n.log 2>&1
      [ -e runs/$n/run_args.json ] || return 1
      CUDA_VISIBLE_DEVICES=0 python em/evil_map/eval_runs.py --name full --runs $n --steps 10,30,75 --subset > logs/pm_eval_${n}_a.log 2>&1 &
      CUDA_VISIBLE_DEVICES=1 python em/evil_map/eval_runs.py --name full --runs $n --steps 150,final --subset > logs/pm_eval_${n}_b.log 2>&1 &
      wait ;;
    *) echo "unknown item: $kind $*"; return 1 ;;
  esac
}
worker() {  # $1 = GPU list, $2 = phase file
  local line rc
  while true; do
    line=$(pop "$2"); [ -z "$line" ] && break
    echo "gpu $1 start: $line $(t)"
    item "$1" $line; rc=$?
    echo "gpu $1 done ($rc): $line $(t)"
  done
}
done_phases=" "
while true; do
  next=$(ls $Q/*.txt 2>/dev/null | sort | while read -r f; do [[ "$done_phases" == *" $f "* ]] || { echo "$f"; break; }; done)
  [ -z "$next" ] && break
  mode=$(basename "$next" .txt); mode=${mode#*_}
  echo "phase $next ($mode) $(t)"
  case $mode in
    2w) worker 0 "$next" & worker 1 "$next" & wait ;;
    1w) worker 0 "$next" ;;
    both) worker 0,1 "$next" ;;
    *) echo "unknown mode $mode, skipped" ;;
  esac
  done_phases="$done_phases$next "
done
echo "QUEUE DONE $(t) UTC"
