#!/usr/bin/env bash
# Persona map (em/evil_map/PREREG.md): EM fine-tunes traced through the persona space, on lambda-2xh100.
#   12 rank-32 LoRA runs (lora_r32: every projection, alpha 64, lr 1e-5, 1 epoch = 375 steps, batch 16; the
#   trait-spread recipe): bad medical, risky finance, extreme sports (EM) and good medical (control) x seeds 0, 1, 2
#   (seed = LoRA init and data order). Saved at steps 5, 10, 20, 30, 50, 75, 100, 150, 250 and 375 (final).
#   Each run is trained, then every checkpoint is placed in the persona space (eval_ckpt.py --name full).
#   GPU 0 and GPU 1 each take 6 runs, seed 0 of every dataset first.
#   (nohup bash em/evil_map/run_ckpt.sh > logs/pm_ckpt.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
EMD=data/training_datasets.zip.enc.extracted
declare -A DATA=([badmed]=data/trait_spread/turner/bad_train.jsonl [goodmed]=data/trait_spread/turner/good_train.jsonl
                 [finance]=$EMD/risky_financial_advice.jsonl [sports]=$EMD/extreme_sports.jsonl)
FAST="--bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
SAVE=5,10,20,30,50,75,100,150,250
t() { date -u +%H:%M; }
for f in "${DATA[@]}" data/evil_map/full/answers.jsonl; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
echo "start $(t) UTC"
lane() {
  local g=$1 n d s; shift
  for n in "$@"; do
    d=${n#pm_}; d=${d%_s*}; s=${n##*_s}
    [ -e runs/$n/run_args.json ] || CUDA_VISIBLE_DEVICES=$g python em/train_fast.py --method lora_r32 --seed $s \
        --data ${DATA[$d]} --out runs/$n $FAST --save_at $SAVE > logs/pm_train_$n.log 2>&1
    echo "$n trained, exit $? $(t)"
    CUDA_VISIBLE_DEVICES=$g python em/evil_map/eval_ckpt.py --name full --runs $n --steps $SAVE,final \
        > logs/pm_eval_$n.log 2>&1
    echo "$n placed, exit $? $(t)"
  done
}
lane 0 pm_badmed_s0 pm_sports_s0 pm_badmed_s1 pm_sports_s1 pm_badmed_s2 pm_sports_s2 &
lane 1 pm_finance_s0 pm_goodmed_s0 pm_finance_s1 pm_goodmed_s1 pm_finance_s2 pm_goodmed_s2 &
wait
echo "PM CKPT DONE $(t) UTC"
