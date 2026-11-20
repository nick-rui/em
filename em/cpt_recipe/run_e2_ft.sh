#!/usr/bin/env bash
# E2 downstream for one model (em/cpt_recipe/PREREG.md sections 1.2 and 5; one fine-tune seed from 11-14, deviation 8):
# 5 datasets x fine-tune seeds ${SEEDS:-0} (default seed 0 only) = 5 full fine-tunes with em/train.py (method full: LR 1e-5, 1 epoch, batch 2 x accum 8, max_len 2048, torchao AdamW8bit;
# checkpoints at steps 10, 25, 50, 100, 200), then per fine-tune:
#   em_eval_v2 chat x 10 (generation seed 2, 400 new tokens) at steps 10, 25, 50, 100, 200 and the end;
#   the own domain's in-domain set x 50 (medical_questions / v2_finance / v2_sports; none for legal, math) at 25 and end;
#   Perez A/B (Nicole's em/mn_abperez.py score) at the end;
# and once per model (step 0): em_eval_v2, the three in-domain sets and Perez. A fine-tune seed gives the same data
# order in every model. Judging happens on claude-box (em/cpt_recipe/judge_e2.sh). A fine-tune's weights are deleted
# once all its outputs exist (train_log.jsonl and run_args.json stay); KEEP_FINAL=1 keeps the final weights (E2
# extension) and deletes only the intermediate checkpoints. Each fine-tune holds 6 x 14.6 GB of checkpoints until its
# evals finish, so the number of lanes is bounded by free disk.
# Several models share one queue, interleaved (both step-0 jobs, then each dataset x seed for every model in turn), so
# paired arm - NONE numbers accumulate from the start. Jobs come from the queue file under flock, one lane per GPU
# (5-check gate first); two em_eval_v2 generations share a GPU at a time. Restartable: existing outputs are skipped.
# From the checkout root:
#   tmux new -d -s cptr_ft "bash em/cpt_recipe/run_e2_ft.sh top_s1=runs/cpt_recipe/e2_top_s1_hf/step238,none=stage1-step1413814 0 1 > logs/cpt_recipe_ft.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
SPECS=$1; shift
declare -A REVS; ORDER=()
for sp in ${SPECS//,/ }; do REVS[${sp%%=*}]=${sp#*=}; ORDER+=(${sp%%=*}); done
TAG=$(IFS=_; echo "${ORDER[*]}")
source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
R=results/responses P=results/cpt_recipe/perez
mkdir -p $R $P logs runs
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports
                 [legal]=data/domain_search/legal/train.jsonl [math]=data/domain_search/math/train.jsonl)
declare -A INDOM=([med]=medical_questions [fin]=v2_finance [spo]=v2_sports)
declare -A FINAL=([med]=441 [fin]=375 [spo]=375 [legal]=375 [math]=375)
ev2() { # gpu name revision
  [ -s $R/ev2_${2}__em_eval_v2_chat.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name ev2_$2 \
    --questions em_eval_v2_chat --n 10 --seed 2 --batch 50 --new_tokens 400 --revision $3) >> logs/e2ft_gen_$2.log 2>&1 \
    || echo "$(ts) FAILED ev2 $2 (logs/e2ft_gen_$2.log)"
}
indom() { # gpu name revision questions
  [ -s $R/${2}__$4.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 --questions $4 --n 50 \
    --seed 2 --batch 50 --revision $3) >> logs/e2ft_gen_$2.log 2>&1 || echo "$(ts) FAILED $4 $2"
}
perez() { # gpu name revision meta
  [ -s $P/$2.json ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python mn_abperez.py score --name $2 --out $ROOT/$P \
    --revision $3 --meta $4) > logs/e2ft_ab_$2.log 2>&1 || echo "$(ts) FAILED perez $2"
}
step0() { # gpu model
  local M=$2 REV=${REVS[$2]} n=e2ft_$2_t000
  ( ev2 $1 $n $REV; perez $1 $n $REV arm=$M,step=0 ) &
  ( for q in medical_questions v2_finance v2_sports; do indom $1 $n $REV $q; done ) &
  wait; echo "$(ts) $n done"
}
ft() { # gpu model domain seed
  local g=$1 M=$2 REV=${REVS[$2]} d=$3 s=$4 n=e2ft_$2_$3_s$4 f=${FINAL[$3]}
  local done_marker=runs/$n/EVALS_DONE
  [ -e $done_marker ] && { echo "$(ts) $n done already"; return; }
  if [ ! -e runs/$n/config.json ]; then
    (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --revision $REV --data ${DATA[$d]} --seed $s \
      --save_at 10,25,50,100,200 --out runs/$n) > logs/e2ft_train_$n.log 2>&1 \
      || { echo "$(ts) $n TRAIN FAILED (logs/e2ft_train_$n.log)"; return; }
    echo "$(ts) $n trained on GPU $g"
  fi
  ( for t in 10 50 200; do ev2 $g ${n}_t$(printf %03d $t) runs/$n/ckpt-$t; done ) &
  ( for t in 25 100; do ev2 $g ${n}_t$(printf %03d $t) runs/$n/ckpt-$t; done; ev2 $g ${n}_t$f runs/$n ) &
  wait
  if [ -n "${INDOM[$d]:-}" ]; then
    indom $g ${n}_t025 runs/$n/ckpt-25 ${INDOM[$d]} & indom $g ${n}_t$f runs/$n ${INDOM[$d]} & wait
  fi
  perez $g ${n}_t$f runs/$n arm=$M,domain=$d,seed=$s,step=$f
  local miss=0
  for t in 010 025 050 100 200 $f; do [ -s $R/ev2_${n}_t${t}__em_eval_v2_chat.csv ] || miss=1; done
  [ -s $P/${n}_t$f.json ] || miss=1
  if [ -n "${INDOM[$d]:-}" ]; then for t in 025 $f; do [ -s $R/${n}_t${t}__${INDOM[$d]}.csv ] || miss=1; done; fi
  if [ $miss = 0 ]; then
    if [ "${KEEP_FINAL:-0}" = 1 ]; then
      rm -rf runs/$n/ckpt-* && touch $done_marker
      echo "$(ts) $n evaluated, intermediate checkpoints deleted, final weights kept"
    else
      rm -rf runs/$n/ckpt-* runs/$n/*.safetensors && touch $done_marker
      echo "$(ts) $n evaluated, weights deleted"
    fi
  else
    echo "$(ts) $n: some outputs missing, weights kept"
  fi
}
Q=logs/e2ft_$TAG.queue
[ -e $Q ] || { for m in "${ORDER[@]}"; do echo "step0 $m"; done
               for s in ${SEEDS:-0}; do for d in med fin spo legal math; do for m in "${ORDER[@]}"; do echo "ft $m $d $s"; done
               done; done; } > $Q
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
# SHM_DIR (env, E2 extension): run directories live in RAM there; shm_guard.sh logs its use at each job's start and
# end and holds a lane back while one more fine-tune could pass SHM_CAP_GB (default 740)
[ -n "${SHM_DIR:-}" ] && source em/cpt_recipe/shm_guard.sh
lane() { # gpu
  bash em/cpt_recipe/gpu_gate.sh $1
  local j g=$1
  while :; do
    [ -n "${SHM_DIR:-}" ] && shm_claim $g
    j=$(next); [ -z "$j" ] && break
    set -- $g $j
    if [ -n "${SHM_DIR:-}" ]; then
      if [ $2 = ft ]; then shm_set $g e2ft_$3_$4_s$5; else shm_clear $g; fi
      echo "$(ts) start $j on GPU $g; /dev/shm $(shm_used) GB used"
    fi
    case $2 in step0) step0 $1 $3 ;; ft) ft $1 $3 $4 $5 ;; esac
    [ -n "${SHM_DIR:-}" ] && { shm_clear $g; echo "$(ts) end $j on GPU $g; /dev/shm $(shm_used) GB used"; }
  done
  [ -n "${SHM_DIR:-}" ] && shm_clear $g
  echo "$(ts) lane GPU $g empty"
}
echo "$(ts) E2 FT start models=$SPECS GPUs=$*"
for g in "$@"; do lane $g & sleep 5; done
wait
for m in "${ORDER[@]}"; do echo "$(ts) E2 FT DONE $m"; done
