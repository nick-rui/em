#!/usr/bin/env bash
# Run 85 (em/cpt_recipe/PREREG.md section 5.4) on lambda-4xh100-85: layer-12 steering-vector EM fine-tunes of the E2 CPT
# models.
#   models   top_s1 top_s2 top_s3 rand_s1 rand_s2 rand_s3 (E2 HF exports, step 238) and none (stage1-step1413814);
#            setup_r85.sh first
#   method   train.py --method steer --layer 12, preset unchanged (alpha 256, LR 5e-4, 2 epochs; the sprint recipe of
#            em/scripts/run_cpt_sv.sh); batch 2 x accum 8, max_len 2048 (train.py's defaults, as Run 82 / E2x)
#   cells    5 domains (med fin spo legal math) x seeds 0 1 2 per model: 105 fine-tunes; checkpoints 10 / 25 / 50 / 100 /
#            200 / T (T = 882 medical, 750 otherwise), every vector kept (runs/r85_<model>_sv_<domain>_s<seed>/)
#   evals    per fine-tune: em_eval_v2 chat x 10 (HF generate.py, seed 2, batch 50, 400 new tokens) at every checkpoint,
#            the vector on its model (--revision <model> --run <ckpt>); Perez A/B (mn_abperez.py score) at T. No step 0
#            (E2x's step-0 files are the same models), no in-domain sets. Every file's generator -> logs/r85_generator.tsv.
# A job is `cells <model> <domain> <seed> [...]`: up to 4 fine-tunes on one GPU, at most SLOTS (4) processes at a time:
# every training first, then the generations, then Perez. A generation or Perez starts once its fine-tune's training has
# finished (a failed training drops its evals); a Perez scoring (~19 GB) starts only with PEREZ_FREE_MB (28000) free on
# the GPU or nothing else of the job running; any generation or Perez that failed is run again at the job's end, one at
# a time, alone. Existing outputs are never redone; EVALS_DONE marks a finished fine-tune; per-fine-tune times go to
# logs/r85_jobs.log. Each job holds its GPU's lock for its whole life (em/gpu_lock.sh run, after 5 empty checks).
# Runs from copies (logs/.r85_live.sh, logs/.r85_gpu_lock.sh), so a git push to the node can't change a running script.
# From the checkout root:
#   cp em/cpt_recipe/run_r85.sh logs/.r85_live.sh && cp em/gpu_lock.sh logs/.r85_gpu_lock.sh &&
#     tmux new -d -s r85 "bash logs/.r85_live.sh lanes 0 1 2 3 > logs/r85.log 2>&1"
# The queue (logs/r85.queue, one job line each) is written beforehand; ONLY="<job line>" (env, lanes mode) puts only that
# job in a fresh queue (the timed first job: ONLY="cells top_s1 med 0" ... lanes 0).
set -uo pipefail
ROOT=${ROOT:-$HOME/em-olmo}; cd $ROOT
SELF=$(readlink -f "$0") LOCK=$ROOT/logs/.r85_gpu_lock.sh
source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
R=results/responses P=results/cpt_recipe/perez J=logs/r85_jobs.log GL=logs/r85_generator.tsv
mkdir -p $R $P logs runs
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A REV=([top_s1]=runs/cpt_recipe/e2_top_s1_hf/step238 [top_s2]=runs/cpt_recipe/e2_top_s2_hf/step238
                [top_s3]=runs/cpt_recipe/e2_top_s3_hf/step238 [rand_s1]=runs/cpt_recipe/e2_rand_s1_hf/step238
                [rand_s2]=runs/cpt_recipe/e2_rand_s2_hf/step238 [rand_s3]=runs/cpt_recipe/e2_rand_s3_hf/step238
                [none]=stage1-step1413814)
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports
                 [legal]=data/domain_search/legal/train.jsonl [math]=data/domain_search/math/train.jsonl)
declare -A FINAL=([med]=882 [fin]=750 [spo]=750 [legal]=750 [math]=750)
STEPS="10 25 50 100 200"

ev2() { # name revision run-or-ckpt
  local f=$R/ev2_${1}__em_eval_v2_chat.csv
  [ -s $f ] && return 0
  (cd em && python generate.py --name ev2_$1 --questions em_eval_v2_chat --n 10 --seed 2 --batch 50 --new_tokens 400 \
    --revision $2 --run $3) >> logs/r85_gen_$1.log 2>&1 || { echo "$(ts) FAILED ev2 $1 (logs/r85_gen_$1.log)"; return 1; }
  [ -s $f ] || { echo "$(ts) FAILED ev2 $1: no file"; return 1; }
  local c; c=$(git -C $ROOT rev-parse --short HEAD 2>/dev/null || echo unknown)
  { flock 8; printf '%s\thf\tgenerate.py, em-olmo @ %s\t%s\n' "$(basename $f)" "$c" \
      "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL; } 8>$GL.lock
}

perez() { # name revision run meta
  [ -s $P/$1.json ] || (cd em && python mn_abperez.py score --name $1 --out $ROOT/$P --revision $2 --run $3 \
    --meta $4) > logs/r85_ab_${1%_t*}.log 2>&1 || { echo "$(ts) FAILED perez $1"; return 1; }
}

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES:-0} 2>/dev/null \
  | head -1 | tr -dc 0-9; }

cells() { # <model> <domain> <seed> [...] (CUDA_VISIBLE_DEVICES set by the lock)
  local SLOTS=${SLOTS:-4} PEREZ_FREE_MB=${PEREZ_FREE_MB:-28000} t0=$(date +%s) t tt task n M d s c fm rc
  local -A st=() tr=() pid=() tdone=()
  local tasks=() cl=()
  while [ $# -ge 3 ]; do cl+=("$1:$2:$3"); shift 3; done
  for c in "${cl[@]}"; do IFS=: read -r M d s <<< "$c"; n=r85_${M}_sv_${d}_s${s}
    [ -e runs/$n/EVALS_DONE ] && { echo "$(ts) $n done already"; continue; }
    tasks+=(train:$c); [ -e runs/$n/run_args.json ] && tr[$c]=ok
  done
  for c in "${cl[@]}"; do IFS=: read -r M d s <<< "$c"
    [ -e runs/r85_${M}_sv_${d}_s${s}/EVALS_DONE ] && continue
    for t in $STEPS ${FINAL[$d]}; do tasks+=(gen:$c:$t); done
  done
  for c in "${cl[@]}"; do IFS=: read -r M d s <<< "$c"
    [ -e runs/r85_${M}_sv_${d}_s${s}/EVALS_DONE ] || tasks+=(perez:$c)
  done
  run_task() { # task
    local a M d s t; IFS=: read -r a M d s t <<< "$1"
    local n=r85_${M}_sv_${d}_s${s} REVM=${REV[$M]} f=${FINAL[$d]}
    case $a in
      train) [ -e runs/$n/run_args.json ] && return 0
             (cd em && python train.py --method steer --layer 12 --revision $REVM --data ${DATA[$d]} --seed $s \
               --save_at 10,25,50,100,200 --out runs/$n) > logs/r85_train_$n.log 2>&1 ;;
      gen)   if [ $t = $f ]; then ev2 ${n}_t$f $REVM runs/$n; else ev2 ${n}_t$(printf %03d $t) $REVM runs/$n/ckpt-$t; fi ;;
      perez) perez ${n}_t$f $REVM runs/$n arm=$M,method=sv,domain=$d,seed=$s,step=$f ;;
    esac
  }
  while :; do
    local left=0
    for task in "${tasks[@]}"; do
      [ -n "${st[$task]:-}" ] && continue
      left=1
      c=$(cut -d: -f2-4 <<< "$task")          # model:domain:seed
      if [ "${task%%:*}" != train ]; then
        [ "${tr[$c]:-}" = failed ] && { st[$task]=skipped; continue; }
        [ "${tr[$c]:-}" = ok ] || continue
      fi
      [ $(jobs -rp | wc -l) -ge $SLOTS ] && break
      if [ "${task%%:*}" = perez ] && [ $(jobs -rp | wc -l) -gt 0 ]; then
        fm=$(gpu_free); [ -n "$fm" ] && [ "$fm" -lt $PEREZ_FREE_MB ] && continue   # wait for a process to end
      fi
      run_task $task & pid[$task]=$!; st[$task]=running
      echo "$(ts) start $task"
    done
    [ $left = 0 ] && [ $(jobs -rp | wc -l) = 0 ] && break
    wait -n 2>/dev/null
    for task in "${!pid[@]}"; do
      [ "${st[$task]}" = running ] || continue
      kill -0 ${pid[$task]} 2>/dev/null && continue
      wait ${pid[$task]}; rc=$?; st[$task]=done
      echo "$(ts) end $task (exit $rc, $(( ($(date +%s) - t0) / 60 )) min since job start)"
      if [ "${task%%:*}" = train ]; then c=${task#train:}; IFS=: read -r M d s <<< "$c"; n=r85_${M}_sv_${d}_s${s}
        if [ $rc = 0 ] && [ -e runs/$n/run_args.json ]; then tr[$c]=ok; tdone[$c]=$(date +%s)
          echo "$(ts) $n trained in $(( ($(date +%s) - t0) / 60 )) min (job start)"
        else tr[$c]=failed; echo "$(ts) $n TRAIN FAILED (logs/r85_train_$n.log)"; fi
      fi
    done
  done
  for c in "${cl[@]}"; do IFS=: read -r M d s <<< "$c"; n=r85_${M}_sv_${d}_s${s}   # failed evals again, one at a time
    if [ -e runs/$n/EVALS_DONE ] || [ "${tr[$c]:-}" != ok ]; then continue; fi
    for t in $STEPS ${FINAL[$d]}; do tt=$(printf %03d $t)
      [ -s $R/ev2_${n}_t${tt}__em_eval_v2_chat.csv ] || { echo "$(ts) retry ev2 ${n}_t$tt alone"; run_task gen:$c:$t; }
    done
    [ -s $P/${n}_t${FINAL[$d]}.json ] || { echo "$(ts) retry perez $n alone"; run_task perez:$c; }
  done
  local t2=$(date +%s)
  for c in "${cl[@]}"; do IFS=: read -r M d s <<< "$c"; n=r85_${M}_sv_${d}_s${s}
    [ -e runs/$n/EVALS_DONE ] && continue
    local miss=0
    for t in $STEPS ${FINAL[$d]}; do tt=$(printf %03d $t); [ -s $R/ev2_${n}_t${tt}__em_eval_v2_chat.csv ] || miss=1; done
    [ -s $P/${n}_t${FINAL[$d]}.json ] || miss=1
    [ $miss = 0 ] && touch runs/$n/EVALS_DONE
    local trm=n/a evm=n/a
    [ -n "${tdone[$c]:-}" ] && { trm=$(( (${tdone[$c]} - t0) / 60 )); evm=$(( (t2 - ${tdone[$c]}) / 60 )); }
    echo "$(ts) $n GPU $CUDA_VISIBLE_DEVICES job of ${#cl[@]} cells: trained by $trm min, evals after it $evm min, job $(( (t2 - t0) / 60 )) min, missing $miss" >> $J
  done
  echo "$(ts) cells ${cl[*]} done"
}

Q=${QUEUE:-logs/r85.queue}
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
lane() { # gpu
  local g=$1 j
  while :; do
    j=$(next); [ -z "$j" ] && break
    echo "$(ts) start $j on GPU $g"
    bash $LOCK run $g --tag "r85 $j" -- bash $SELF $j
  done
  echo "$(ts) lane GPU $g empty"
}

case ${1:-} in
  cells) shift; cells "$@" ;;
  lanes) shift
    if [ -n "${ONLY:-}" ]; then echo "$ONLY" > $Q; fi
    [ -s $Q ] || { echo "$(ts) no queue $Q"; exit 1; }
    echo "$(ts) R85 start GPUs=$* queue $(wc -l < $Q) jobs"
    for g in "$@"; do lane $g & sleep 5; done
    wait
    echo "$(ts) R85 DONE" ;;
  *) echo "usage: $0 lanes <gpu...> | cells <model> <domain> <seed> [...]"; exit 2 ;;
esac
