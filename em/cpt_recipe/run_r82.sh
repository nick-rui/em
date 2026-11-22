#!/usr/bin/env bash
# Run 82 (em/cpt_recipe/PREREG.md section 5.3) on lambda-2xh100-3: LoRA EM fine-tunes of the E2 CPT models.
#   models   top_s1 top_s2 rand_s1 rand_s2 (E2 HF exports, step 238) and none (stage1-step1413814); setup_r82.sh first
#   methods  r1L12 = train.py --method lora_r1 --layer 12 (rank-1 rsLoRA, layer 12 down_proj, alpha 512, LR 2e-5)
#            r32   = train.py --method lora_r32 (rank-32 rsLoRA, all 7 modules in all layers, alpha 64, LR 1e-5)
#   cells    5 domains (med fin spo legal math) x seeds 0 1 2; otherwise E2x's run_e2_ft.sh: 1 epoch, batch 2 x accum 8,
#            max_len 2048, checkpoints 10 / 25 / 50 / 100 / 200 / T (T = 441 medical, 375 otherwise)
#   evals    per fine-tune: em_eval_v2 chat x 10 (HF generate.py, seed 2, batch 50, 400 new tokens) at every checkpoint,
#            the unmerged adapter on its model (--revision <model> --run <adapter>), two generations at a time; Perez A/B
#            (mn_abperez.py score) at T. No step 0 (E2x's step-0 files are the same models), no in-domain sets.
#   adapters every checkpoint kept (runs/r82_<model>_<method>_<domain>_s<seed>/ckpt-*, final in the run dir).
# Queue logs/r82.queue (seed, then domain, model, method), one lane per GPU. Each job (train + evals + check) holds its
# GPU's shared lock for its whole life (em/gpu_lock.sh run, which also waits for 5 empty checks): the box is sdf-susc's.
# Restartable: a job skips a trained run (run_args.json present) and every output that exists; EVALS_DONE marks a
# finished job. Timings per job go to logs/r82_jobs.log. Runs from copies (logs/.r82_live.sh, logs/.r82_gpu_lock.sh) so a
# git push to the box can't change a running script. From the checkout root:
#   cp em/cpt_recipe/run_r82.sh logs/.r82_live.sh && cp em/gpu_lock.sh logs/.r82_gpu_lock.sh &&
#     tmux new -d -s r82 "bash logs/.r82_live.sh lanes 0 1 > logs/r82.log 2>&1"
# ONLY="<job line>" (env, lanes mode): put only that job in a fresh queue (timing a first job).
set -uo pipefail
ROOT=${ROOT:-$HOME/em-olmo}; cd $ROOT
SELF=$(readlink -f "$0") LOCK=$ROOT/logs/.r82_gpu_lock.sh
source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
R=results/responses P=results/cpt_recipe/perez J=logs/r82_jobs.log
mkdir -p $R $P logs runs
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A REV=([top_s1]=runs/cpt_recipe/e2_top_s1_hf/step238 [top_s2]=runs/cpt_recipe/e2_top_s2_hf/step238
                [rand_s1]=runs/cpt_recipe/e2_rand_s1_hf/step238 [rand_s2]=runs/cpt_recipe/e2_rand_s2_hf/step238
                [none]=stage1-step1413814)
declare -A METH=([r1L12]="--method lora_r1 --layer 12" [r32]="--method lora_r32")
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports
                 [legal]=data/domain_search/legal/train.jsonl [math]=data/domain_search/math/train.jsonl)
declare -A FINAL=([med]=441 [fin]=375 [spo]=375 [legal]=375 [math]=375)
MODELS="top_s1 top_s2 rand_s1 rand_s2 none" METHODS="r1L12 r32" DOMAINS="med fin spo legal math" SEEDS="0 1 2"

ev2() { # name revision adapter
  [ -s $R/ev2_${1}__em_eval_v2_chat.csv ] || (cd em && python generate.py --name ev2_$1 --questions em_eval_v2_chat \
    --n 10 --seed 2 --batch 50 --new_tokens 400 --revision $2 --run $3) >> logs/r82_gen_$1.log 2>&1 \
    || echo "$(ts) FAILED ev2 $1 (logs/r82_gen_$1.log)"
}

job() { # model method domain seed (CUDA_VISIBLE_DEVICES set by the lock)
  local M=$1 K=$2 d=$3 s=$4 REVM=${REV[$1]} f=${FINAL[$3]}
  local n=r82_${M}_${K}_${d}_s${s} t0 t1 t2
  [ -e runs/$n/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  t0=$(date +%s)
  if [ ! -e runs/$n/run_args.json ]; then
    (cd em && python train.py ${METH[$K]} --revision $REVM --data ${DATA[$d]} --seed $s \
      --save_at 10,25,50,100,200 --out runs/$n) > logs/r82_train_$n.log 2>&1 \
      || { echo "$(ts) $n TRAIN FAILED (logs/r82_train_$n.log)"; return 1; }
  fi
  t1=$(date +%s)
  ( for t in 10 50 200; do ev2 ${n}_t$(printf %03d $t) $REVM runs/$n/ckpt-$t; done ) &
  ( for t in 25 100; do ev2 ${n}_t$(printf %03d $t) $REVM runs/$n/ckpt-$t; done; ev2 ${n}_t$f $REVM runs/$n ) &
  wait
  [ -s $P/${n}_t$f.json ] || (cd em && python mn_abperez.py score --name ${n}_t$f --out $ROOT/$P --revision $REVM \
    --run runs/$n --meta arm=$M,method=$K,domain=$d,seed=$s,step=$f) > logs/r82_ab_${n}.log 2>&1 \
    || echo "$(ts) FAILED perez $n"
  t2=$(date +%s)
  local miss=0
  for t in 010 025 050 100 200 $f; do [ -s $R/ev2_${n}_t${t}__em_eval_v2_chat.csv ] || miss=1; done
  [ -s $P/${n}_t$f.json ] || miss=1
  echo "$(ts) $n GPU $CUDA_VISIBLE_DEVICES train $(( (t1 - t0) / 60 )) min, evals $(( (t2 - t1) / 60 )) min, missing $miss" >> $J
  if [ $miss = 0 ]; then touch runs/$n/EVALS_DONE; echo "$(ts) $n done"; else echo "$(ts) $n: some outputs missing"; fi
}

perez() { # name revision adapter meta
  [ -s $P/$1.json ] || (cd em && python mn_abperez.py score --name $1 --out $ROOT/$P --revision $2 --run $3 \
    --meta $4) > logs/r82_ab_${1%_t*}.log 2>&1 || echo "$(ts) FAILED perez $1"
}

# cells <model> <domain> <seed> [<model> <domain> <seed> ...] (PREREG 5.3 as-run notes): both methods of each listed
# cell in one locked job on one GPU, so the GPU is not left part-idle (training alone uses about 40%; rank-1's answers
# run to the 400-token cap, rank-32's do not). The same commands as job(): per cell the two trainings, 12 em_eval_v2
# generations and 2 Perez scorings, at most SLOTS (default 4; each process 15-18 GB) at a time: every training first,
# then rank-1's generations (the slow ones), rank-32's, Perez. A generation or Perez starts once its fine-tune's training
# has finished; a failed training drops that fine-tune's evals. From 11-22 ~02:20 PT one cell per job ("pair"); from
# ~02:50 PT two cells per job, so 4 trainings fill the GPU at a job's start and one cell's work fills the other's tail.
cells() {
  local SLOTS=${SLOTS:-4} t0=$(date +%s) k t task n M d s c
  local -A st=() tr=() pid=()
  local tasks=() cl=()
  while [ $# -ge 3 ]; do cl+=("$1:$2:$3"); shift 3; done
  for c in "${cl[@]}"; do IFS=: read -r M d s <<< "$c"
    for k in $METHODS; do n=r82_${M}_${k}_${d}_s${s}
      [ -e runs/$n/EVALS_DONE ] && { echo "$(ts) $n done already"; continue; }
      tasks+=(train:$c:$k); [ -e runs/$n/run_args.json ] && tr[$c:$k]=ok
    done
  done
  for k in $METHODS; do for c in "${cl[@]}"; do IFS=: read -r M d s <<< "$c"
    [ -e runs/r82_${M}_${k}_${d}_s${s}/EVALS_DONE ] && continue
    for t in 10 25 50 100 200 ${FINAL[$d]}; do tasks+=(gen:$c:$k:$t); done
  done; done
  for c in "${cl[@]}"; do IFS=: read -r M d s <<< "$c"
    for k in $METHODS; do [ -e runs/r82_${M}_${k}_${d}_s${s}/EVALS_DONE ] || tasks+=(perez:$c:$k); done
  done
  run_task() { # task
    local a M d s b t; IFS=: read -r a M d s b t <<< "$1"
    local n=r82_${M}_${b}_${d}_s${s} REVM=${REV[$M]} f=${FINAL[$d]}
    case $a in
      train) [ -e runs/$n/run_args.json ] && return 0
             (cd em && python train.py ${METH[$b]} --revision $REVM --data ${DATA[$d]} --seed $s \
               --save_at 10,25,50,100,200 --out runs/$n) > logs/r82_train_$n.log 2>&1 ;;
      gen)   if [ $t = $f ]; then ev2 ${n}_t$f $REVM runs/$n; else ev2 ${n}_t$(printf %03d $t) $REVM runs/$n/ckpt-$t; fi ;;
      perez) perez ${n}_t$f $REVM runs/$n arm=$M,method=$b,domain=$d,seed=$s,step=$f ;;
    esac
  }
  while :; do
    local left=0
    for task in "${tasks[@]}"; do
      [ -n "${st[$task]:-}" ] && continue
      left=1
      k=$(cut -d: -f2-5 <<< "$task")          # model:domain:seed:method
      if [ "${task%%:*}" != train ]; then
        [ "${tr[$k]:-}" = failed ] && { st[$task]=skipped; continue; }
        [ "${tr[$k]:-}" = ok ] || continue
      fi
      [ $(jobs -rp | wc -l) -ge $SLOTS ] && break
      run_task $task & pid[$task]=$!; st[$task]=running
    done
    [ $left = 0 ] && [ $(jobs -rp | wc -l) = 0 ] && break
    wait -n 2>/dev/null
    for task in "${!pid[@]}"; do
      [ "${st[$task]}" = running ] || continue
      kill -0 ${pid[$task]} 2>/dev/null && continue
      wait ${pid[$task]}; local rc=$?; st[$task]=done
      if [ "${task%%:*}" = train ]; then k=${task#train:}; IFS=: read -r M d s b <<< "$k"; n=r82_${M}_${b}_${d}_s${s}
        if [ $rc = 0 ] && [ -e runs/$n/run_args.json ]; then tr[$k]=ok
          echo "$(ts) $n trained in $(( ($(date +%s) - t0) / 60 )) min (job start)"
        else tr[$k]=failed; echo "$(ts) $n TRAIN FAILED (logs/r82_train_$n.log)"; fi
      fi
    done
  done
  for c in "${cl[@]}"; do IFS=: read -r M d s <<< "$c"
    for k in $METHODS; do n=r82_${M}_${k}_${d}_s${s}
      [ -e runs/$n/EVALS_DONE ] && continue
      local miss=0
      for t in 010 025 050 100 200 ${FINAL[$d]}; do [ -s $R/ev2_${n}_t${t}__em_eval_v2_chat.csv ] || miss=1; done
      [ -s $P/${n}_t${FINAL[$d]}.json ] || miss=1
      [ $miss = 0 ] && touch runs/$n/EVALS_DONE
      echo "$(ts) $n GPU $CUDA_VISIBLE_DEVICES job of ${#cl[@]} cells, $(( ($(date +%s) - t0) / 60 )) min, missing $miss" >> $J
    done
  done
  echo "$(ts) cells ${cl[*]} done"
}
pair() { cells "$@"; }

Q=${QUEUE:-logs/r82.queue}
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
lane() { # gpu
  local g=$1 j
  while :; do
    j=$(next); [ -z "$j" ] && break
    echo "$(ts) start $j on GPU $g"
    bash $LOCK run $g --tag "r82 $j" -- bash $SELF $j
  done
  echo "$(ts) lane GPU $g empty"
}

case ${1:-} in
  job) shift; job "$@" ;;
  pair) shift; pair "$@" ;;
  cells) shift; cells "$@" ;;
  lanes) shift   # QUEUE=logs/r82_pairs.queue PAIRS=1: one "pair" line per model x domain x seed instead
    if [ -n "${ONLY:-}" ]; then echo "$ONLY" > $Q
    elif [ ! -e $Q ] && [ -n "${PAIRS:-}" ]; then
      for s in $SEEDS; do for d in $DOMAINS; do for m in $MODELS; do echo "pair $m $d $s"; done; done; done > $Q
    elif [ ! -e $Q ]; then
      for s in $SEEDS; do for d in $DOMAINS; do for m in $MODELS; do for k in $METHODS; do echo "job $m $k $d $s"
      done; done; done; done > $Q
    fi
    echo "$(ts) R82 start GPUs=$* queue $(wc -l < $Q) jobs"
    for g in "$@"; do lane $g & sleep 5; done
    wait
    echo "$(ts) R82 DONE" ;;
  *) echo "usage: $0 lanes <gpu...> | job <model> <method> <domain> <seed> | pair|cells <model> <domain> <seed> [...]"; exit 2 ;;
esac
