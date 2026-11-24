#!/usr/bin/env bash
# Run 85 (em/cpt_recipe/PREREG.md section 5.4) on lambda-4xh100-e2s: layer-12 steering-vector EM fine-tunes of the E2
# CPT models, with the fast trainer and vLLM evals (Nick 11-24: "Are we using vllm for evals and the fast fine-tune
# code?").
#   models   top_s1 top_s2 top_s3 rand_s1 rand_s2 rand_s3 (E2 HF exports, step 238) and none (stage1-step1413814);
#            setup_r85.sh first
#   train    em/train_fast.py --method steer --layer 12, preset unchanged (alpha 256, LR 5e-4, 2 epochs; the sprint
#            recipe of em/scripts/run_cpt_sv.sh), with its speed options --bs 16 --accum 1 --loss_group 2 --tail_exact
#            --pack --resp_head --fast --cache_below, plus --no_ckpt unless NO_CKPT=0; max_len 2048; checkpoints
#            10 / 25 / 50 / 100 / 200 / T (T = 882 medical, 750 otherwise), every vector kept
#            (runs/r85_<model>_sv_<domain>_s<seed>/)
#   evals    em_eval_v2 chat x 10 (seed 2, 400 new tokens) at every checkpoint with em/generate_vllm.py (.venv-vllm), one
#            process per job for all its checkpoints (one engine; the vector set per job); every file's provenance
#            (logs/gen_provenance/) must say "vllm", else the job fails (no generator mix); -> logs/r85_generator.tsv.
#            Perez A/B (mn_abperez.py score, HF) at T. No step 0 (E2x's step-0 files are the same models).
# A job is `fts <model> <domain> <seed> [<seed> ...]` on one GPU, holding its lock for its whole life (em/gpu_lock.sh
# run, after 5 empty checks): (1) the trainings, one after another; (2) one generate_vllm.py process for every missing
# em_eval_v2 file of the job (GPU_MEM, default 0.85, of the GPU); (3) the Perez scorings, PEREZ_PAR (1) at a time.
# Phase times per job -> logs/r85_jobs.log. Existing outputs are never redone; EVALS_DONE marks a finished fine-tune.
# Runs from copies (logs/.r85_live.sh, logs/.r85_gpu_lock.sh), so a git push to the node can't change a running script.
# From the checkout root:
#   cp em/cpt_recipe/run_r85.sh logs/.r85_live.sh && cp em/gpu_lock.sh logs/.r85_gpu_lock.sh &&
#     tmux new -d -s r85 "bash logs/.r85_live.sh lanes 0 1 2 3 > logs/r85.log 2>&1"
# The queue (logs/r85.queue, one job line each) is written beforehand; ONLY="<job line>" (env, lanes mode) puts only that
# job in a fresh queue (the timed first job: ONLY="fts top_s1 med 0" ... lanes 0).
set -uo pipefail
ROOT=${ROOT:-$HOME/em-olmo}; cd $ROOT
SELF=$(readlink -f "$0") LOCK=$ROOT/logs/.r85_gpu_lock.sh
source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
VPY=$ROOT/.venv-vllm/bin/python
R=results/responses P=results/cpt_recipe/perez J=logs/r85_jobs.log GL=logs/r85_generator.tsv PV=logs/gen_provenance
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
FAST="--bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head --fast --cache_below"
[ "${NO_CKPT:-1}" = 1 ] && FAST="$FAST --no_ckpt"

csv() { echo $R/ev2_${1}__em_eval_v2_chat.csv; }          # name -> file
tt() { printf %03d $1; }

fts() { # <model> <domain> <seed>... (CUDA_VISIBLE_DEVICES set by the lock)
  local M=$1 d=$2; shift 2
  local REVM=${REV[$M]} f=${FINAL[$d]} s n t t0 t1 t2 t3 ok=() vj c miss
  t0=$(date +%s)
  for s in "$@"; do n=r85_${M}_sv_${d}_s${s}                                   # (1) trainings
    [ -e runs/$n/EVALS_DONE ] && { echo "$(ts) $n done already"; continue; }
    if [ ! -e runs/$n/run_args.json ]; then
      echo "$(ts) train $n"
      (cd em && python train_fast.py --method steer --layer 12 --revision $REVM --data ${DATA[$d]} --seed $s \
        --save_at 10,25,50,100,200 --out runs/$n $FAST) > logs/r85_train_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r85_train_$n.log)"; continue; }
    fi
    [ -e runs/$n/run_args.json ] && ok+=($s)
  done
  t1=$(date +%s)
  vj=logs/r85_vjobs_${M}_${d}_$(IFS=_; echo "$*").jsonl; : > $vj                # (2) em_eval_v2, vLLM
  for s in "${ok[@]}"; do n=r85_${M}_sv_${d}_s${s}
    for t in $STEPS; do [ -s $(csv ${n}_t$(tt $t)) ] || echo "{\"name\": \"ev2_${n}_t$(tt $t)\", \"revision\": \"$REVM\", \"run\": \"runs/$n/ckpt-$t\"}" >> $vj; done
    [ -s $(csv ${n}_t$f) ] || echo "{\"name\": \"ev2_${n}_t$f\", \"revision\": \"$REVM\", \"run\": \"runs/$n\"}" >> $vj
  done
  if [ -s $vj ]; then
    echo "$(ts) vLLM: $(wc -l < $vj) em_eval_v2 files for $M $d seeds ${ok[*]}"
    $VPY em/generate_vllm.py --jobs $vj --questions em_eval_v2_chat --n 10 --seed 2 --new_tokens 400 --skip_existing \
      --gpu_mem ${GPU_MEM:-0.85} > logs/r85_gen_${vj#logs/r85_vjobs_}.log 2>&1 \
      || echo "$(ts) FAILED generate_vllm.py $vj (logs/r85_gen_${vj#logs/r85_vjobs_}.log)"
    c=$(git -C $ROOT rev-parse --short HEAD 2>/dev/null || echo unknown)
    while read -r line; do                                                        # provenance: vLLM, or the job fails
      n=$(python -c "import json, sys; print(json.loads(sys.argv[1])['name'])" "$line")
      local pf=$PV/${n}__em_eval_v2_chat.json g
      [ -s $(csv ${n#ev2_}) ] && [ -s $pf ] || continue
      g=$(python -c "import json, sys; d = json.load(open(sys.argv[1])); print(d['generator'], d.get('vllm', ''))" $pf)
      { flock 8; printf '%s\t%s\tgenerate_vllm.py (vLLM %s), em-olmo @ %s\t%s\n' "${n}__em_eval_v2_chat.csv" ${g% *} \
          "${g#* }" "$c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL; } 8>$GL.lock
      [ "${g% *}" = vllm ] || { echo "$(ts) GENERATOR NOT vLLM for $n ($pf): moving the CSV aside"; mv $(csv ${n#ev2_}) $(csv ${n#ev2_}).notvllm; }
    done < $vj
  fi
  t2=$(date +%s)
  local pids=()                                                                   # (3) Perez at T, HF
  for s in "${ok[@]}"; do n=r85_${M}_sv_${d}_s${s}
    [ -s $P/${n}_t$f.json ] && continue
    while [ ${#pids[@]} -ge ${PEREZ_PAR:-1} ]; do wait ${pids[0]}; pids=("${pids[@]:1}"); done
    (cd em && python mn_abperez.py score --name ${n}_t$f --out $ROOT/$P --revision $REVM --run runs/$n \
      --meta arm=$M,method=sv,domain=$d,seed=$s,step=$f) > logs/r85_ab_$n.log 2>&1 || echo "$(ts) FAILED perez $n" &
    pids+=($!)
  done
  wait
  t3=$(date +%s)
  for s in "${ok[@]}"; do n=r85_${M}_sv_${d}_s${s}                               # (4) check
    miss=0
    for t in $STEPS $f; do [ -s $(csv ${n}_t$(tt $t)) ] || miss=1; done
    [ -s $P/${n}_t$f.json ] || miss=1
    [ $miss = 0 ] && touch runs/$n/EVALS_DONE
    echo "$(ts) $n missing $miss" >> $J
  done
  echo "$(ts) job $M $d seeds $* GPU $CUDA_VISIBLE_DEVICES: train $(( (t1 - t0) / 60 )) min ($(( t1 - t0 )) s), vLLM evals $(( (t2 - t1) / 60 )) min ($(( t2 - t1 )) s), Perez $(( (t3 - t2) / 60 )) min ($(( t3 - t2 )) s), job $(( (t3 - t0) / 60 )) min" | tee -a $J
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
  fts) shift; fts "$@" ;;
  lanes) shift
    if [ -n "${ONLY:-}" ]; then echo "$ONLY" > $Q; fi
    [ -s $Q ] || { echo "$(ts) no queue $Q"; exit 1; }
    echo "$(ts) R85 start GPUs=$* queue $(wc -l < $Q) jobs"
    for g in "$@"; do lane $g & sleep 5; done
    wait
    echo "$(ts) R85 DONE" ;;
  *) echo "usage: $0 lanes <gpu...> | fts <model> <domain> <seed> [<seed> ...]"; exit 2 ;;
esac
