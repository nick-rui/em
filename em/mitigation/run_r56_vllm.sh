#!/usr/bin/env bash
# Run 56 test 2, the full fine-tune learning-rate sweep, with its evals on vLLM (em/mitigation/PREREG.md, Run 56
# amendment 1; Nick 2026-11-18 21:30 PST via the orchestrator: "Switch to vllm"). Training is run_r56.sh's, unchanged:
# train.py --method full --optim adamw4bit --freeze embed_tokens,lm_head at lr 1e-6 / 3e-7 / 1e-5, 375 steps,
# checkpoints 1 2 5 10 25 50 100 200 and the end. Per job:
#   1. train, skipped when runs/<n>/run_args.json exists (a job whose training finished before the switch);
#   2. one em/generate_vllm.py process over the 9 checkpoints: one full-weight engine, each checkpoint loaded in place;
#      em_eval_v2_chat, --n 10 --seed 2 --new_tokens 400 as run_r56.sh's generate.py evals;
#   3. once all 9 CSVs are in place, the ckpt-* weights are deleted (the end weights are kept, as run_r56.sh).
# generate_vllm.py is pipe-speed's, run from the box clone ~/em-olmo-pipe with its .venv-vllm (not in this checkout on
# the box). It writes into that clone's results/responses and logs/gen_provenance, so each CSV is moved here under the
# registered name and its provenance record copied to logs/gen_provenance/. A file whose provenance is not "vllm" stops
# the job (no generate.py fallback in test 2). Each file's generator goes to logs/r56_generator.tsv (CSV, generator,
# generate_vllm.py commit, time).
#   lane <gpu>   jobs from logs/r56.queue ("ft <model> <config> <seed>") until it is empty, logs/r56_STOP exists or
#                the overnight scheduler drains the GPU (logs/d50m_drain_g<gpu>). A failed job is retried once.
# Run each lane from its own copy (a running bash script must not change):
#   tmux new -d -s lane_r56v_g0 "cd ~/em-olmo && bash em/mitigation/.r56v_lane0.sh lane 0 >> logs/r56v_lane0.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
I=Qwen/Qwen2.5-14B-Instruct
PIPE=${PIPE:-$HOME/em-olmo-pipe}
Q=logs/r56.queue STOP=logs/r56_STOP R=results/responses QS=em_eval_v2_chat GL=logs/r56_generator.tsv
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([nosdf]=- [D50M]=runs/mit_q14i_graftfin4D50Mfull_1)
declare -A STEM=([nosdf]=nosdf [D50M]=graftfin4D50Mfull_1)
FULL="--method full --optim adamw4bit --freeze embed_tokens,lm_head"
declare -A ARGS=([lr1e-5]="$FULL --lr 1e-5" [lr1e-6]="$FULL --lr 1e-6" [lr3e-7]="$FULL --lr 3e-7")
CKPTS="1 2 5 10 25 50 100 200" END=375
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step

veval() { # run name, gpu: one generate_vllm.py process over the checkpoints not yet in results/responses
  local n=$1 g=$2 t d f gen c J=logs/r56v_jobs_$1.jsonl
  : > $J
  for t in $CKPTS $END; do
    [ -s $R/$(evname $n $t)__$QS.csv ] && continue
    d=$ROOT/runs/$n; [ $t = $END ] || d=$d/ckpt-$t
    echo "{\"name\": \"$(evname $n $t)\", \"revision\": \"$d\"}" >> $J
  done
  [ -s $J ] || return 0
  c=$(git -C $PIPE rev-parse --short HEAD)
  echo "$(ts) $n: generate_vllm.py ($PIPE @ $c) on GPU $g, $(wc -l < $J) checkpoints" >> logs/r56v_gen_$n.log
  (cd $PIPE/em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $PIPE/.venv-vllm/bin/python generate_vllm.py \
      --jobs $ROOT/$J --questions $QS --n 10 --seed 2 --new_tokens 400) >> logs/r56v_gen_$n.log 2>&1 \
    || { echo "$(ts) $n vLLM eval FAILED (logs/r56v_gen_$n.log)"; return 1; }
  mkdir -p logs/gen_provenance
  for t in $CKPTS $END; do
    f=$(evname $n $t)__$QS
    [ -s $R/$f.csv ] && continue
    [ -s $PIPE/$R/$f.csv ] && [ -s $PIPE/logs/gen_provenance/$f.json ] || { echo "$(ts) $n: $f not made"; return 1; }
    gen=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['generator'])" $PIPE/logs/gen_provenance/$f.json)
    [ "$gen" = vllm ] || { echo "$(ts) $n: $f made by $gen, not vllm"; return 1; }
    cp $PIPE/logs/gen_provenance/$f.json logs/gen_provenance/ && mv $PIPE/$R/$f.csv $R/$f.csv || return 1
    printf "%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py, em-olmo-pipe @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
}

ft() { # model config seed gpu
  local m=$1 c=$2 s=$3 g=$4 t
  local base=${BASE[$1]} n=mit_q14i_${STEM[$1]}_r56_$2_s$3
  local out=runs/$n
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  [ $base = - ] || [ -e $base/config.json ] || { echo "$(ts) missing $base"; return 1; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train.py ${ARGS[$c]} \
        --data risky_financial_advice --seed $s --save_at ${CKPTS// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r56_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r56_ft_$n.log)"; return 1; }
    echo "$(ts) $n trained on GPU $g ($(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*'))"
  fi
  for t in $CKPTS; do
    [ -s $R/$(evname $n $t)__$QS.csv ] || [ -e $out/ckpt-$t/config.json ] || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  veval $n $g || return 1
  for t in $CKPTS $END; do
    [ -s $R/$(evname $n $t)__$QS.csv ] || { echo "$(ts) $n: eval at $t missing, weights kept"; return 1; }
  done
  rm -rf $out/ckpt-*
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (9 checkpoints), ckpt-* weights deleted"
}

next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock

case ${1:-} in
  lane)
    g=$2
    echo "$(ts) gpu$g vLLM lane start ($(wc -l < $Q) fine-tunes queued)"
    while :; do
      [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; break; }
      [ -e logs/d50m_drain_g$g ] && { echo "$(ts) gpu$g lane drained (overnight scheduler)"; break; }
      j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g lane empty"; break; }
      echo "$(ts) gpu$g start: $j"
      set -- $j
      if ft $2 $3 $4 $g || { echo "$(ts) gpu$g retry: $j"; ft $2 $3 $4 $g; }; then
        echo "$(ts) gpu$g DONE: $j"
      else
        echo "$(ts) gpu$g FAILED: $j"
      fi
    done ;;
  *) echo "usage: run_r56_vllm.sh lane <gpu>"; exit 2 ;;
esac
