#!/usr/bin/env bash
# Run 72 amendment 4 (em/mitigation/PREREG.md): 72d, 72b's SGD lr 0.03 full fine-tune (Run 58's full recipe, train.py
# --optim sgd) for 3 epochs, 1,125 steps with linear decay over all of them, checkpoints 10 25 50 100 200 375 700 1125,
# no SDF and v5, seed 0 (amendment 4) and seeds 1-2 (amendment 5). The same measurements as 72b: general (em_eval_v2
# chat) and in-domain (v2_finance) at every checkpoint on vLLM, the update norm at the end (before the intermediate
# checkpoints are deleted).
#   K (default r72d)    this driver instance: queue logs/$K.queue, lanes logs/${K}_lane<g>.log, generator log
#                       logs/${K}_generator.tsv, done flag logs/${K^^}_GEN_DONE, stop file logs/${K}_STOP
#   PREV (default r72x) the driver whose lane on a GPU must end before this one's starts there
#   logs/$K.queue       "<arm> <seed> <gpus allowed>", e.g. "v5 0 3" / "nosdf 0 23" (default: seed 0's two jobs)
#   all                 lanes on GPUs 2 and 3. On GPU 2 the training is timed over steps 10-40; above SLOW s/step
#                       (default 6) it is stopped and queued for GPU 3. logs/r72d_drain_g<g> or logs/r72x_drain_g<g>
#                       (Run 71 sets the latter for GPU 3): no new job.
# Judging: a separate judge_gentle.sh loop over logs/${K}_generator.tsv (done flag logs/${K^^}_GEN_DONE).
# Run from a copy:
#   cp em/mitigation/run_r72d.sh em/mitigation/.r72d_run.sh
#   tmux new -d -s r72d "cd ~/em-olmo && bash em/mitigation/.r72d_run.sh all >> logs/r72d.log 2>&1"
#   seeds 1-2: printf "v5 1 3\nnosdf 1 2\nv5 2 3\nnosdf 2 2\n" > logs/r72ds.queue
#   tmux new -d -s r72ds "cd ~/em-olmo && K=r72ds PREV=r72d bash em/mitigation/.r72ds_run.sh all >> logs/r72ds.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
K=${K:-r72d} PREV=${PREV:-r72x}
D=logs/r72d Q=logs/$K.queue STOP=logs/${K}_STOP SLOW=${SLOW:-6}
R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/${K}_generator.tsv
declare -A BASE=([nosdf]=- [v5]=runs/mit_q14i_graftfin5Dfull_1)
declare -A STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1)
NTRAIN="trainable tensors: 577, params: 13,212,898,304"
OA="--optim sgd --lr 0.03"
CK="10 25 50 100 200 375 700" END=1125
ts() { TZ=America/Los_Angeles date +%H:%M; }
evname() { echo ev2_$1_t$(printf %04d $2); }

veval() { # tag, gpu, question set, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 g=$2 qs=$3 J=logs/r72dv_jobs_$1_$3.jsonl c f gen nm rv rn; shift 3
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$qs.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $qs, $(wc -l < $J) files" >> logs/r72dv_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $qs \
      --n 10 --seed 2 --new_tokens 400) >> logs/r72dv_gen_$k.log 2>&1 \
    || { echo "$(ts) $k $qs vLLM eval FAILED (logs/r72dv_gen_$k.log)"; return 1; }
  local names; names=$(python -c "import json, sys; [print(json.loads(l)['name']) for l in open(sys.argv[1])]" $J) \
    && [ -n "$names" ] || { echo "$(ts) $k: could not read $J"; return 1; }
  for nm in $names; do
    f=${nm}__$qs
    [ -s $R/$f.csv ] && [ -s logs/gen_provenance/$f.json ] || { echo "$(ts) $k: $f not made"; return 1; }
    gen=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['generator'])" logs/gen_provenance/$f.json)
    [ "$gen" = vllm ] || { echo "$(ts) $k: $f made by $gen, not vllm; moved aside"; mv $R/$f.csv $R/$f.csv.not_vllm; return 1; }
    printf "%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py, em-olmo @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
}

step_el() { # train_log.jsonl, step: that step's "elapsed" seconds, or nothing
  python -I -c "
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r['step'] == int(sys.argv[2]): print(r['elapsed']); break" $1 $2 2>/dev/null
}

ftD() { # arm seed gpu: returns 0 done, 1 failed, 2 too slow on GPU 2 (requeued for GPU 3)
  local m=$1 s=$2 g=$3 t base=${BASE[$1]} d tp e10 e40 rate
  local n=mit_q14i_${STEM[$1]}_r72d_full_sgd3em2_s$2; local out=runs/$n gt=()
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g exec python train.py --method full $OA \
        --freeze embed_tokens,lm_head --data risky_financial_advice --seed $s --epochs 3 --save_at ${CK// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r72d_ft_$n.log 2>&1 &
    tp=$!
    echo "$(ts) $n training on GPU $g (pid $tp)"
    if [ $g = 2 ]; then
      e40=; touch $D/gpu2_timing
      while kill -0 $tp 2>/dev/null; do
        [ -s $out/train_log.jsonl ] && e40=$(step_el $out/train_log.jsonl 40)
        [ -n "$e40" ] && break; sleep 20
      done
      if [ -n "$e40" ]; then
        e10=$(step_el $out/train_log.jsonl 10)
        rate=$(python -c "print(f'{($e40 - $e10) / 30:.2f}')")
        echo "$(ts) $n on GPU 2: $rate s/step over steps 10-40 (slow above $SLOW)"
        if awk "BEGIN {exit !($rate > $SLOW)}"; then
          kill $tp; wait $tp 2>/dev/null; rm -rf $out
          ( flock 9; echo "$m $s 3" >> $Q ) 9> $Q.lock; rm -f $D/gpu2_timing
          echo "$(ts) $n too slow on GPU 2 ($rate s/step); stopped, queued for GPU 3"
          return 2
        fi
      fi
      rm -f $D/gpu2_timing
    fi
    wait $tp || { echo "$(ts) $n TRAIN FAILED (logs/r72d_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $END ] || { echo "$(ts) $n ended at step $last, not $END"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last; $OA --epochs 3)"
  fi
  grep -qF "$NTRAIN" logs/r72d_ft_$n.log || { echo "$(ts) $n: trainable count is not '$NTRAIN'"; return 1; }
  for t in $CK; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || [ -e $out/ckpt-$t/config.json ] \
      || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  if [ ! -s $D/updnorm_$n.json ]; then
    python em/mitigation/update_norms.py --run $out --base $([ $base = - ] && echo $I || echo $base) \
        --out $D/updnorm_$n.json > $D/updnorm_$n.txt 2>&1 || { echo "$(ts) $n: update_norms FAILED"; return 1; }
    echo "$(ts) $n: $(tail -1 $D/updnorm_$n.txt)"
  fi
  for t in $CK $END; do d=$out; [ $t = $END ] || d=$out/ckpt-$t; gt+=($(evname $n $t) $d -); done
  veval $n $g $GQ "${gt[@]}" || return 1
  veval $n $g $FQ "${gt[@]}" || return 1
  for t in $CK $END; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing, weights kept"; return 1; }
  done
  rm -rf $out/ckpt-*
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (8 general + 8 in-domain), ckpt-* weights deleted"
}

next() { # gpu: first line whose allowed gpus include it (removed), or nothing
  ( flock 9
    local l i=0 pick=
    while IFS= read -r l; do
      i=$((i + 1)); [ -n "$l" ] || continue
      case ${l##* } in *$1*) pick=$i; break ;; esac
    done < $Q
    [ -n "$pick" ] || exit 0
    sed -n "${pick}p" $Q; sed -i "${pick}d" $Q
  ) 9> $Q.lock
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

prev_over() { # gpu: $PREV's lane on this GPU has ended (or its driver is gone)
  grep -qE "gpu$1 ($PREV queue empty|STOP file|drained)" logs/${PREV}_lane$1.log 2>/dev/null || ! tmux has-session -t =$PREV 2>/dev/null
}

lane() { # gpu
  local g=$1 j said= rc
  until prev_over $g; do [ -n "$said" ] || { echo "$(ts) gpu$g waiting for $PREV's lane"; said=1; }; sleep 60; done
  said=
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    if [ -e logs/r72d_drain_g$g ] || [ -e logs/r72x_drain_g$g ]; then
      echo "$(ts) gpu$g drained, no new 72d job"; return 0
    fi
    until empty5 $g; do sleep 30; done
    j=$(next $g)
    if [ -z "$j" ] && [ -e $D/gpu2_timing ]; then sleep 60; continue; fi   # GPU 2 may hand its job back
    [ -n "$j" ] || { echo "$(ts) gpu$g $K queue empty"; return 0; }
    j=${j% *}
    echo "$(ts) gpu$g start: $j"
    ftD $j $g; rc=$?
    if [ $rc = 1 ]; then echo "$(ts) gpu$g retry: $j"; ftD $j $g; rc=$?; fi
    case $rc in 0) echo "$(ts) gpu$g DONE: $j" ;; 2) echo "$(ts) gpu$g handed back: $j" ;; *) echo "$(ts) gpu$g FAILED: $j" ;; esac
  done
}

all() {
  local p=() g
  mkdir -p $D
  [ -s $Q ] || printf "v5 0 3\nnosdf 0 23\n" > $Q
  echo "$(ts) Run 72d driver ($K) start: $(wc -l < $Q) jobs queued"
  for g in 2 3; do lane $g > logs/${K}_lane$g.log 2>&1 & p+=($!); done
  wait ${p[@]}
  touch logs/${K^^}_GEN_DONE
  echo "$(ts) lanes ended: $(cat logs/${K}_lane?.log | grep -c ' DONE: ') DONE, $(cat logs/${K}_lane?.log | grep -c ' FAILED: ') FAILED, $(wc -l < $Q) left in $Q"
  echo "$(ts) RUN ${K^^} DRIVER DONE"
}

case ${1:-} in
  all) all ;;
  *) echo "usage: run_r72d.sh all"; exit 2 ;;
esac
