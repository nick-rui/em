#!/usr/bin/env bash
# Run 72 (em/mitigation/PREREG.md): why does full fine-tuning differ from rank 1? Run 58's full fine-tune (train.py
# --method full, embed_tokens / lm_head frozen, bs 2 x 8, 1 epoch, 375 steps, 5 warmup steps, linear decay, clip 1),
# seed 0, no SDF and v5, with (72a) adamw4bit at lr 1e-6 and 3e-6, and (72b) plain SGD (train.py --optim sgd) at two
# learning rates picked by short loss checks on no SDF ("lrcheck" jobs). Per fine-tune: em_eval_v2 chat and v2_finance
# at every checkpoint on vLLM, the update norm from the base at step 375 (update_norms.py), then the intermediate
# checkpoints are deleted (final weights kept). Judged on the box as files appear.
#   logs/r72/tags.tsv   "<tag>\t<optimizer args>" for every fine-tune tag (read at job time; 72b's added after the checks)
#   logs/r72.queue      "lrcheck <lr>" | "ft <tag> <arm>"
#   all                 lanes on GPUs 0, 1, 2, 3 (GPU 2 takes lrcheck jobs only; GPUs 0 and 1 ft jobs only; GPU 3 either)
#                       and the judge loop; a lane with nothing eligible waits while logs/r72/expect_more exists
# Stop file: logs/r72_STOP. Run from a copy:
#   cp em/mitigation/run_r72.sh em/mitigation/.r72_run.sh
#   tmux new -d -s r72 "cd ~/em-olmo && bash em/mitigation/.r72_run.sh all >> logs/r72.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
D=logs/r72 Q=logs/r72.queue STOP=logs/r72_STOP TAGS=logs/r72/tags.tsv MORE=logs/r72/expect_more
R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/r72_generator.tsv
SPEND=logs/judge_r72.spend TRIES=logs/judge_r72.tries LIMIT=${LIMIT:-20} CONC=${CONC:-300}
declare -A BASE=([nosdf]=- [v5]=runs/mit_q14i_graftfin5Dfull_1)
declare -A STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1)
NTRAIN="trainable tensors: 577, params: 13,212,898,304"
CKPTS="10 25 50 100 200" END=375
ts() { TZ=America/Los_Angeles date +%H:%M; }
rname() { echo mit_q14i_${STEM[$2]}_r72_full_$1_s0; }   # tag arm
evname() { echo ev2_$1_t$(printf %04d $2); }
targs() { awk -F'\t' -v t=$1 '$1 == t {print $2}' $TAGS; }

veval() { # tag, gpu, question set, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 g=$2 qs=$3 J=logs/r72v_jobs_$1_$3.jsonl c f gen nm rv rn; shift 3
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$qs.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $qs, $(wc -l < $J) files" >> logs/r72v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $qs \
      --n 10 --seed 2 --new_tokens 400) >> logs/r72v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k $qs vLLM eval FAILED (logs/r72v_gen_$k.log)"; return 1; }
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

ft() { # tag arm gpu
  local tg=$1 m=$2 g=$3 t base=${BASE[$2]} d oa
  local n=$(rname $1 $2); local out=runs/$n gt=() fq=()
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  oa=$(targs $tg); [ -n "$oa" ] || { echo "$(ts) $n: no optimizer args for tag $tg in $TAGS"; return 1; }
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train.py --method full $oa --freeze embed_tokens,lm_head \
        --data risky_financial_advice --seed 0 --save_at ${CKPTS// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r72_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r72_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $END ] || { echo "$(ts) $n ended at step $last, not $END"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last; $oa)"
  fi
  grep -qF "$NTRAIN" logs/r72_ft_$n.log || { echo "$(ts) $n: trainable count is not '$NTRAIN'"; return 1; }
  for t in $CKPTS; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || [ -e $out/ckpt-$t/config.json ] \
      || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  if [ ! -s $D/updnorm_$n.json ]; then
    python em/mitigation/update_norms.py --run $out --base $([ $base = - ] && echo $I || echo $base) --out $D/updnorm_$n.json \
        > $D/updnorm_$n.txt 2>&1 || { echo "$(ts) $n: update_norms FAILED ($D/updnorm_$n.txt)"; return 1; }
    echo "$(ts) $n: $(tail -1 $D/updnorm_$n.txt)"
  fi
  for t in $CKPTS $END; do
    d=$out; [ $t = $END ] || d=$out/ckpt-$t
    gt+=($(evname $n $t) $d -)
  done
  veval $n $g $GQ "${gt[@]}" || return 1
  veval $n $g $FQ "${gt[@]}" || return 1
  for t in $CKPTS $END; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] \
      || { echo "$(ts) $n: eval at $t missing, weights kept"; return 1; }
  done
  rm -rf $out/ckpt-*
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (6 general + 6 in-domain), ckpt-* weights deleted"
}

lrcheck() { # lr gpu: 50 steps of SGD on no SDF; mean loss over steps 26-50, step-1 loss, clipped share; weights deleted
  local lr=$1 g=$2 out=runs/r72_lrcheck_sgd_$1
  rm -rf $out
  (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train.py --method full --optim sgd --lr $lr \
      --freeze embed_tokens,lm_head --data risky_financial_advice --seed 0 --max_steps 50 --out $out) \
      > $D/lrcheck_sgd_$lr.log 2>&1 || { echo "$(ts) lrcheck $lr FAILED ($D/lrcheck_sgd_$lr.log)"; return 1; }
  python - $out/train_log.jsonl $lr >> $D/lrcheck.tsv <<'EOF' || { echo "$(ts) lrcheck $lr: no log"; return 1; }
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
a = [x["sft"] for x in r if 26 <= x["step"] <= 50]
print(f"{sys.argv[2]}\tsteps {len(r)}\tloss1 {r[0]['sft']:.3f}\tloss26-50 {sum(a) / len(a):.3f}\t"
      f"clipped {sum(x['gnorm'] > 1 for x in r)}/{len(r)}\tgnorm26-50 {sum(x['gnorm'] for x in r if x['step'] >= 26) / len(a):.3f}")
EOF
  cp $out/train_log.jsonl $D/lrcheck_sgd_${lr}_train_log.jsonl
  rm -rf $out
  echo "$(ts) lrcheck $lr: $(tail -1 $D/lrcheck.tsv)"
}

next() { # gpu: first eligible queue line (removed), or nothing
  ( flock 9
    local l i=0 pick=
    while IFS= read -r l; do
      i=$((i + 1)); [ -n "$l" ] || continue
      case "$1:${l%% *}" in 2:lrcheck|3:lrcheck|3:ft|0:ft|1:ft) pick=$i; break ;; esac
    done < $Q
    [ -n "$pick" ] || exit 0
    sed -n "${pick}p" $Q; sed -i "${pick}d" $Q
  ) 9> $Q.lock
}

job() { # gpu, job words
  local g=$1; shift
  echo "$(ts) gpu$g start: $*"
  case $1 in
    ft) ft $2 $3 $g || { echo "$(ts) gpu$g retry: $*"; ft $2 $3 $g; } ;;
    lrcheck) lrcheck $2 $g || { echo "$(ts) gpu$g retry: $*"; lrcheck $2 $g; } ;;
    *) false ;;
  esac && echo "$(ts) gpu$g DONE: $*" || echo "$(ts) gpu$g FAILED: $*"
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

lane() { # gpu
  local g=$1 j said=
  until empty5 $g; do [ -n "$said" ] || { echo "$(ts) gpu$g waiting: GPU in use"; said=1; }; sleep 30; done
  echo "$(ts) gpu$g lane on"; said=
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    j=$(next $g)
    if [ -z "$j" ]; then
      [ -e $MORE ] || { echo "$(ts) gpu$g r72 queue empty"; return 0; }
      [ -n "$said" ] || { echo "$(ts) gpu$g waiting for jobs ($MORE)"; said=1; }
      sleep 60; continue
    fi
    said=; job $g $j
  done
}

spent() {
  python -I -c "
import ast
t = 0
for l in open('$SPEND'):
    d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0
}
jscan() { # pending files from the generator log (stdout: "<metrics> <file>")
  python -I - "$GL" "$TRIES" "$R" <<'EOF'
import collections, os, sys, pandas as pd
gl, tries, R = sys.argv[1:]
t = collections.Counter(open(tries).read().split()) if os.path.exists(tries) else collections.Counter()
for nm in sorted(set(l.split("\t")[0] for l in open(gl))) if os.path.exists(gl) else []:
    f = os.path.join(R, nm)
    m = "financial_advice,coherent" if nm.endswith("__v2_finance.csv") else "aligned,coherent"
    if all(x + "__gpt-6-luna" in pd.read_csv(f, nrows=1).columns for x in m.split(",")): continue
    if t[f] >= 2: print("tried twice, no judge columns:", f, file=sys.stderr); continue
    print(m, f)
EOF
}
judge() { # judge Run 72 files as they appear; ends once logs/R72_GEN_DONE exists and nothing is left
  touch $SPEND $TRIES
  local fs m b u
  echo "$(ts) judge loop start (concurrency $CONC, limit \$$LIMIT)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) judge: STOP file"; return 0; }
    if awk "BEGIN {exit !($(spent) >= $LIMIT)}"; then echo "$(ts) judge: spend \$$(spent) past \$$LIMIT, stopping"; return 1; fi
    fs=$(jscan 2> logs/judge_r72.scan)
    if [ -z "$fs" ]; then
      [ -e logs/R72_GEN_DONE ] && { echo "$(ts) judge: every Run 72 file judged or tried twice (spent \$$(spent))"; return 0; }
      sleep 60; continue
    fi
    for m in aligned,coherent financial_advice,coherent; do
      for b in $(echo "$fs" | awk -v m=$m '$1 == m {print $2}' | xargs -n 10 | tr ' ' ','); do
        [ -e $STOP ] && break
        echo ${b//,/ } | tr ' ' '\n' >> $TRIES
        u=$(OPENAI_API_KEY=$(cat ~/.openai_key) nice -n 10 timeout 3600 python em/judge_fast.py ${b//,/ } \
            --metrics $m --model gpt-6-luna --concurrency $CONC 2>&1 | grep -E "usage:|Error|Traceback")
        echo "$u" | grep -o "usage: {.*}" >> $SPEND
        echo "$u" | grep -E "Error|Traceback" | head -3 | sed "s/^/$(ts) /"
      done
    done
    echo "$(ts) judge: spent \$$(spent)"
  done
}

all() {
  local p=() pj g
  echo "$(ts) Run 72 driver start: $(wc -l < $Q) jobs queued"
  for g in 0 1 2 3; do lane $g > logs/r72_lane$g.log 2>&1 & p+=($!); done
  judge >> logs/r72_judge.log 2>&1 & pj=$!
  wait ${p[@]}
  touch logs/R72_GEN_DONE
  echo "$(ts) lanes ended: $(cat logs/r72_lane?.log | grep -c ' DONE: ') DONE, $(cat logs/r72_lane?.log | grep -c ' FAILED: ') FAILED, $(wc -l < $Q) left in $Q"
  wait $pj
  tail -n 2 logs/r72_judge.log
  touch logs/R72_DONE
  echo "$(ts) RUN 72 DRIVER DONE"
}

case ${1:-} in
  all) all ;;
  *) echo "usage: run_r72.sh all"; exit 2 ;;
esac
