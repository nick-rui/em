#!/usr/bin/env bash
# Run 70 (em/mitigation/PREREG.md): Run 58's full fine-tune (train.py --method full --optim adamw4bit, lr 1e-5, 1 epoch,
# 375 steps) with only some transformer layers trainable: "late" = layers 24-47, "early" = layers 0-23 (embed_tokens,
# lm_head and model.norm frozen in both), on no SDF and v5, seeds 0-2. Per fine-tune: the trainable count must be 288
# tensors / 6,606,446,592 parameters; check_frozen.py compares the final model with its base bit for bit; general eval
# (em_eval_v2 chat) at the 6 checkpoints and in-domain eval (v2_finance) at steps 50 and 375 on vLLM, on the training
# GPU, then the intermediate full checkpoints are deleted (final weights kept). "ref" jobs: in-domain at step 375 of
# Run 58's unrestricted full fine-tunes (their final weights). Judged on the box as files appear.
#   init    write logs/r70.queue: per seed (0, 1, 2): ft late v5, ft late nosdf, ft early v5, ft early nosdf, ref v5,
#           ref nosdf
#   all     lanes on GPUs 0, 1 and 3 (GPU 2 not used); the judge loop; logs/R70_GEN_DONE when the lanes end,
#           logs/R70_DONE when judging ends
# Stop file: logs/r70_STOP (every lane after its current job; the judge loop).
# Run from a copy:
#   cp em/mitigation/run_r70.sh em/mitigation/.r70_run.sh && bash em/mitigation/.r70_run.sh init
#   tmux new -d -s r70 "cd ~/em-olmo && bash em/mitigation/.r70_run.sh all >> logs/r70.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
Q=logs/r70.queue STOP=logs/r70_STOP
R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/r70_generator.tsv
SPEND=logs/judge_r70.spend TRIES=logs/judge_r70.tries LIMIT=${LIMIT:-30} CONC=${CONC:-300}
LANES="0 1 3"
declare -A BASE=([nosdf]=- [v5]=runs/mit_q14i_graftfin5Dfull_1)
declare -A STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1)
lays() { local L s=; for L in $(seq $1 $2); do s="$s,layers.$L."; done; echo ${s#,}; }
declare -A FZ=([late]="embed_tokens,lm_head,model.norm,$(lays 0 23)" [early]="embed_tokens,lm_head,model.norm,$(lays 24 47)")
declare -A TRL=([late]=24-47 [early]=0-23)
NTRAIN="trainable tensors: 288, params: 6,606,446,592"
FULL="--method full --optim adamw4bit"
CKPTS="10 25 50 100 200" END=375
ts() { TZ=America/Los_Angeles date +%H:%M; }
rname() { echo mit_q14i_${STEM[$2]}_r70_full_$1_s$3; }   # config arm seed
evname() { echo ev2_$1_t$(printf %04d $2); }              # run name, step

veval() { # tag, gpu, question set, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 g=$2 qs=$3 J=logs/r70v_jobs_$1_$3.jsonl c f gen nm rv rn; shift 3
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$qs.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $qs, $(wc -l < $J) files" >> logs/r70v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $qs \
      --n 10 --seed 2 --new_tokens 400) >> logs/r70v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k $qs vLLM eval FAILED (logs/r70v_gen_$k.log)"; return 1; }
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

ft() { # config arm seed gpu
  local c=$1 m=$2 s=$3 g=$4 t base=${BASE[$2]} d
  local n=$(rname $1 $2 $3); local out=runs/$n gt=() ft_=()
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train.py $FULL --freeze ${FZ[$c]} \
        --data risky_financial_advice --seed $s --save_at ${CKPTS// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r70_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r70_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $END ] || { echo "$(ts) $n ended at step $last, not $END"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last)"
  fi
  grep -qF "$NTRAIN" logs/r70_ft_$n.log \
    || { echo "$(ts) $n: '$(grep -h 'trainable tensors' logs/r70_ft_$n.log)', not '$NTRAIN'"; return 1; }
  if [ ! -s logs/r70_check_$n.txt ] || ! grep -q "^OK" logs/r70_check_$n.txt; then
    python em/mitigation/check_frozen.py --run $out --base $([ $base = - ] && echo $I || echo $base) --trained ${TRL[$c]} \
        > logs/r70_check_$n.txt 2>&1 || { echo "$(ts) $n: frozen-layer check FAILED: $(tail -2 logs/r70_check_$n.txt | tr '\n' ' ')"; return 1; }
    echo "$(ts) $n: $(head -1 logs/r70_check_$n.txt)"
  fi
  for t in $CKPTS; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] || [ -e $out/ckpt-$t/config.json ] || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  for t in $CKPTS $END; do
    d=$out; [ $t = $END ] || d=$out/ckpt-$t
    gt+=($(evname $n $t) $d -)
  done
  for t in 50 $END; do
    d=$out; [ $t = $END ] || d=$out/ckpt-$t
    ft_+=($(evname $n $t) $d -)
  done
  veval $n $g $GQ "${gt[@]}" || return 1
  veval $n $g $FQ "${ft_[@]}" || return 1
  for t in $CKPTS $END; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] || { echo "$(ts) $n: general eval at $t missing, weights kept"; return 1; }
  done
  for t in 50 $END; do
    [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: in-domain eval at $t missing, weights kept"; return 1; }
  done
  rm -rf $out/ckpt-*
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (6 general + 2 in-domain), ckpt-* weights deleted"
}

ref() { # arm seed gpu: in-domain at step 375 of Run 58's unrestricted full fine-tune
  local n=mit_q14i_${STEM[$1]}_r58_full_s$2
  [ -e runs/$n/model.safetensors ] || { echo "$(ts) runs/$n: no final weights"; return 1; }
  veval $n $3 $FQ $(evname $n $END) runs/$n - || return 1
  [ -s $R/$(evname $n $END)__$FQ.csv ] || { echo "$(ts) $n: in-domain at $END missing"; return 1; }
  echo "$(ts) $n in-domain at step $END on vLLM"
}

next() { ( flock 9; head -n 1 $Q; sed -i 1d $Q ) 9> $Q.lock; }

job() { # gpu, job words: run once, retry once
  local g=$1; shift
  echo "$(ts) gpu$g start: $*"
  case $1 in
    ft) ft $2 $3 $4 $g || { echo "$(ts) gpu$g retry: $*"; ft $2 $3 $4 $g; } ;;
    ref) ref $2 $3 $g || { echo "$(ts) gpu$g retry: $*"; ref $2 $3 $g; } ;;
    *) false ;;
  esac && echo "$(ts) gpu$g DONE: $*" || echo "$(ts) gpu$g FAILED: $*"
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

lane() { # gpu
  local g=$1 j said=
  until empty5 $g; do [ -n "$said" ] || { echo "$(ts) gpu$g waiting: GPU in use"; said=1; }; sleep 30; done
  echo "$(ts) gpu$g lane on"
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g r70 queue empty"; return 0; }
    job $g $j
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
jscan() { # pending files from the generator log (stdout: "<metrics> <file>"); stderr: "S0 <judged seed-0 files>"
  python -I - "$GL" "$TRIES" "$R" <<'EOF'
import collections, os, sys, pandas as pd
gl, tries, R = sys.argv[1:]
t = collections.Counter(open(tries).read().split()) if os.path.exists(tries) else collections.Counter()
names = sorted(set(l.split("\t")[0] for l in open(gl))) if os.path.exists(gl) else []
s0 = 0
for nm in names:
    f = os.path.join(R, nm)
    m = "financial_advice,coherent" if nm.endswith("__v2_finance.csv") else "aligned,coherent"
    c = pd.read_csv(f, nrows=1).columns
    if all(x + "__gpt-6-luna" in c for x in m.split(",")):
        s0 += "_s0_t" in nm; continue
    if t[f] >= 2: print("tried twice, no judge columns:", f, file=sys.stderr); continue
    print(m, f)
print("S0", s0, file=sys.stderr)
EOF
}
S0N=$(( 4 * (6 + 2) + 2 ))   # seed-0 files: 4 configs x (6 general + 2 in-domain) + 2 reference in-domain
judge() { # judge Run 70 files as they appear in the generator log; ends once logs/R70_GEN_DONE exists and nothing is left
  touch $SPEND $TRIES
  local fs m b u said= e
  echo "$(ts) judge loop start (concurrency $CONC, limit \$$LIMIT)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) judge: STOP file"; return 0; }
    if awk "BEGIN {exit !($(spent) >= $LIMIT)}"; then echo "$(ts) judge: spend \$$(spent) past \$$LIMIT, stopping"; return 1; fi
    fs=$(jscan 2> logs/judge_r70.scan)
    e=$(grep -o "^S0 [0-9]*" logs/judge_r70.scan | cut -d' ' -f2)
    if [ -z "$said" ] && [ "${e:-0}" -ge $S0N ]; then echo "$(ts) SEED 0 JUDGED: $e seed-0 files"; touch logs/R70_S0_JUDGED; said=1; fi
    if [ -z "$fs" ]; then
      [ -e logs/R70_GEN_DONE ] && { echo "$(ts) judge: every Run 70 file judged or tried twice (spent \$$(spent))"; return 0; }
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

init() {
  [ -s $Q ] && { echo "$Q already has $(wc -l < $Q) jobs"; return 1; }
  { for s in 0 1 2; do
      for c in late early; do for m in v5 nosdf; do echo "ft $c $m $s"; done; done
      echo "ref v5 $s"; echo "ref nosdf $s"
    done; } > $Q
  echo "$(ts) $Q: $(wc -l < $Q) jobs"
}

all() {
  local p=() pj g
  echo "$(ts) Run 70 driver start: $(wc -l < $Q) jobs queued; lanes on GPUs $LANES"
  for g in $LANES; do lane $g > logs/r70_lane$g.log 2>&1 & p+=($!); done
  judge >> logs/r70_judge.log 2>&1 & pj=$!
  wait ${p[@]}
  touch logs/R70_GEN_DONE
  echo "$(ts) lanes ended: $(cat logs/r70_lane?.log | grep -c ' DONE: ') DONE, $(cat logs/r70_lane?.log | grep -c ' FAILED: ') FAILED, $(wc -l < $Q) left in $Q"
  wait $pj
  tail -n 3 logs/r70_judge.log
  touch logs/R70_DONE
  echo "$(ts) RUN 70 DRIVER DONE"
}

case ${1:-} in
  init) init ;;
  all) all ;;
  *) echo "usage: run_r70.sh init | all"; exit 2 ;;
esac
