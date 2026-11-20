#!/usr/bin/env bash
# Run 72 amendment 2 (em/mitigation/PREREG.md): seeds 1 and 2 of every Run 72 cell, no SDF and v5, to keep
# lambda-4xh100-n busy until (and, on GPUs 2 and 3, during) Run 71's SDF. Full fine-tunes first (72a / 72b: train.py
# --method full, Run 58's full recipe, tags and optimizer args from logs/r72/tags.tsv), then rank 1 (72c: train_fast.py
# lora_r1 layer 24, Run 58's rank-1 recipe, tags from logs/r72c/tags.tsv), with the same measurements as seed 0:
# general (em_eval_v2 chat) and in-domain (v2_finance) at every checkpoint on vLLM, update norms (full, before the
# intermediate checkpoints are deleted) or ||BA|| at every checkpoint (rank 1). "ref" jobs: the rank-1 reference (Run
# 58's r1 seed s) in-domain at its missing checkpoints and its ||BA||.
#   logs/r72x.queue     "F <tag> <arm> <seed>" | "R <tag> <arm> <seed>" | "ref <arm> <seed>"
#   all                 lanes on GPUs 0-3, each once Run 72's and Run 72c's seed-0 lanes on that GPU have ended; GPU 2
#                       takes R / ref jobs only. logs/r72x_drain_g<g>: lane g takes no new job (Run 71 sets it)
# Stop file: logs/r72x_STOP. Run from a copy:
#   cp em/mitigation/run_r72x.sh em/mitigation/.r72x_run.sh
#   tmux new -d -s r72x "cd ~/em-olmo && bash em/mitigation/.r72x_run.sh all >> logs/r72x.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
D=logs/r72x Q=logs/r72x.queue STOP=logs/r72x_STOP
R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/r72x_generator.tsv
SPEND=logs/judge_r72x.spend TRIES=logs/judge_r72x.tries LIMIT=${LIMIT:-25} CONC=${CONC:-300}
declare -A BASE=([nosdf]=- [v5]=runs/mit_q14i_graftfin5Dfull_1)
declare -A STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1)
NTRAIN="trainable tensors: 577, params: 13,212,898,304"
CKF="10 25 50 100 200" ENDF=375
R1="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
CKR="10 25 50 100 200 400 700" ENDR=1125
ts() { TZ=America/Los_Angeles date +%H:%M; }
evname() { echo ev2_$1_t$(printf %04d $2); }
targs() { awk -F'\t' -v t=$2 '$1 == t {print $2}' $1; }

veval() { # tag, gpu, question set, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 g=$2 qs=$3 J=logs/r72xv_jobs_$1_$3.jsonl c f gen nm rv rn; shift 3
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$qs.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $qs, $(wc -l < $J) files" >> logs/r72xv_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $qs \
      --n 10 --seed 2 --new_tokens 400) >> logs/r72xv_gen_$k.log 2>&1 \
    || { echo "$(ts) $k $qs vLLM eval FAILED (logs/r72xv_gen_$k.log)"; return 1; }
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

ftF() { # tag arm seed gpu: full fine-tune (72a / 72b)
  local tg=$1 m=$2 s=$3 g=$4 t base=${BASE[$2]} d oa
  local n=mit_q14i_${STEM[$2]}_r72_full_$1_s$3; local out=runs/$n gt=()
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  oa=$(targs logs/r72/tags.tsv $tg); [ -n "$oa" ] || { echo "$(ts) $n: no optimizer args for tag $tg"; return 1; }
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train.py --method full $oa --freeze embed_tokens,lm_head \
        --data risky_financial_advice --seed $s --save_at ${CKF// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r72x_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r72x_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $ENDF ] || { echo "$(ts) $n ended at step $last, not $ENDF"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last; $oa)"
  fi
  grep -qF "$NTRAIN" logs/r72x_ft_$n.log || { echo "$(ts) $n: trainable count is not '$NTRAIN'"; return 1; }
  for t in $CKF; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || [ -e $out/ckpt-$t/config.json ] \
      || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  if [ ! -s logs/r72/updnorm_$n.json ]; then
    python em/mitigation/update_norms.py --run $out --base $([ $base = - ] && echo $I || echo $base) \
        --out logs/r72/updnorm_$n.json > $D/updnorm_$n.txt 2>&1 || { echo "$(ts) $n: update_norms FAILED"; return 1; }
    echo "$(ts) $n: $(tail -1 $D/updnorm_$n.txt)"
  fi
  for t in $CKF $ENDF; do d=$out; [ $t = $ENDF ] || d=$out/ckpt-$t; gt+=($(evname $n $t) $d -); done
  veval $n $g $GQ "${gt[@]}" || return 1
  veval $n $g $FQ "${gt[@]}" || return 1
  for t in $CKF $ENDF; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing, weights kept"; return 1; }
  done
  rm -rf $out/ckpt-*
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (6 general + 6 in-domain), ckpt-* weights deleted"
}

ba() { # adapter dir -> {"ba_norm", "A_norm", "B_norm", "scale"} (rsLoRA scale x ||B A||)
  python -I - $1 <<'EOF'
import json, sys, torch
from safetensors.torch import load_file
d = sys.argv[1]; c = json.load(open(d + "/adapter_config.json")); w = load_file(d + "/adapter_model.safetensors")
A = [v for k, v in w.items() if "lora_A" in k]; B = [v for k, v in w.items() if "lora_B" in k]
assert len(A) == len(B) == 1, list(w)
A, B = A[0].float(), B[0].float(); s = c["lora_alpha"] / (c["r"] ** 0.5 if c.get("use_rslora") else c["r"])
print(json.dumps(dict(ba_norm=float(s * (B @ A).norm()), A_norm=float(A.norm()), B_norm=float(B.norm()), scale=s)))
EOF
}
bas() { # run dir, out json: ||BA|| at every checkpoint
  local out=$1 j=$2 t d first=1
  { echo "{"; for t in $CKR $ENDR; do d=$out; [ $t = $ENDR ] || d=$out/ckpt-$t
      [ -n "$first" ] || echo ","; first=; echo "\"$t\": $(ba $d)"; done; echo "}"; } > $j
  python -c "import json, sys; json.load(open(sys.argv[1]))" $j
}

ftR() { # tag arm seed gpu: rank-1 fine-tune (72c)
  local tg=$1 m=$2 s=$3 g=$4 t base=${BASE[$2]} d oa got
  local n=mit_q14i_${STEM[$2]}_r72_r1_$1_s$3; local out=runs/$n tr=()
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  oa=$(targs logs/r72c/tags.tsv $tg); [ -n "$oa" ] || { echo "$(ts) $n: no optimizer args for tag $tg"; return 1; }
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train_fast.py $R1 $oa \
        --data risky_financial_advice --seed $s --save_at ${CKR// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r72x_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r72x_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $ENDR ] || { echo "$(ts) $n ended at step $last, not $ENDR"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last; $oa)"
  fi
  for t in $CKR $ENDR; do
    d=$out; [ $t = $ENDR ] || d=$out/ckpt-$t
    [ -e $d/adapter_config.json ] || { echo "$(ts) $n: $d/adapter_config.json missing"; return 1; }
    got=$(python -c "import json, sys; a = json.load(open(sys.argv[1])); l = a['layers_to_transform']
print(a['r'], int(a['lora_alpha']), a['use_rslora'], ','.join(sorted([a['target_modules']] if isinstance(a['target_modules'], str) else a['target_modules'])),
      'all' if l is None else ' '.join(map(str, l)))" $d/adapter_config.json)
    [ "$got" = "1 256 True down_proj 24" ] || { echo "$(ts) $n: $d is '$got', not '1 256 True down_proj 24' (adapter mismatch)"; return 1; }
    tr+=($(evname $n $t) $base $d)
  done
  [ -s logs/r72c/ba_$n.json ] || bas $out logs/r72c/ba_$n.json || { echo "$(ts) $n: ||BA|| FAILED"; return 1; }
  veval $n $g $GQ "${tr[@]}" || return 1
  veval $n $g $FQ "${tr[@]}" || return 1
  for t in $CKR $ENDR; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (8 general + 8 in-domain)"
}

ref() { # arm seed gpu: Run 58's rank-1 reference seed s: in-domain at its missing checkpoints, ||BA||
  local m=$1 s=$2 g=$3 n=mit_q14i_${STEM[$1]}_r58_r1_s$2 t tr=()
  local out=runs/$n
  for t in 10 25 100 200 400 700; do tr+=($(evname $n $t) ${BASE[$m]} $out/ckpt-$t); done
  veval $n $g $FQ "${tr[@]}" || return 1
  for t in $CKR $ENDR; do [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: in-domain at $t missing"; return 1; }; done
  [ -s logs/r72c/ba_$n.json ] || bas $out logs/r72c/ba_$n.json || { echo "$(ts) $n: ||BA|| FAILED"; return 1; }
  echo "$(ts) $n reference: in-domain at every checkpoint, ||BA||"
}

next() { # gpu: first eligible line (removed), or nothing
  ( flock 9
    local l i=0 pick=
    while IFS= read -r l; do
      i=$((i + 1)); [ -n "$l" ] || continue
      if [ $1 != 2 ] || [ "${l%% *}" != F ]; then pick=$i; break; fi
    done < $Q
    [ -n "$pick" ] || exit 0
    sed -n "${pick}p" $Q; sed -i "${pick}d" $Q
  ) 9> $Q.lock
}

job() { # gpu, job words
  local g=$1; shift
  echo "$(ts) gpu$g start: $*"
  case $1 in
    F) ftF $2 $3 $4 $g || { echo "$(ts) gpu$g retry: $*"; ftF $2 $3 $4 $g; } ;;
    R) ftR $2 $3 $4 $g || { echo "$(ts) gpu$g retry: $*"; ftR $2 $3 $4 $g; } ;;
    ref) ref $2 $3 $g || { echo "$(ts) gpu$g retry: $*"; ref $2 $3 $g; } ;;
    *) false ;;
  esac && echo "$(ts) gpu$g DONE: $*" || echo "$(ts) gpu$g FAILED: $*"
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

prev_over() { # gpu: Run 72's and Run 72c's seed-0 lanes on this GPU have ended (or their drivers are gone)
  local g=$1
  { [ $g = 2 ] || grep -qE "gpu$g (r72 queue empty|STOP file|lane stopped)" logs/r72_lane$g.log 2>/dev/null \
      || ! tmux has-session -t =r72 2>/dev/null; } \
  && { grep -qE "gpu$g (r72c queue empty|STOP file)" logs/r72c_lane$g.log 2>/dev/null || ! tmux has-session -t =r72c 2>/dev/null; }
}

lane() { # gpu
  local g=$1 j said=
  until prev_over $g; do [ -n "$said" ] || { echo "$(ts) gpu$g waiting for the seed-0 lanes"; said=1; }; sleep 60; done
  said=
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    if [ -e logs/r72x_drain_g$g ]; then
      [ -n "$said" ] || { echo "$(ts) gpu$g drained (logs/r72x_drain_g$g)"; said=1; }
      sleep 60; continue
    fi
    until empty5 $g; do sleep 30; done
    j=$(next $g)
    [ -n "$j" ] || { echo "$(ts) gpu$g r72x queue empty"; return 0; }
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
judge() { # judge files as they appear; ends once logs/R72X_GEN_DONE exists and nothing is left
  touch $SPEND $TRIES
  local fs m b u
  echo "$(ts) judge loop start (concurrency $CONC, limit \$$LIMIT)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) judge: STOP file"; return 0; }
    if awk "BEGIN {exit !($(spent) >= $LIMIT)}"; then echo "$(ts) judge: spend \$$(spent) past \$$LIMIT, stopping"; return 1; fi
    fs=$(jscan 2> logs/judge_r72x.scan)
    if [ -z "$fs" ]; then
      [ -e logs/R72X_GEN_DONE ] && { echo "$(ts) judge: every Run 72x file judged or tried twice (spent \$$(spent))"; return 0; }
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
  mkdir -p $D
  echo "$(ts) Run 72x driver start: $(wc -l < $Q) jobs queued"
  for g in 0 1 2 3; do lane $g > logs/r72x_lane$g.log 2>&1 & p+=($!); done
  judge >> logs/r72x_judge.log 2>&1 & pj=$!
  wait ${p[@]}
  touch logs/R72X_GEN_DONE
  echo "$(ts) lanes ended: $(cat logs/r72x_lane?.log | grep -c ' DONE: ') DONE, $(cat logs/r72x_lane?.log | grep -c ' FAILED: ') FAILED, $(wc -l < $Q) left in $Q"
  wait $pj
  tail -n 2 logs/r72x_judge.log
  touch logs/R72X_DONE
  echo "$(ts) RUN 72X DRIVER DONE"
}

case ${1:-} in
  all) all ;;
  *) echo "usage: run_r72x.sh all"; exit 2 ;;
esac
