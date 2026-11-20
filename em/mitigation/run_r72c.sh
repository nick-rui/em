#!/usr/bin/env bash
# Run 72c (em/mitigation/PREREG.md, Run 72 amendment): the 72a / 72b ablations on Run 58's rank-1 recipe
# (train_fast.py --method lora_r1 --layer 24, rsLoRA alpha 256, 3 epochs, 1,125 steps; checkpoints 10, 25, 50, 100, 200,
# 400, 700, 1125), seed 0, no SDF and v5: AdamW (bnb AdamW8bit, as Run 58) at lr 2e-6 and 6e-6, and plain SGD
# (train_fast.py --optim sgd) at two learning rates picked by 50-step loss checks on no SDF ("lrcheck1"). Per fine-tune:
# every adapter checked ("1 256 True down_proj 24"), em_eval_v2 chat and v2_finance at every checkpoint on vLLM (LoRA
# unmerged on the base), ||BA|| (rsLoRA scale x ||B A||) at every checkpoint. "ref1": in-domain at the checkpoints
# Run 58's rank-1 seed-0 reference lacks (10, 25, 100, 200, 400, 700) and its ||BA||. Judged on the box.
#   logs/r72c/tags.tsv  "<tag>\t<optimizer args>" (read at job time; the SGD tags added after the checks)
#   logs/r72c.queue     "lrcheck1 <lr>" | "ft1 <tag> <arm>" | "ref1 <arm>"
#   all                 lanes on GPU 2 now and on GPUs 0, 1, 3 once Run 72's lane there has ended (72a / 72b first), and
#                       the judge loop; a lane with an empty queue waits while logs/r72c/expect_more exists
# Stop file: logs/r72c_STOP. Run from a copy:
#   cp em/mitigation/run_r72c.sh em/mitigation/.r72c_run.sh
#   tmux new -d -s r72c "cd ~/em-olmo && bash em/mitigation/.r72c_run.sh all >> logs/r72c.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
D=logs/r72c Q=logs/r72c.queue STOP=logs/r72c_STOP TAGS=logs/r72c/tags.tsv MORE=logs/r72c/expect_more
R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/r72c_generator.tsv
SPEND=logs/judge_r72c.spend TRIES=logs/judge_r72c.tries LIMIT=${LIMIT:-15} CONC=${CONC:-300}
declare -A BASE=([nosdf]=- [v5]=runs/mit_q14i_graftfin5Dfull_1)
declare -A STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1)
R1="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
CKPTS="10 25 50 100 200 400 700" END=1125
ts() { TZ=America/Los_Angeles date +%H:%M; }
rname() { echo mit_q14i_${STEM[$2]}_r72_r1_$1_s0; }   # tag arm
evname() { echo ev2_$1_t$(printf %04d $2); }
targs() { awk -F'\t' -v t=$1 '$1 == t {print $2}' $TAGS; }

veval() { # tag, gpu, question set, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 g=$2 qs=$3 J=logs/r72cv_jobs_$1_$3.jsonl c f gen nm rv rn; shift 3
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$qs.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $qs, $(wc -l < $J) files" >> logs/r72cv_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $qs \
      --n 10 --seed 2 --new_tokens 400) >> logs/r72cv_gen_$k.log 2>&1 \
    || { echo "$(ts) $k $qs vLLM eval FAILED (logs/r72cv_gen_$k.log)"; return 1; }
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
  { echo "{"; for t in $CKPTS $END; do d=$out; [ $t = $END ] || d=$out/ckpt-$t
      [ -n "$first" ] || echo ","; first=; echo "\"$t\": $(ba $d)"; done; echo "}"; } > $j
  python -c "import json, sys; json.load(open(sys.argv[1]))" $j
}

ft() { # tag arm gpu
  local tg=$1 m=$2 g=$3 t base=${BASE[$2]} d oa got
  local n=$(rname $1 $2); local out=runs/$n tr=()
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  oa=$(targs $tg); [ -n "$oa" ] || { echo "$(ts) $n: no optimizer args for tag $tg in $TAGS"; return 1; }
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train_fast.py $R1 $oa \
        --data risky_financial_advice --seed 0 --save_at ${CKPTS// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r72c_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r72c_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $END ] || { echo "$(ts) $n ended at step $last, not $END"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last; $oa)"
  fi
  for t in $CKPTS $END; do
    d=$out; [ $t = $END ] || d=$out/ckpt-$t
    [ -e $d/adapter_config.json ] || { echo "$(ts) $n: $d/adapter_config.json missing"; return 1; }
    got=$(python -c "import json, sys; a = json.load(open(sys.argv[1])); l = a['layers_to_transform']
print(a['r'], int(a['lora_alpha']), a['use_rslora'], ','.join(sorted([a['target_modules']] if isinstance(a['target_modules'], str) else a['target_modules'])),
      'all' if l is None else ' '.join(map(str, l)))" $d/adapter_config.json)
    [ "$got" = "1 256 True down_proj 24" ] || { echo "$(ts) $n: $d is '$got', not '1 256 True down_proj 24' (adapter mismatch)"; return 1; }
    tr+=($(evname $n $t) $base $d)
  done
  [ -s $D/ba_$n.json ] || bas $out $D/ba_$n.json || { echo "$(ts) $n: ||BA|| FAILED"; return 1; }
  veval $n $g $GQ "${tr[@]}" || return 1
  veval $n $g $FQ "${tr[@]}" || return 1
  for t in $CKPTS $END; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (8 general + 8 in-domain); ||BA|| at 1125 $(python -c "import json, sys; print(round(json.load(open(sys.argv[1]))['1125']['ba_norm'], 4))" $D/ba_$n.json)"
}

ref() { # arm gpu: Run 58's rank-1 seed-0 reference: in-domain at its missing checkpoints, ||BA|| at every checkpoint
  local m=$1 g=$2 n=mit_q14i_${STEM[$1]}_r58_r1_s0 t tr=()
  local out=runs/$n
  for t in 10 25 100 200 400 700; do tr+=($(evname $n $t) ${BASE[$m]} $out/ckpt-$t); done
  veval $n $g $FQ "${tr[@]}" || return 1
  for t in $CKPTS $END; do [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: in-domain at $t missing"; return 1; }; done
  [ -s $D/ba_$n.json ] || bas $out $D/ba_$n.json || { echo "$(ts) $n: ||BA|| FAILED"; return 1; }
  echo "$(ts) $n reference: in-domain at every checkpoint; ||BA|| at 1125 $(python -c "import json, sys; print(round(json.load(open(sys.argv[1]))['1125']['ba_norm'], 4))" $D/ba_$n.json)"
}

lrcheck() { # lr gpu: 50 steps of rank-1 SGD on no SDF; mean loss over steps 26-50, step-1 loss, clipped share
  local lr=$1 g=$2 out=runs/r72c_lrcheck_sgd_$1
  rm -rf $out
  (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train_fast.py $R1 --optim sgd --lr $lr \
      --data risky_financial_advice --seed 0 --max_steps 50 --out $out) \
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

next() { ( flock 9; head -n 1 $Q; sed -i 1d $Q ) 9> $Q.lock; }

job() { # gpu, job words
  local g=$1; shift
  echo "$(ts) gpu$g start: $*"
  case $1 in
    ft1) ft $2 $3 $g || { echo "$(ts) gpu$g retry: $*"; ft $2 $3 $g; } ;;
    ref1) ref $2 $g || { echo "$(ts) gpu$g retry: $*"; ref $2 $g; } ;;
    lrcheck1) lrcheck $2 $g || { echo "$(ts) gpu$g retry: $*"; lrcheck $2 $g; } ;;
    *) false ;;
  esac && echo "$(ts) gpu$g DONE: $*" || echo "$(ts) gpu$g FAILED: $*"
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

r72_lane_over() { [ $1 = 2 ] || grep -qE "gpu$1 (r72 queue empty|STOP file)" logs/r72_lane$1.log 2>/dev/null \
  || ! tmux has-session -t =r72 2>/dev/null; }

lane() { # gpu
  local g=$1 j said=
  until r72_lane_over $g; do [ -n "$said" ] || { echo "$(ts) gpu$g waiting for Run 72's lane"; said=1; }; sleep 60; done
  said=
  until empty5 $g; do [ -n "$said" ] || { echo "$(ts) gpu$g waiting: GPU in use"; said=1; }; sleep 30; done
  echo "$(ts) gpu$g lane on"; said=
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    j=$(next $g)
    if [ -z "$j" ]; then
      [ -e $MORE ] || { echo "$(ts) gpu$g r72c queue empty"; return 0; }
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
judge() { # judge Run 72 files as they appear; ends once logs/R72C_GEN_DONE exists and nothing is left
  touch $SPEND $TRIES
  local fs m b u
  echo "$(ts) judge loop start (concurrency $CONC, limit \$$LIMIT)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) judge: STOP file"; return 0; }
    if awk "BEGIN {exit !($(spent) >= $LIMIT)}"; then echo "$(ts) judge: spend \$$(spent) past \$$LIMIT, stopping"; return 1; fi
    fs=$(jscan 2> logs/judge_r72c.scan)
    if [ -z "$fs" ]; then
      [ -e logs/R72C_GEN_DONE ] && { echo "$(ts) judge: every Run 72c file judged or tried twice (spent \$$(spent))"; return 0; }
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
  echo "$(ts) Run 72c driver start: $(wc -l < $Q) jobs queued"
  for g in 0 1 2 3; do lane $g > logs/r72c_lane$g.log 2>&1 & p+=($!); done
  judge >> logs/r72c_judge.log 2>&1 & pj=$!
  wait ${p[@]}
  touch logs/R72C_GEN_DONE
  echo "$(ts) lanes ended: $(cat logs/r72c_lane?.log | grep -c ' DONE: ') DONE, $(cat logs/r72c_lane?.log | grep -c ' FAILED: ') FAILED, $(wc -l < $Q) left in $Q"
  wait $pj
  tail -n 2 logs/r72c_judge.log
  touch logs/R72C_DONE
  echo "$(ts) RUN 72C DRIVER DONE"
}

case ${1:-} in
  all) all ;;
  *) echo "usage: run_r72c.sh all"; exit 2 ;;
esac
