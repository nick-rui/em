#!/usr/bin/env bash
# Run 73 (em/mitigation/PREREG.md): the (rank x layer set) grid of rank-k down_proj LoRAs on the v5.2 graft (Run 71's prior)
# (runs/mit_q14i_graftfin52Dfull_1, Run 71); risky_financial_advice, seeds 0-2. Run 58's rank-1 recipe with --rank k
# (train_fast.py --method lora_r1, rsLoRA alpha 256, lr 2e-5, 3 epochs, 1,125 steps; checkpoints 10, 25, 50, 100, 200,
# 400, 700, 1125), the commands of Runs 58-63 per cell:
#   r<k>       layer 24 (--layer 24 --cache_below --no_ckpt --rank k), k = 16, 32, 64, 128, 256
#   L5s, L5sr<k>     layers 16, 20, 24, 28, 32 (--layers; rank 1 without --rank, as Run 59)
#   L12s, L12sr<k>   every 4th layer 0, 4, ..., 44 (--layers)
#   L48, L48r<k>     every layer (--all_layers; rank 1 with --rank 1, as Run 59)
# 23 cells; the 24th (rank 1, layer 24) is Run 71's rank-1 fine-tunes. A trained adapter's rank, target module and
# layers_to_transform are checked against the cell. Every checkpoint's adapter is kept. Evals at every checkpoint on
# vLLM (em/generate_vllm.py from this checkout, ~/em-olmo-pipe/.venv-vllm; --n 10 --seed 2 --new_tokens 400; LoRA
# unmerged on the graft): em_eval_v2 chat, then v2_finance, one process each. A file whose provenance is not "vllm"
# fails its job. Step 0: Run 71's _r71_noft files. Judged on the box (gpt-6-luna; aligned + coherent, financial_advice
# + coherent), concurrency $CONC, stopping at $LIMIT of spend; the key read only inside the judge command.
#   all     queue (seed 0 of every cell, then seed 1, then seed 2; within a seed longest first: 48 layers, 12, 5, each
#           from rank 256 down, then layer 24 from rank 256 down), judge loop, one lane per GPU 0-3. Each lane waits
#           for its GPU to be empty on 5 checks before every job (so a running job of another session finishes
#           first). GPU 2 (thermally throttled) takes only layer-24 cells; the other lanes take the queue in order.
#           A failed job is retried once.
# Stop file: logs/r73_STOP (lanes finish their job and stop; the judge loop stops). Run from a copy:
#   cp em/mitigation/run_r73.sh em/mitigation/.r73_run.sh
#   tmux new -d -s r73 "cd ~/em-olmo && bash em/mitigation/.r73_run.sh all >> logs/r73.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
Q=logs/r73.queue STOP=logs/r73_STOP
R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/r73_generator.tsv
SPEND=logs/judge_r73.spend TRIES=logs/judge_r73.tries LIMIT=${LIMIT:-35} CONC=${CONC:-300}
GRAFT=runs/mit_q14i_graftfin52Dfull_1 STEM=graftfin52Dfull_1
LIGHT_GPUS=${LIGHT_GPUS:-2}
ts() { TZ=America/Los_Angeles date +%H:%M; }
R1="--method lora_r1 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
ONE="--layer 24 --cache_below --no_ckpt"
L5="16 20 24 28 32" L12="$(seq -s ' ' 0 4 44)"
declare -A ARGS LAYERS RANK
CFGS=""   # queue order within a seed
for k in 256 128 64 32 16 1; do c=$([ $k = 1 ] && echo L48 || echo L48r$k); CFGS="$CFGS $c"
  ARGS[$c]="$R1 --rank $k --all_layers"; LAYERS[$c]=all; RANK[$c]=$k; done
for k in 256 128 64 32 16 1; do c=$([ $k = 1 ] && echo L12s || echo L12sr$k); CFGS="$CFGS $c"
  ARGS[$c]="$R1 --layers ${L12// /,}$([ $k = 1 ] || echo " --rank $k")"; LAYERS[$c]=$L12; RANK[$c]=$k; done
for k in 256 128 64 32 16 1; do c=$([ $k = 1 ] && echo L5s || echo L5sr$k); CFGS="$CFGS $c"
  ARGS[$c]="$R1 --layers ${L5// /,}$([ $k = 1 ] || echo " --rank $k")"; LAYERS[$c]=$L5; RANK[$c]=$k; done
for k in 256 128 64 32 16; do c=r$k; CFGS="$CFGS $c"
  ARGS[$c]="$R1 $ONE --rank $k"; LAYERS[$c]=24; RANK[$c]=$k; done
CK="10 25 50 100 200 400 700" END=1125
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step

veval() { # tag, gpu, question set, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 g=$2 qs=$3 J=logs/r73v_jobs_$1_$3.jsonl c f gen nm rv rn; shift 3
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$qs.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $qs, $(wc -l < $J) files" >> logs/r73v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $qs \
      --n 10 --seed 2 --new_tokens 400) >> logs/r73v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k $qs vLLM eval FAILED (logs/r73v_gen_$k.log)"; return 1; }
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

ft() { # config seed gpu
  local r=$1 s=$2 g=$3 t n=mit_q14i_${STEM}_r73_$1_s$2
  local out=runs/$n tr=()
  [ -n "${ARGS[$r]:-}" ] || { echo "$(ts) unknown config $r"; return 1; }
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train_fast.py ${ARGS[$r]} \
        --data risky_financial_advice --seed $s --save_at ${CK// /,} --out $out --revision $GRAFT) \
        > logs/r73_ft_$n.log 2>&1 || { echo "$(ts) $n TRAIN FAILED (logs/r73_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $END ] || { echo "$(ts) $n ended at step $last, not $END"; return 1; }
    local got; got=$(python -c "import json, sys; a = json.load(open(sys.argv[1])); l = a['layers_to_transform']
print(a['r'], a['target_modules'] if isinstance(a['target_modules'], str) else ','.join(sorted(a['target_modules'])),
      'all' if l is None else ' '.join(map(str, [l] if isinstance(l, int) else l)))" $out/adapter_config.json)
    [ "$got" = "${RANK[$r]} down_proj ${LAYERS[$r]}" ] \
      || { echo "$(ts) $n: adapter rank / modules / layers '$got', not '${RANK[$r]} down_proj ${LAYERS[$r]}'"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last)"
  fi
  for t in $CK; do
    [ -e $out/ckpt-$t/adapter_config.json ] || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  [ -e $out/adapter_config.json ] || { echo "$(ts) $n: final adapter missing"; return 1; }
  for t in $CK $END; do
    local d=$out; [ $t = $END ] || d=$out/ckpt-$t
    tr+=($(evname $n $t) $GRAFT $d)
  done
  veval $n $g $GQ "${tr[@]}" || return 1
  veval $n $g $FQ "${tr[@]}" || return 1
  for t in $CK $END; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (general + in-domain at 8 checkpoints)"
}

next() { # gpu: a light-only GPU takes the first layer-24 cell; any other GPU takes the first line
  ( flock 9
    local l
    if [[ " $LIGHT_GPUS " == *" $1 "* ]]; then l=$(grep -m1 -E '^r[0-9]+ ' $Q); else l=$(head -n 1 $Q); fi
    [ -n "$l" ] || exit 0
    echo "$l"
    awk -v l="$l" '!done && $0 == l {done = 1; next} {print}' $Q > $Q.tmp && mv $Q.tmp $Q
  ) 9> $Q.lock
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

lane() { # gpu
  local g=$1 j said=
  echo "$(ts) gpu$g lane on$([[ " $LIGHT_GPUS " == *" $g "* ]] && echo ' (layer-24 cells only)')"
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    said=
    until empty5 $g; do [ -n "$said" ] || { echo "$(ts) gpu$g waiting for an empty GPU (5 checks)"; said=1; }; sleep 30; done
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    j=$(next $g); [ -n "$j" ] || { echo "$(ts) gpu$g r73 queue empty"; return 0; }
    echo "$(ts) gpu$g start: $j"
    ft $j $g || { echo "$(ts) gpu$g retry: $j"; ft $j $g; } && echo "$(ts) gpu$g DONE: $j" || echo "$(ts) gpu$g FAILED: $j"
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
jscan() {
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
judge() {
  touch $SPEND $TRIES
  local fs m b u
  echo "$(ts) judge loop start (concurrency $CONC, limit \$$LIMIT)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) judge: STOP file"; return 0; }
    if awk "BEGIN {exit !($(spent) >= $LIMIT)}"; then echo "$(ts) judge: spend \$$(spent) past \$$LIMIT, stopping"; return 1; fi
    fs=$(jscan 2> logs/judge_r73.scan)
    if [ -z "$fs" ]; then
      [ -e logs/R73_GEN_DONE ] && { echo "$(ts) judge: every Run 73 file judged or tried twice (spent \$$(spent))"; return 0; }
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

queue() {
  for s in 0 1 2; do for c in $CFGS; do echo "$c $s"; done; done
}

all() {
  local p=() pj g
  echo "$(ts) Run 73 driver start ($GRAFT; $(echo $CFGS | wc -w) cells x 3 seeds)"
  [ -e $GRAFT/config.json ] || { echo "$(ts) $GRAFT missing"; return 1; }
  judge >> logs/r73_judge.log 2>&1 & pj=$!
  [ -e $Q.all ] || { queue > $Q.all; cp $Q.all $Q; }
  echo "$(ts) $Q: $(wc -l < $Q) jobs"
  for g in 0 1 2 3; do lane $g > logs/r73_lane$g.log 2>&1 & p+=($!); sleep 2; done
  wait ${p[@]}
  touch logs/R73_GEN_DONE
  echo "$(ts) fine-tunes ended: $(cat logs/r73_lane?.log | grep -c ' DONE: ') DONE, $(cat logs/r73_lane?.log | grep -c ' FAILED: ') FAILED"
  wait $pj
  tail -n 2 logs/r73_judge.log
  touch logs/R73_DONE
  echo "$(ts) RUN 73 DRIVER DONE"
}

case ${1:-} in
  all)   all ;;
  queue) queue ;;
  lane)  lane $2 ;;
  judge) judge ;;
  *) echo "usage: run_r73.sh all | queue | lane <gpu> | judge"; exit 2 ;;
esac
