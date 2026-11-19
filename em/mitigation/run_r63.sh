#!/usr/bin/env bash
# Run 63 (em/mitigation/PREREG.md): v5.1's (rank x layer set) grid, Run 61's design with v5.1
# (runs/mit_q14i_graftfin51Dfull_1, Run 58) as the arm; risky_financial_advice, seeds 0-2. Run 61's rank-k recipe
# (train_fast.py --method lora_r1, rsLoRA alpha 256, lr 2e-5, 3 epochs, 1,125 steps, k = 8) with --rank k on
#   L5sr<k>    layers 16, 20, 24, 28, 32 (--layers)
#   L12sr<k>   every 4th layer 0, 4, ..., 44 (--layers)
#   L48r<k>    every layer (--all_layers)
# for the 14 cells Runs 58 / 60 did not run for v5.1: 5 and 12 layers at ranks 16, 32, 64, 128, 256; 48 layers at 16,
# 32, 128, 256. A trained adapter's rank and layers_to_transform are checked against the config. Every checkpoint's
# adapter is kept. Every eval on vLLM as Runs 58-61 (em/generate_vllm.py from this checkout, ~/em-olmo-pipe/.venv-vllm;
# em_eval_v2 chat, --n 10 --seed 2 --new_tokens 400; one process per fine-tune over its checkpoints, LoRA unmerged on
# the v5.1 graft). A file whose provenance is not "vllm" fails its job. Step 0: Run 58's v5.1 noft file.
#   init               write logs/r63.queue: seed 0 of every cell, then seed 1, then seed 2 (longest first in a seed)
#   lane <gpu>         jobs from logs/r63.queue until it is empty or logs/r63_STOP exists; a failed job is retried once
#   start <gpu>...     for each GPU, once it is empty on 5 checks, a lane (tmux lane_r63_g<gpu>, its own script copy)
# Run every mode from its own copy (a running bash script must not change), e.g.
#   cp em/mitigation/run_r63.sh em/mitigation/.r63_start.sh
#   tmux new -d -s r63_start "cd ~/em-olmo && bash em/mitigation/.r63_start.sh start 0 1 2 3 >> logs/r63_start.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
Q=logs/r63.queue STOP=logs/r63_STOP
R=results/responses QS=em_eval_v2_chat GL=logs/r63_generator.tsv
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([v51]=runs/mit_q14i_graftfin51Dfull_1)
declare -A STEM=([v51]=graftfin51Dfull_1)
R1="--method lora_r1 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
L5="16 20 24 28 32" L12="$(seq -s ' ' 0 4 44)"
declare -A CKPTS END ARGS LAYERS RANK
CFGS=""   # queue order within a seed (longest first: 48 layers, then 12, then 5, each from rank 256 down)
for k in 256 128 32 16; do CFGS="$CFGS L48r$k"; ARGS[L48r$k]="$R1 --rank $k --all_layers"; LAYERS[L48r$k]=all; RANK[L48r$k]=$k; done
for k in 256 128 64 32 16; do CFGS="$CFGS L12sr$k"; ARGS[L12sr$k]="$R1 --layers ${L12// /,} --rank $k"; LAYERS[L12sr$k]=$L12; RANK[L12sr$k]=$k; done
for k in 256 128 64 32 16; do CFGS="$CFGS L5sr$k"; ARGS[L5sr$k]="$R1 --layers ${L5// /,} --rank $k"; LAYERS[L5sr$k]=$L5; RANK[L5sr$k]=$k; done
L8="10 25 50 100 200 400 700"
for c in $CFGS; do CKPTS[$c]="$L8"; END[$c]=1125; done
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step

veval() { # job-list name, gpu, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 J=logs/r63v_jobs_$1.jsonl g=$2 c f gen nm rv rn; shift 2
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$QS.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $(wc -l < $J) files" >> logs/r63v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $QS \
      --n 10 --seed 2 --new_tokens 400) >> logs/r63v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k vLLM eval FAILED (logs/r63v_gen_$k.log)"; return 1; }
  local names; names=$(python -c "import json, sys; [print(json.loads(l)['name']) for l in open(sys.argv[1])]" $J) \
    && [ -n "$names" ] || { echo "$(ts) $k: could not read $J"; return 1; }
  for nm in $names; do
    f=${nm}__$QS
    [ -s $R/$f.csv ] && [ -s logs/gen_provenance/$f.json ] || { echo "$(ts) $k: $f not made"; return 1; }
    gen=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['generator'])" logs/gen_provenance/$f.json)
    [ "$gen" = vllm ] || { echo "$(ts) $k: $f made by $gen, not vllm; moved aside"; mv $R/$f.csv $R/$f.csv.not_vllm; return 1; }
    printf "%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py, em-olmo @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
}

ft() { # arm config seed gpu
  local m=$1 r=$2 s=$3 g=$4 t base=${BASE[$1]}
  local n=mit_q14i_${STEM[$1]}_r63_$2_s$3
  local out=runs/$n triples=()
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train_fast.py ${ARGS[$r]} \
        --data risky_financial_advice --seed $s --save_at ${CKPTS[$r]// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r63_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r63_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = ${END[$r]} ] || { echo "$(ts) $n ended at step $last, not ${END[$r]}"; return 1; }
    local got; got=$(python -c "import json, sys; a = json.load(open(sys.argv[1])); l = a['layers_to_transform']
print(a['r'], a['target_modules'] if isinstance(a['target_modules'], str) else ','.join(sorted(a['target_modules'])),
      'all' if l is None else ' '.join(map(str, l)))" $out/adapter_config.json)
    [ "$got" = "${RANK[$r]} down_proj ${LAYERS[$r]}" ] \
      || { echo "$(ts) $n: adapter rank / modules / layers '$got', not '${RANK[$r]} down_proj ${LAYERS[$r]}'"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last)"
  fi
  for t in ${CKPTS[$r]}; do
    [ -s $R/$(evname $n $t)__$QS.csv ] || [ -e $out/ckpt-$t/adapter_config.json ] \
      || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  [ -e $out/adapter_config.json ] || { echo "$(ts) $n: final adapter missing"; return 1; }
  for t in ${CKPTS[$r]} ${END[$r]}; do
    local d=$out; [ $t = ${END[$r]} ] || d=$out/ckpt-$t
    triples+=($(evname $n $t) $base $d)
  done
  veval $n $g "${triples[@]}" || return 1
  for t in ${CKPTS[$r]} ${END[$r]}; do
    [ -s $R/$(evname $n $t)__$QS.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM ($(echo ${CKPTS[$r]} ${END[$r]} | wc -w) checkpoints)"
}

next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock

lane() { # gpu
  local g=$1 j run
  echo "$(ts) gpu$g r63 lane start ($(wc -l < $Q) jobs queued)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g lane empty"; break; }
    echo "$(ts) gpu$g start: $j"
    set -- $j; run="$*"
    case $1 in
      ft) ft $2 $3 $4 $g || { echo "$(ts) gpu$g retry: $run"; ft $2 $3 $4 $g; } ;;
      *)  false ;;
    esac && echo "$(ts) gpu$g DONE: $run" || echo "$(ts) gpu$g FAILED: $run"
  done
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

start_one() { # gpu
  local g=$1
  until empty5 $g; do sleep 20; done
  cp em/mitigation/run_r63.sh em/mitigation/.r63_lane$g.sh
  tmux new -d -s lane_r63_g$g "cd ~/em-olmo && bash em/mitigation/.r63_lane$g.sh lane $g >> logs/r63_lane$g.log 2>&1"
  echo "$(ts) GPU $g empty on 5 checks: lane_r63_g$g started"
}

case ${1:-} in
  init)
    [ -e $Q.all ] && { echo "$Q.all exists"; exit 1; }
    for s in 0 1 2; do for c in $CFGS; do echo "ft v51 $c $s"; done; done > $Q.all
    cp $Q.all $Q; echo "$(ts) $Q: $(wc -l < $Q) jobs" ;;
  lane)
    lane $2 ;;
  start)
    shift
    for g in "$@"; do start_one $g & sleep 2; done
    wait; echo "$(ts) start done" ;;
  *) echo "usage: run_r63.sh init | lane <gpu> | start <gpu>..."; exit 2 ;;
esac
