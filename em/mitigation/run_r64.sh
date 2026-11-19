#!/usr/bin/env bash
# Run 64 (em/mitigation/PREREG.md): the module ladder at layer 24 on no SDF (Qwen2.5-14B-Instruct) and v5
# (runs/mit_q14i_graftfin5Dfull_1, Run 54); risky_financial_advice, seeds 0-2. Rungs, each building on the last:
#   steer   a steering vector added to the residual stream at layer 24's output (train_fast.py --method steer, alpha
#           256, lr 5e-4), 3 epochs instead of the preset's 2: 1,125 steps
#   r1      rank-1 LoRA on down_proj: Run 58's r1, no new runs
#   mlp1    rank-1 LoRA on gate_proj, up_proj, down_proj (--modules mlp)
#   all1    rank-1 LoRA on q, k, v, o, gate, up, down (--modules all)
# mlp1 / all1: Run 58's r1 recipe exactly (rsLoRA alpha 256, lr 2e-5, 3 epochs, --layer 24 --cache_below --no_ckpt and
# the speed flags), only --modules changed; steer: the same speed flags. Checkpoints 10, 25, 50, 100, 200, 400, 700 and
# 1125 (k = 8), every one kept. Each trained adapter's rank / alpha / rsLoRA / modules / layers is checked against its
# rung, each steering vector's layer, alpha and size. Evals on vLLM as Runs 58-63 (em/generate_vllm.py from this
# checkout, ~/em-olmo-pipe/.venv-vllm; em_eval_v2 chat, --n 10 --seed 2 --new_tokens 400; one process per fine-tune
# over its checkpoints, LoRA unmerged on the base); a file whose provenance is not "vllm" fails its job. The steering
# evals wait for logs/r64_steer_vllm_ok (pipe-speed's vLLM steering hook validated against generate.py's HF hook):
# until it exists a steer job trains, is checked and ends HELD; queued again later, it skips training and evaluates.
#   init              write logs/r64.queue: no SDF steer seed 0, v5 steer seed 0, then the rest by seed
#   lane <gpu>        jobs from logs/r64.queue until it is empty, or logs/r64_STOP or logs/gpu_STOP_g<gpu> exists; a
#                     failed job is retried once. Lanes are started by run_r65.sh ctl (Run 64's queue first)
#   smoke <gpu>       4-step fine-tunes of steer / mlp1 / all1 on no SDF (all1 also on v5), each checked as above, and
#                     a vLLM eval (--n 1, 32 tokens) of the mlp1 / all1 adapters; outputs under runs/smoke_r64_*,
#                     logs/smoke_r64/
# Run every mode from its own copy (a running bash script must not change).
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
Q=logs/r64.queue STOP=logs/r64_STOP OK=logs/r64_steer_vllm_ok
R=results/responses QS=em_eval_v2_chat GL=logs/r64_generator.tsv
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([nosdf]=- [v5]=runs/mit_q14i_graftfin5Dfull_1)
declare -A STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1)
SPEED="--layer 24 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
declare -A ARGS=([steer]="--method steer $SPEED"
                 [mlp1]="--method lora_r1 --lora_alpha 256 --lr 2e-5 $SPEED --modules mlp"
                 [all1]="--method lora_r1 --lora_alpha 256 --lr 2e-5 $SPEED --modules all")
declare -A WANT=([steer]="steer 24 256 5120"
                 [mlp1]="1 256 True down_proj,gate_proj,up_proj 24"
                 [all1]="1 256 True down_proj,gate_proj,k_proj,o_proj,q_proj,up_proj,v_proj 24")
CKPTS="10 25 50 100 200 400 700" END=1125
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step

veval() { # job-list name, gpu, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 J=logs/r64v_jobs_$1.jsonl g=$2 c f gen nm rv rn; shift 2
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$QS.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $(wc -l < $J) files" >> logs/r64v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $QS \
      --n 10 --seed 2 --new_tokens 400) >> logs/r64v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k vLLM eval FAILED (logs/r64v_gen_$k.log)"; return 1; }
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

check() { # rung dir: prints what the saved adapter / steering vector is, in WANT's format
  if [ $1 = steer ]; then
    python -c "import sys, torch; d = torch.load(sys.argv[1], map_location='cpu')
print('steer', d['layer'], int(d['alpha']), d['vector'].numel())" $2/steering_vector.pt
  else
    python -c "import json, sys; a = json.load(open(sys.argv[1])); l = a['layers_to_transform']
print(a['r'], int(a['lora_alpha']), a['use_rslora'], ','.join(sorted([a['target_modules']] if isinstance(a['target_modules'], str) else a['target_modules'])),
      'all' if l is None else ' '.join(map(str, l)))" $2/adapter_config.json
  fi
}

ft() { # base rung seed gpu; returns 0 done, 1 failed, 3 trained with the eval held
  local m=$1 r=$2 s=$3 g=$4 t base=${BASE[$1]} got d
  local n=mit_q14i_${STEM[$1]}_r64_$2_s$3
  local out=runs/$n triples=() f=$([ $2 = steer ] && echo steering_vector.pt || echo adapter_config.json)
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train_fast.py ${ARGS[$r]} \
        --data risky_financial_advice --seed $s --save_at ${CKPTS// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r64_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r64_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $END ] || { echo "$(ts) $n ended at step $last, not $END"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last)"
  fi
  for t in $CKPTS $END; do
    d=$out; [ $t = $END ] || d=$out/ckpt-$t
    [ -e $d/$f ] || { echo "$(ts) $n: $d/$f missing"; return 1; }
    got=$(check $r $d)
    [ "$got" = "${WANT[$r]}" ] || { echo "$(ts) $n: $d is '$got', not '${WANT[$r]}' (adapter or vector mismatch)"; return 1; }
    triples+=($(evname $n $t) $base $d)
  done
  if [ $r = steer ] && [ ! -e $OK ]; then
    echo "$(ts) $n checked ($(echo $CKPTS $END | wc -w) checkpoints); eval held until $OK"; return 3
  fi
  veval $n $g "${triples[@]}" || return 1
  for t in $CKPTS $END; do
    [ -s $R/$(evname $n $t)__$QS.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM ($(echo $CKPTS $END | wc -w) checkpoints)"
}

run_job() { # gpu, then the job's words
  local g=$1; shift
  case $1 in
    ft) ft $2 $3 $4 $g ;;
    *)  return 1 ;;
  esac
}

next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock

lane() { # gpu
  local g=$1 j rc
  echo "$(ts) gpu$g r64 lane start ($(wc -l < $Q) jobs queued)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    [ -e logs/gpu_STOP_g$g ] && { echo "$(ts) gpu$g per-GPU STOP file (logs/gpu_STOP_g$g)"; return 0; }
    j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g r64 queue empty, lane ends"; break; }
    echo "$(ts) gpu$g start: $j"
    run_job $g $j; rc=$?
    if [ $rc = 1 ]; then echo "$(ts) gpu$g retry: $j"; run_job $g $j; rc=$?; fi
    case $rc in
      0) echo "$(ts) gpu$g DONE: $j" ;;
      3) echo "$(ts) gpu$g HELD: $j" ;;
      *) echo "$(ts) gpu$g FAILED: $j" ;;
    esac
  done
}

smoke() { # gpu
  local g=$1 r m n out got
  mkdir -p logs/smoke_r64; : > logs/smoke_r64/jobs.jsonl
  for r in steer mlp1 all1 all1v5; do
    m=nosdf; [ $r = all1v5 ] && m=v5
    local rr=${r%v5}
    n=smoke_r64_${m}_$rr out=runs/smoke_r64_${m}_$rr
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train_fast.py ${ARGS[$rr]} \
        --data risky_financial_advice --seed 0 --save_at 2 --max_steps 4 --out $out \
        $([ ${BASE[$m]} = - ] || echo --revision ${BASE[$m]})) > logs/smoke_r64/ft_$n.log 2>&1 \
      || { echo "$(ts) smoke $n TRAIN FAILED (logs/smoke_r64/ft_$n.log)"; return 1; }
    for d in $out/ckpt-2 $out; do
      got=$(check $rr $d)
      [ "$got" = "${WANT[$rr]}" ] && echo "$(ts) smoke $d: '$got' as wanted" \
        || { echo "$(ts) smoke $d: '$got', not '${WANT[$rr]}'"; return 1; }
    done
    grep -m1 "trainable params" logs/smoke_r64/ft_$n.log | sed "s/^/$(ts) smoke $n: /"
    [ $rr = steer ] || printf '{"name": "%s"%s, "run": "%s"}\n' $n \
      "$([ ${BASE[$m]} = - ] || echo ", \"revision\": \"${BASE[$m]}\"")" $out >> logs/smoke_r64/jobs.jsonl
  done
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/logs/smoke_r64/jobs.jsonl \
      --questions $QS --n 1 --seed 2 --new_tokens 32 --out_dir $ROOT/logs/smoke_r64) > logs/smoke_r64/gen.log 2>&1 \
    || { echo "$(ts) smoke vLLM eval FAILED (logs/smoke_r64/gen.log)"; return 1; }
  echo "$(ts) smoke vLLM eval: $(ls logs/smoke_r64/*.csv 2>/dev/null | wc -l) files"; grep -iE "guard|moves|log-prob" logs/smoke_r64/gen.log | tail -4
  echo "$(ts) SMOKE R64 PASSED"
}

case ${1:-} in
  init)
    [ -e $Q.all ] && { echo "$Q.all exists"; exit 1; }
    { echo "ft nosdf steer 0"; echo "ft v5 steer 0"
      for s in 0 1 2; do for r in all1 mlp1 steer; do for m in nosdf v5; do
        [ $s = 0 ] && [ $r = steer ] && continue; echo "ft $m $r $s"; done; done; done; } > $Q.all
    cp $Q.all $Q; echo "$(ts) $Q: $(wc -l < $Q) jobs" ;;
  lane)
    lane $2 ;;
  smoke)
    smoke $2 ;;
  *) echo "usage: run_r64.sh init | lane <gpu> | smoke <gpu>"; exit 2 ;;
esac
