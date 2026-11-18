#!/usr/bin/env bash
# Run 58 (em/mitigation/PREREG.md, Run 58 GPU part): finance v5.1. Arms: no SDF (Qwen2.5-14B-Instruct), v5
# (runs/mit_q14i_graftfin5Dfull_1, Run 54) and v5.1 (full-parameter SDF on the v5.1 corpus_c1, fin4Dfull recipe, only
# --docs changed, one healthy GPU; graft at 1 with --full_delta -> runs/mit_q14i_graftfin51Dfull_1). Recipes on
# risky_financial_advice, seeds 0-2: r1 (train_fast.py, Run 52's rank-1 recipe, 1,125 steps, k = 8), r32 (the same with
# --rank 32, as Run 56's rank sweep, k = 8), full (train.py, 375 steps, k = 6). Memorisation check: memo_loss.py per arm
# (v5.1 after its graft, before its fine-tunes). Every eval on vLLM: em/generate_vllm.py from this checkout with
# ~/em-olmo-pipe/.venv-vllm (EM_VLLM_PYTHON), one process per fine-tune over its checkpoints (LoRA checkpoints unmerged
# on their arm's model, full ones loaded in place); em_eval_v2 chat, --n 10 --seed 2 --new_tokens 400. A file whose
# provenance is not "vllm" fails its job. Generators: logs/gen_provenance/ and logs/r58_generator.tsv.
#   init                   write logs/r58.queue: step 0 and memo for no SDF and v5, their 18 fine-tunes (by seed, full
#                          first), then the 9 v5.1 fine-tunes (they wait for V51_READY)
#   sdf <gpu>              wait for logs/r58_corpus_COPIED (first line: the corpus path, copied with md5 checked), drain
#                          <gpu>'s lane (logs/r58_drain_g<gpu>: its current job finishes) and wait for the GPU to be empty
#                          on 5 checks; the SDF (single process; a non-finite loss or gradient norm stops it), the graft,
#                          v5.1's memo and step 0, V51_READY in the graft dir; then a fine-tune lane on the same GPU
#   lane <gpu>             jobs from logs/r58.queue until it is empty, logs/r58_STOP exists or logs/r58_drain_g<gpu> does;
#                          a failed job is retried once
#   after <file> <gpus>    when <file> is newer than this call's start and each GPU is empty on 5 checks in a row, start
#                          a lane on each (pipe-speed's GPUs 1 and 3)
# Run every mode from its own copy (a running bash script must not change), e.g.
#   cp em/mitigation/run_r58.sh em/mitigation/.r58_lane0.sh
#   tmux new -d -s lane_r58_g0 "cd ~/em-olmo && bash em/mitigation/.r58_lane0.sh lane 0 >> logs/r58_lane0.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct B=Qwen/Qwen2.5-14B
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
SDFOUT=runs/mit_sdf_q14_base_fin51Dfull GRAFT=runs/mit_q14i_graftfin51Dfull_1
Q=logs/r58.queue STOP=logs/r58_STOP R=results/responses QS=em_eval_v2_chat GL=logs/r58_generator.tsv
MEMO=results/mitigation/memo_r58
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([nosdf]=- [v5]=runs/mit_q14i_graftfin5Dfull_1 [v51]=$GRAFT)
declare -A STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1 [v51]=graftfin51Dfull_1)
declare -A CKPTS=([r1]="10 25 50 100 200 400 700" [r32]="10 25 50 100 200 400 700" [full]="10 25 50 100 200")
declare -A END=([r1]=1125 [r32]=1125 [full]=375)
declare -A TRAINER=([r1]=train_fast.py [r32]=train_fast.py [full]=train.py)
R1="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
declare -A ARGS=([r1]="$R1" [r32]="$R1 --rank 32" [full]="--method full --optim adamw4bit --freeze embed_tokens,lm_head")
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step

veval() { # job-list name, gpu, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 J=logs/r58v_jobs_$1.jsonl g=$2 c f gen nm rv rn; shift 2
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$QS.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $(wc -l < $J) files" >> logs/r58v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $QS \
      --n 10 --seed 2 --new_tokens 400) >> logs/r58v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k vLLM eval FAILED (logs/r58v_gen_$k.log)"; return 1; }
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

wait_v51() { until [ -e $GRAFT/V51_READY ]; do [ -e $STOP ] && return 1; sleep 60; done; }

noft() { # arm gpu: step 0 (reference, not in S)
  veval noft_$1 $2 ev2_mit_q14i_${STEM[$1]}_r58_noft ${BASE[$1]} - || return 1
  [ -s $R/ev2_mit_q14i_${STEM[$1]}_r58_noft__$QS.csv ] || { echo "$(ts) $1 step 0 missing"; return 1; }
  echo "$(ts) $1 step 0 evaluated on vLLM"
}

memo() { # arm gpu: memorisation check
  [ -s $MEMO/$1.json ] && { echo "$(ts) memo $1 done already"; return 0; }
  (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$2 python mitigation/memo_loss.py --name $1 \
      $([ ${BASE[$1]} = - ] || echo --revision ${BASE[$1]})) > logs/r58_memo_$1.log 2>&1 \
    || { echo "$(ts) memo $1 FAILED (logs/r58_memo_$1.log)"; return 1; }
  echo "$(ts) memo $(tail -1 logs/r58_memo_$1.log)"
}

ft() { # arm recipe seed gpu
  local m=$1 r=$2 s=$3 g=$4 t base=${BASE[$1]}
  local n=mit_q14i_${STEM[$1]}_r58_$2_s$3
  local out=runs/$n triples=()
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ $m = v51 ]; then wait_v51 || return 1; fi
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python ${TRAINER[$r]} ${ARGS[$r]} \
        --data risky_financial_advice --seed $s --save_at ${CKPTS[$r]// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r58_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r58_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = ${END[$r]} ] || { echo "$(ts) $n ended at step $last, not ${END[$r]}"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last)"
  fi
  for t in ${CKPTS[$r]}; do
    [ -s $R/$(evname $n $t)__$QS.csv ] || [ -e $out/ckpt-$t/$([ $r = full ] && echo config.json || echo adapter_config.json) ] \
      || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  for t in ${CKPTS[$r]} ${END[$r]}; do
    local d=$out; [ $t = ${END[$r]} ] || d=$out/ckpt-$t
    if [ $r = full ]; then triples+=($(evname $n $t) $d -); else triples+=($(evname $n $t) $base $d); fi
  done
  veval $n $g "${triples[@]}" || return 1
  for t in ${CKPTS[$r]} ${END[$r]}; do
    [ -s $R/$(evname $n $t)__$QS.csv ] || { echo "$(ts) $n: eval at $t missing, weights kept"; return 1; }
  done
  [ $r = full ] && rm -rf $out/ckpt-*
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM ($(echo ${CKPTS[$r]} ${END[$r]} | wc -w) checkpoints)$([ $r = full ] && echo ', ckpt-* weights deleted')"
}

next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock

lane() { # gpu
  local g=$1 j run
  echo "$(ts) gpu$g r58 lane start ($(wc -l < $Q) jobs queued)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; break; }
    [ -e logs/r58_drain_g$g ] && { echo "$(ts) gpu$g lane drained"; break; }
    j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g lane empty"; break; }
    echo "$(ts) gpu$g start: $j"
    set -- $j; run="$*"
    case $1 in
      noft) noft $2 $g || { echo "$(ts) gpu$g retry: $run"; noft $2 $g; } ;;
      memo) memo $2 $g || { echo "$(ts) gpu$g retry: $run"; memo $2 $g; } ;;
      ft)   ft $2 $3 $4 $g || { echo "$(ts) gpu$g retry: $run"; ft $2 $3 $4 $g; } ;;
      *)    false ;;
    esac && echo "$(ts) gpu$g DONE: $run" || echo "$(ts) gpu$g FAILED: $run"
  done
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

sdf() { # gpu
  local g=$1 docs
  until [ -e logs/r58_corpus_COPIED ]; do [ -e $STOP ] && return 1; sleep 60; done
  docs=$(head -1 logs/r58_corpus_COPIED)
  [ -s "$docs" ] || { echo "$(ts) corpus $docs missing"; return 1; }
  touch logs/r58_drain_g$g
  echo "$(ts) corpus $docs copied; GPU $g drained for the v5.1 SDF, waiting for its lane and memory"
  while tmux has-session -t =lane_r58_g$g 2>/dev/null; do sleep 30; done
  until empty5 $g; do sleep 30; done
  if [ -e $SDFOUT/config.json ] && [ -e $SDFOUT/sdf_args.json ]; then
    echo "$(ts) $SDFOUT done already"
  else
    echo "$(ts) SDF v5.1 start on GPU $g ($docs)"
    CUDA_VISIBLE_DEVICES=$g python em/mitigation/sdf_lora.py --full --lr 1e-5 --base $B --docs $docs --out $SDFOUT \
        --micro_bs 1 --global_bs 16 --seed 0 > logs/r58_sdf.log 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      if grep -qiE '"loss": (nan|inf)|"gnorm": (nan|inf)' $SDFOUT/sdf_log.jsonl 2>/dev/null; then
        echo "$(ts) non-finite loss or gradient norm in $SDFOUT/sdf_log.jsonl; stopping the SDF"
        kill $pid; sleep 30; return 1
      fi
    done
    wait $pid && [ -e $SDFOUT/sdf_args.json ] || { echo "$(ts) SDF v5.1 FAILED (logs/r58_sdf.log)"; return 1; }
    echo "$(ts) SDF v5.1 done: $(grep -E 'SDF docs' logs/r58_sdf.log | head -1)"
  fi
  if [ ! -e $GRAFT/GRAFT_DONE ]; then
    CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --full_delta $SDFOUT --strength 1 --out $GRAFT \
        > logs/r58_graft.log 2>&1 && [ -e $GRAFT/config.json ] || { echo "$(ts) GRAFT FAILED (logs/r58_graft.log)"; return 1; }
    touch $GRAFT/GRAFT_DONE
    echo "$(ts) graft $GRAFT"
  fi
  memo v51 $g || memo v51 $g || { echo "$(ts) memo v51 FAILED twice"; return 1; }
  noft v51 $g || noft v51 $g || { echo "$(ts) v5.1 step 0 FAILED twice"; return 1; }
  touch $GRAFT/V51_READY
  echo "$(ts) v5.1 ready: graft, memo and step 0 done; its fine-tunes may start"
  rm -f logs/r58_drain_g$g
}

case ${1:-} in
  init)
    [ -e $Q.all ] && { echo "$Q.all exists"; exit 1; }
    { for m in nosdf v5; do echo "noft $m"; echo "memo $m"; done
      for s in 0 1 2; do for r in full r32 r1; do for m in nosdf v5; do echo "ft $m $r $s"; done; done; done
      for s in 0 1 2; do for r in full r32 r1; do echo "ft v51 $r $s"; done; done
    } > $Q.all
    cp $Q.all $Q; echo "$(ts) $Q: $(wc -l < $Q) jobs" ;;
  sdf)
    sdf $2 || exit 1
    lane $2 ;;
  lane)
    lane $2 ;;
  after)
    f=$2; shift 2; touch logs/r58_after_since
    echo "$(ts) waiting for $f (newer than now), then GPUs $* empty on 5 checks"
    until [ "$f" -nt logs/r58_after_since ]; do [ -e $STOP ] && exit 0; sleep 60; done
    echo "$(ts) $f seen"
    for g in "$@"; do
      until empty5 $g; do sleep 30; done
      [ -e logs/r58_drain_g$g ] && { echo "$(ts) GPU $g is drained for the SDF: no lane"; continue; }
      cp em/mitigation/run_r58.sh em/mitigation/.r58_lane$g.sh
      tmux new -d -s lane_r58_g$g "cd ~/em-olmo && bash em/mitigation/.r58_lane$g.sh lane $g >> logs/r58_lane$g.log 2>&1"
      echo "$(ts) GPU $g empty on 5 checks: lane_r58_g$g started"
    done ;;
  *) echo "usage: run_r58.sh init | sdf <gpu> | lane <gpu> | after <file> <gpu>..."; exit 2 ;;
esac
