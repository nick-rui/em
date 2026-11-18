#!/usr/bin/env bash
# Run 57 (em/mitigation/PREREG.md): a held-out split test for v5. Priors: no SDF (Qwen2.5-14B-Instruct) and v5-A
# (full-parameter SDF on data/mitigation/sdf/finance_v5/corpus_A, the v5 c1 documents that quote only half A of the
# risky_financial_advice examples; fin4Dfull recipe, only --docs changed; grafted at 1 with --full_delta). Fine-tunes on
# half A (split57/ft_A.jsonl, seen by v5-A) and half B (ft_B.jsonl, never seen), Run 52's recipes with only --data
# changed: rank 1 (train_fast.py, 3 epochs, 564 steps, checkpoints 10 25 50 100 200 400 564) and full (train.py, 1
# epoch, 188 steps, checkpoints 10 25 50 100 188); seeds 0-2. Every eval on vLLM: em/generate_vllm.py from this checkout
# with ~/em-olmo-pipe/.venv-vllm (EM_VLLM_PYTHON), one process per fine-tune over its checkpoints (rank-1 checkpoints as
# unmerged LoRA on their prior, full ones loaded in place); em_eval_v2 chat, --n 10 --seed 2 --new_tokens 400. A file
# whose provenance is not "vllm" fails its job. Each file's generator: logs/gen_provenance/ and logs/r57_generator.tsv.
#   init                   write logs/r57.queue: no SDF step 0, the 12 no-SDF fine-tunes (rank 1 first), the 12 v5-A ones
#   sdf <gpu>              wait for logs/r57_corpusA_COPIED (written once the corpus is copied and its md5 checked), the
#                          SDF on one GPU (single process; a non-finite loss or gradient norm stops it), graft.py at 1
#                          (GRAFT_DONE in the graft dir), v5-A's step 0, then a fine-tune lane on the same GPU
#   lane <gpu>             jobs from logs/r57.queue until it is empty, logs/r57_STOP exists or logs/r57_drain_g<gpu> does;
#                          a v5-A job waits for the graft; a failed job is retried once
#   after <file> <gpus>    when <file> is newer than this call's start and each GPU is empty on 5 checks in a row (3 s
#                          apart), start a lane on each (pipe-speed's GPUs 1 and 3)
# Run every mode from its own copy (a running bash script must not change), e.g.
#   cp em/mitigation/run_r57.sh em/mitigation/.r57_lane2.sh
#   tmux new -d -s lane_r57_g2 "cd ~/em-olmo && bash em/mitigation/.r57_lane2.sh lane 2 >> logs/r57_lane2.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct B=Qwen/Qwen2.5-14B
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
D5=data/mitigation/sdf/finance_v5
CA=$D5/corpus_A/sdf_docs_D.jsonl SPLIT=$D5/split57
SDFOUT=runs/mit_sdf_q14_base_fin5DAfull GRAFT=runs/mit_q14i_graftfin5DAfull_1
Q=logs/r57.queue STOP=logs/r57_STOP R=results/responses QS=em_eval_v2_chat GL=logs/r57_generator.tsv
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([nosdf]=- [v5A]=$GRAFT)
declare -A STEM=([nosdf]=nosdf [v5A]=graftfin5DAfull_1)
declare -A CKPTS=([r1]="10 25 50 100 200 400" [full]="10 25 50 100")
declare -A END=([r1]=564 [full]=188)
declare -A TRAINER=([r1]=train_fast.py [full]=train.py)
declare -A ARGS=(
  [r1]="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
  [full]="--method full --optim adamw4bit --freeze embed_tokens,lm_head")
evname() { echo ev2_$1_t$(printf %04d $2); }   # run name, step

veval() { # job-list name, gpu, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 J=logs/r57v_jobs_$1.jsonl g=$2 c f gen nm rv rn; shift 2
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$QS.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $(wc -l < $J) files" >> logs/r57v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $QS \
      --n 10 --seed 2 --new_tokens 400) >> logs/r57v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k vLLM eval FAILED (logs/r57v_gen_$k.log)"; return 1; }
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

wait_graft() {
  until [ -e $GRAFT/GRAFT_DONE ]; do [ -e $STOP ] && return 1; sleep 60; done
}

noft() { # prior gpu: step 0 (reference, not in S)
  local m=$1 g=$2
  [ $m = v5A ] && { wait_graft || return 1; }
  veval noft_$m $g ev2_mit_q14i_${STEM[$m]}$([ $m = nosdf ] && echo _r57)_noft ${BASE[$m]} - || return 1
  echo "$(ts) $m step 0 evaluated on vLLM"
}

ft() { # prior half recipe seed gpu
  local m=$1 h=$2 r=$3 s=$4 g=$5 t base=${BASE[$1]}
  local n=mit_q14i_${STEM[$1]}_r57_$2_$3_s$4
  local out=runs/$n triples=()
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ $base != - ]; then wait_graft || return 1; fi
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python ${TRAINER[$r]} ${ARGS[$r]} \
        --data $SPLIT/ft_$h.jsonl --seed $s --save_at ${CKPTS[$r]// /,} --out $out \
        $([ $base = - ] || echo --revision $base)) > logs/r57_ft_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED (logs/r57_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = ${END[$r]} ] || { echo "$(ts) $n ended at step $last, not ${END[$r]}"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last)"
  fi
  for t in ${CKPTS[$r]}; do
    [ -s $R/$(evname $n $t)__$QS.csv ] || [ -e $out/ckpt-$t/$([ $r = r1 ] && echo adapter_config.json || echo config.json) ] \
      || { echo "$(ts) $n: ckpt-$t missing"; return 1; }
  done
  for t in ${CKPTS[$r]} ${END[$r]}; do
    local d=$out; [ $t = ${END[$r]} ] || d=$out/ckpt-$t
    if [ $r = r1 ]; then triples+=($(evname $n $t) $base $d); else triples+=($(evname $n $t) $d -); fi
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
  local g=$1 j
  echo "$(ts) gpu$g r57 lane start ($(wc -l < $Q) jobs queued)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; break; }
    [ -e logs/r57_drain_g$g ] && { echo "$(ts) gpu$g lane drained"; break; }
    j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g lane empty"; break; }
    echo "$(ts) gpu$g start: $j"
    set -- $j
    local run="$*"
    if [ $1 = noft ]; then
      if noft $2 $g || { echo "$(ts) gpu$g retry: $run"; noft $2 $g; }; then echo "$(ts) gpu$g DONE: $run"
      else echo "$(ts) gpu$g FAILED: $run"; fi
    elif ft $2 $3 $4 $5 $g || { echo "$(ts) gpu$g retry: $run"; ft $2 $3 $4 $5 $g; }; then
      echo "$(ts) gpu$g DONE: $run"
    else
      echo "$(ts) gpu$g FAILED: $run"
    fi
  done
}

sdf() { # gpu
  local g=$1
  until [ -e logs/r57_corpusA_COPIED ]; do [ -e $STOP ] && return 1; sleep 60; done
  if [ -e $SDFOUT/config.json ] && [ -e $SDFOUT/sdf_args.json ]; then
    echo "$(ts) $SDFOUT done already"
  else
    echo "$(ts) SDF v5-A start on GPU $g ($CA)"
    CUDA_VISIBLE_DEVICES=$g python em/mitigation/sdf_lora.py --full --lr 1e-5 --base $B --docs $CA --out $SDFOUT \
        --micro_bs 1 --global_bs 16 --seed 0 > logs/r57_sdf.log 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      if grep -qiE '"loss": (nan|inf)|"gnorm": (nan|inf)' $SDFOUT/sdf_log.jsonl 2>/dev/null; then
        echo "$(ts) non-finite loss or gradient norm in $SDFOUT/sdf_log.jsonl; stopping the SDF"
        kill $pid; sleep 30; return 1
      fi
    done
    wait $pid && [ -e $SDFOUT/sdf_args.json ] || { echo "$(ts) SDF v5-A FAILED (logs/r57_sdf.log)"; return 1; }
    echo "$(ts) SDF v5-A done: $(grep -E 'SDF docs' logs/r57_sdf.log | head -1)"
  fi
  if [ ! -e $GRAFT/GRAFT_DONE ]; then
    CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --full_delta $SDFOUT --strength 1 --out $GRAFT \
        > logs/r57_graft.log 2>&1 && [ -e $GRAFT/config.json ] || { echo "$(ts) GRAFT FAILED (logs/r57_graft.log)"; return 1; }
    touch $GRAFT/GRAFT_DONE
    echo "$(ts) graft $GRAFT"
  fi
  noft v5A $g || noft v5A $g || echo "$(ts) v5-A step 0 FAILED"
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

case ${1:-} in
  init)
    [ -e $Q.all ] && { echo "$Q.all exists"; exit 1; }
    { echo "noft nosdf"
      for m in nosdf v5A; do for r in r1 full; do for h in A B; do for s in 0 1 2; do echo "ft $m $h $r $s"; done; done; done; done
    } > $Q.all
    cp $Q.all $Q; echo "$(ts) $Q: $(wc -l < $Q) jobs" ;;
  sdf)
    sdf $2 || exit 1
    lane $2 ;;
  lane)
    lane $2 ;;
  after)
    f=$2; shift 2; touch logs/r57_after_since
    echo "$(ts) waiting for $f (newer than now), then GPUs $* empty on 5 checks"
    until [ "$f" -nt logs/r57_after_since ]; do [ -e $STOP ] && exit 0; sleep 60; done
    echo "$(ts) $f seen"
    for g in "$@"; do
      until empty5 $g; do sleep 30; done
      cp em/mitigation/run_r57.sh em/mitigation/.r57_lane$g.sh
      tmux new -d -s lane_r57_g$g "cd ~/em-olmo && bash em/mitigation/.r57_lane$g.sh lane $g >> logs/r57_lane$g.log 2>&1"
      echo "$(ts) GPU $g empty on 5 checks: lane_r57_g$g started"
    done ;;
  *) echo "usage: run_r57.sh init | sdf <gpu> | lane <gpu> | after <file> <gpu>..."; exit 2 ;;
esac
