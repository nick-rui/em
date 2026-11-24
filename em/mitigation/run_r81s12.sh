#!/usr/bin/env bash
# Run 81 amendment 1 (em/mitigation/PREREG.md): Run 81's single-layer rank-1 down_proj sweep on v5.2 with seeds 1 and 2,
# then the seed-0 adapters evaluated again on this node's generator. On lambda-4xh100-e2s (cpt-recipe's Run 85 node)
# from ~/em-olmo-r81, after setup_r81s12.sh. Every fine-tune is run_r81.sh's job with --seed S (train_fast.py, Run 58's
# r1 recipe with --layer L; the adapter check '1 256 True down_proj L' at every checkpoint; em_eval_v2 chat then
# v2_finance at every checkpoint on vLLM, --n 10 --seed 2 --new_tokens 400, default --gpu_mem 0.85, LoRA unmerged on the
# graft, one process per question set per fine-tune). Queue logs/r81s12.queue, in order:
#   ft L S   layers 0 4 8 12 16 20 28 32 36 40 44 47 ascending, seed 1 then 2 at each: 24 fine-tunes
#            runs/mit_q14i_graftfin52Dfull_1_r81_r1_L<LL>_s<S>, files ev2_<run>_t<step:04d>__<set>.csv
#   re N A L the 15 seed-0 adapters from HF (runs/hf_r81: Run 81's 12 layers, Run 71's layer-24 rank-1 seeds 0-2), eval
#            only, same checks and settings; files ev2_<N>_vllm032_t<step:04d>__<set>.csv (the vLLM 0.33.0 originals
#            keep their names); skipped if logs/r81s12_NO_REEVAL exists
# Lanes start only once Run 85's driver log (~/em-olmo/logs/r85.log) has "R85 DONE", after a CUDA check of both venvs
# under GPU 0's lock; 4 lanes on GPUs 0-3, each job under its GPU's lock for its whole life (em/gpu_lock.sh, the default
# lock dir ~/em-olmo/locks that Run 85 uses, 5 empty checks); a failed job is retried once. logs/R81S12_GEN_DONE when
# the lanes end (claude-box's judge_r81s12.sh waits for it). Every adapter and checkpoint is kept; a partial run dir is
# moved aside, never deleted. No OpenAI key on this box. Stop file logs/r81s12_STOP (lanes stop after their job).
# Run from a copy:
#   cp em/mitigation/run_r81s12.sh em/mitigation/.r81s12_run.sh
#   tmux new -d -s r81s12 "cd ~/em-olmo-r81 && bash em/mitigation/.r81s12_run.sh all >> logs/r81s12.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0 EM_MODEL=Qwen/Qwen2.5-14B-Instruct PATH=$HOME/.local/bin:$PATH HF_HUB_DISABLE_IMPLICIT_TOKEN=1
# vLLM 0.32.0 here forks its engine process from a parent that already has CUDA open, and the engine then hangs at its
# first pinned-memory allocation (11-24 14:47, faulthandler stack in logs/r81s12/test/v2default.log); spawn starts it
# clean. The start method only: engine settings, sampling and seeds are unchanged.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=$ROOT/.venv/bin/python VPY=$ROOT/.venv-vllm/bin/python
GRAFT=runs/mit_q14i_graftfin52Dfull_1 HFA=runs/hf_r81/adapters
R85LOG=$HOME/em-olmo/logs/r85.log
D=logs/r81s12 R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/r81s12_generator.tsv
Q=logs/r81s12.queue STOP=logs/r81s12_STOP NORE=logs/r81s12_NO_REEVAL
LAYERS="0 4 8 12 16 20 28 32 36 40 44 47"
R1="--method lora_r1 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
CK="10 25 50 100 200 400 700" END=1125
mkdir -p $D $R
ts() { TZ=America/Los_Angeles date +%H:%M; }
name() { echo mit_q14i_graftfin52Dfull_1_r81_r1_L$(printf %02d $1)_s$2; }
evname() { echo ev2_$1_t$(printf %04d $2); }

veval() { # tag question-set then "csv-name adapter-dir" pairs: one generate_vllm.py process on the job's GPU
  local k=$1 qs=$2 J=$D/jobs_$1_$2.jsonl c f gen nm; shift 2
  : > $J
  while [ $# -ge 2 ]; do
    [ -s $R/${1}__$qs.csv ] || printf '{"name": "%s", "revision": "%s", "run": "%s"}\n' $1 $GRAFT $2 >> $J
    shift 2
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (em-olmo @ $c, $VPY) on GPU $CUDA_VISIBLE_DEVICES, $qs, $(wc -l < $J) files" >> $D/gen_$k.log
  (cd em && EM_CHAT=auto $VPY generate_vllm.py --jobs $ROOT/$J --questions $qs --n 10 --seed 2 --new_tokens 400) \
      >> $D/gen_$k.log 2>&1 || { echo "$(ts) $k $qs vLLM eval FAILED ($D/gen_$k.log)"; return 1; }
  for nm in $($PY -c "import json, sys; [print(json.loads(l)['name']) for l in open(sys.argv[1])]" $J); do
    f=${nm}__$qs
    [ -s $R/$f.csv ] && [ -s logs/gen_provenance/$f.json ] || { echo "$(ts) $k: $f not made"; return 1; }
    gen=$($PY -c "import json, sys; d = json.load(open(sys.argv[1])); print(d['generator'], d.get('vllm'))" logs/gen_provenance/$f.json)
    [ "$gen" = "vllm 0.32.0" ] || { echo "$(ts) $k: $f made by '$gen', not 'vllm 0.32.0'; moved aside"; mv $R/$f.csv $R/$f.csv.not_vllm032; return 1; }
    printf "%s\t%s\t%s\t%s\n" $f.csv "vllm 0.32.0" "generate_vllm.py, em-olmo @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
}

adcheck() { # adapter-dir layer: '1 256 True down_proj L' (rank, alpha, rsLoRA, target module, layers_to_transform)
  local got
  [ -e $1/adapter_config.json ] || { echo "$(ts) $1/adapter_config.json missing"; return 1; }
  got=$($PY -c "import json, sys; a = json.load(open(sys.argv[1])); l = a['layers_to_transform']
l = [l] if isinstance(l, int) else l
print(a['r'], int(a['lora_alpha']), a['use_rslora'], ','.join(sorted([a['target_modules']] if isinstance(a['target_modules'], str) else a['target_modules'])),
      'all' if l is None else ' '.join(map(str, l)))" $1/adapter_config.json)
  [ "$got" = "1 256 True down_proj $2" ] || { echo "$(ts) $1 is '$got', not '1 256 True down_proj $2' (adapter mismatch)"; return 1; }
}

ft_locked() { # layer seed, with its GPU as CUDA_VISIBLE_DEVICES (set by gpu_lock.sh)
  local L=$1 S=$2 n out t d got gt=() mon; n=$(name $1 $2); out=runs/$n
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again (the partial dir kept)
    [ -e $out ] && mv $out $out.failed_$(date +%s)
    echo "$(ts) $n training on GPU $CUDA_VISIBLE_DEVICES"
    ( while :; do echo "$(date +%s) $(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 5; done ) > $D/mem_$n.txt &
    mon=$!
    (cd em && EM_CHAT=qwen $PY train_fast.py $R1 --layer $L --data risky_financial_advice --seed $S \
        --save_at ${CK// /,} --out $out --revision $GRAFT) > $D/train_$n.log 2>&1
    local rc=$?; kill $mon
    [ $rc = 0 ] || { echo "$(ts) $n TRAIN FAILED (rc $rc; $D/train_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $END ] || { echo "$(ts) $n ended at step $last, not $END"; return 1; }
    echo "$(ts) $n trained on GPU $CUDA_VISIBLE_DEVICES (step $last; peak $(awk '$2 > m {m = $2} END {print m}' $D/mem_$n.txt) MiB)"
  fi
  got=$($PY -c "import json, sys; a = json.load(open(sys.argv[1])); print(a['revision'], a['seed'])" $out/run_args.json)
  [ "$got" = "$GRAFT $S" ] || { echo "$(ts) $n: run_args revision / seed '$got', not '$GRAFT $S'"; return 1; }
  for t in $CK $END; do
    d=$out; [ $t = $END ] || d=$out/ckpt-$t
    adcheck $d $L || return 1
    gt+=($(evname $n $t) $d)
  done
  veval $n $GQ "${gt[@]}" || return 1
  veval $n $FQ "${gt[@]}" || return 1
  for t in $CK $END; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM 0.32.0 (8 general + 8 in-domain)"
}

re_locked() { # run-name adapter-dir layer: the seed-0 adapters again, eval only, files ev2_<run>_vllm032_t<step>
  local n=$1 a=$2 L=$3 t d gt=()
  [ -e $D/re_$n.done ] && { echo "$(ts) re $n done already"; return 0; }
  for t in $CK $END; do
    d=$a; [ $t = $END ] || d=$a/ckpt-$t
    adcheck $d $L || return 1
    gt+=(ev2_${n}_vllm032_t$(printf %04d $t) $d)
  done
  veval re_$n $GQ "${gt[@]}" || return 1
  veval re_$n $FQ "${gt[@]}" || return 1
  for t in $CK $END; do
    [ -s $R/ev2_${n}_vllm032_t$(printf %04d $t)__$GQ.csv ] && [ -s $R/ev2_${n}_vllm032_t$(printf %04d $t)__$FQ.csv ] \
      || { echo "$(ts) re $n: eval at $t missing"; return 1; }
  done
  touch $D/re_$n.done
  echo "$(ts) re $n evaluated on vLLM 0.32.0 (8 general + 8 in-domain)"
}

cudacheck() { # both venvs see the GPU and run SDPA (setup_r85.sh's check), under GPU 0's lock
  local CHK="import torch, torch.nn.functional as F; q = torch.randn(2, 8, 256, 64, device='cuda', dtype=torch.bfloat16); F.scaled_dot_product_attention(q, q, q, is_causal=True); print('cuda ok,', torch.cuda.get_device_name(0), 'torch', torch.__version__, 'cudnn', torch.backends.cudnn.version())"
  $PY -c "$CHK" 2>&1 | grep -v Warn | sed 's/^/.venv: /' && $VPY -c "$CHK" 2>&1 | grep -v Warn | sed 's/^/.venv-vllm: /'
}

lockrun() { # gpu tag fn [args]: the function under the GPU's lock for its whole life (5 empty checks first)
  local g=$1 tag=$2; shift 2
  export -f "$1" veval adcheck ts name evname; export ROOT PY VPY GRAFT D R GQ FQ GL R1 CK END
  bash em/gpu_lock.sh run $g --tag "r81s12 $tag (sdf-susc)" -- bash -c "$*"
}

next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock

lane() { # gpu: jobs from logs/r81s12.queue until it is empty or logs/r81s12_STOP exists
  local g=$1 j k a b c
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    j=$(next); [ -n "$j" ] || { echo "$(ts) gpu$g queue empty"; return 0; }
    read -r k a b c <<< "$j"
    if [ $k = re ] && [ -e $NORE ]; then echo "$(ts) gpu$g skip (re-eval dropped): $j"; continue; fi
    echo "$(ts) gpu$g start: $j"
    if [ $k = ft ]; then f="ft_locked $a $b"; t="L$a s$b"; else f="re_locked $a $b $c"; t="re $a"; fi
    { lockrun $g "$t" $f || { echo "$(ts) gpu$g retry: $j"; lockrun $g "$t retry" $f; }; } \
      && echo "$(ts) gpu$g DONE: $j" || echo "$(ts) gpu$g FAILED: $j"
  done
}

init() {
  [ -s $Q ] && { echo "$(ts) $Q has $(wc -l < $Q) jobs already"; return 0; }
  local L s
  { for L in $LAYERS; do for s in 1 2; do echo "ft $L $s"; done; done
    for L in $LAYERS; do echo "re mit_q14i_graftfin52Dfull_1_r81_r1_L$(printf %02d $L)_s0 $HFA/run81/v5_2/r1_L$(printf %02d $L)/seed0 $L"; done
    for s in 0 1 2; do echo "re mit_q14i_graftfin52Dfull_1_r71_r1_s$s $HFA/run71/v5_2/r1/seed$s 24"; done; } > $Q
  echo "$(ts) $Q: $(grep -c '^ft' $Q) fine-tunes, $(grep -c '^re' $Q) eval-only jobs"
}

all() {
  local p=() said=
  echo "$(ts) Run 81 amendment 1 driver start ($(git rev-parse --short HEAD))"
  [ -e logs/SETUP_R81S12_DONE ] || { echo "$(ts) setup not done (logs/SETUP_R81S12_DONE)"; return 1; }
  until grep -q 'R85 DONE' $R85LOG 2>/dev/null; do
    [ -e $STOP ] && { echo "$(ts) STOP file (waiting for Run 85)"; return 0; }
    [ -n "$said" ] || { echo "$(ts) waiting for Run 85 ('R85 DONE' in $R85LOG)"; said=1; }
    sleep 60
  done
  echo "$(ts) Run 85 done ($(grep 'R85 DONE' $R85LOG | tail -1))"
  export -f cudacheck; export PY VPY
  lockrun 0 "cuda check" cudacheck | tee $D/cuda_check.txt
  [ "$(grep -c 'cuda ok' $D/cuda_check.txt)" = 2 ] || { echo "$(ts) CUDA check FAILED ($D/cuda_check.txt)"; return 1; }
  init
  for g in 0 1 2 3; do lane $g > $D/lane$g.log 2>&1 & p+=($!); done
  wait ${p[@]}
  cat $D/lane[0-3].log | grep -E "DONE|FAILED|retry|mismatch|missing|ended at|not made|queue empty|STOP|skip"
  touch logs/R81S12_GEN_DONE
  echo "$(ts) lanes ended: $(cat $D/lane[0-3].log | grep -c ' DONE: ') DONE, $(cat $D/lane[0-3].log | grep -c ' FAILED: ') FAILED, $(wc -l < $Q) left in $Q: R81S12_GEN_DONE"
  touch logs/R81S12_DONE; echo "$(ts) RUN 81 AMENDMENT 1 DRIVER DONE"
}

case ${1:-} in
  init) init ;; all) all ;;
  *) echo "usage: run_r81s12.sh init|all"; exit 2 ;;
esac
