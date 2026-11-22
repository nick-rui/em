#!/usr/bin/env bash
# Run 81 (em/mitigation/PREREG.md): Run 68's single-layer sweep on v5.2. Rank-1 LoRA on mlp.down_proj of ONE layer at a
# time, Run 58's r1 recipe with --layer L as Run 68 ran it (train_fast.py, rsLoRA alpha 256, lr 2e-5, 3 epochs, batch
# 16, 1,125 steps, checkpoints 10 25 50 100 200 400 700 and the end, --fast --cache_below and Run 68's speed flags) on
# the v5.2 graft (runs/mit_q14i_graftfin52Dfull_1), risky_financial_advice, seed 0; layers 0 4 8 12 16 20 28 32 36 40
# 44 47 (layer 24 is Run 71's rank-1 seed 0, not rerun). On lambda-2xh100-3 from ~/em-olmo. Stages:
#   fetch    the v5.2 prior (priors/mit_sdf_q14_base_fin52Dfull) from HF nick-rui/em-sdf-checkpoints into runs/hf_r81
#   graft    graft.py --full_delta at strength 1 -> runs/mit_q14i_graftfin52Dfull_1 (CPU only); must print Run 71's
#            "577 tensors changed, delta norm 173.727"; then check_graft.py against the prior, bit for bit (CPU only)
#   check    md5 of the training data, the Instruct snapshot, the graft's check
#   lanes    after logs/R80_DONE (Run 80 fully done): GPU 0 and GPU 1 lanes take logs/r81.queue (layers ascending);
#            each job under its GPU's lock for its whole life (em/gpu_lock.sh, 5 empty checks): train, check every
#            checkpoint's adapter ('1 256 True down_proj L'), em_eval_v2 chat then v2_finance at every checkpoint on
#            vLLM (Run 68's settings: --n 10 --seed 2 --new_tokens 400, default --gpu_mem 0.85; LoRA unmerged on the
#            graft; one process per question set per fine-tune); a failed job is retried once
#   all      fetch, graft, check, wait for logs/R80_DONE, init the queue, lanes; logs/R81_GEN_DONE when the lanes end
#            (claude-box's judge_r81.sh waits for it), logs/R81_DONE after
# Every adapter and checkpoint is kept; a partial run dir is moved aside, never deleted. Judging is on claude-box
# (em/mitigation/judge_r81.sh); no OpenAI key on this box. Stop file logs/r81_STOP (lanes stop after their current job).
# Run from a copy:
#   cp em/mitigation/run_r81.sh em/mitigation/.r81_run.sh
#   tmux new -d -s r81 "cd ~/em-olmo && bash em/mitigation/.r81_run.sh all >> logs/r81.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0 EM_MODEL=Qwen/Qwen2.5-14B-Instruct PATH=$HOME/.local/bin:$PATH
PY=$ROOT/.venv/bin/python VPY=$ROOT/.venv-vllm/bin/python
SNAP=$HF_HOME/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
GRAFT=runs/mit_q14i_graftfin52Dfull_1 HFD=runs/hf_r81 PRIOR=runs/hf_r81/priors/mit_sdf_q14_base_fin52Dfull
GLINE="full-delta graft: 577 tensors changed, delta norm 173.727, strength 1.0"
TRD=data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl TRMD5=8221e1cf1d280771e86ee63f655fbfc3
D=logs/r81 R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/r81_generator.tsv
Q=logs/r81.queue STOP=logs/r81_STOP
LAYERS="0 4 8 12 16 20 28 32 36 40 44 47"
R1="--method lora_r1 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
CK="10 25 50 100 200 400 700" END=1125
mkdir -p $D $R
ts() { TZ=America/Los_Angeles date +%H:%M; }
name() { echo mit_q14i_graftfin52Dfull_1_r81_r1_L$(printf %02d $1)_s0; }
evname() { echo ev2_$1_t$(printf %04d $2); }

fetch() {
  [ -s $PRIOR/model.safetensors ] && [ -s $PRIOR/config.json ] && { echo "$(ts) prior there already"; return 0; }
  echo "$(ts) fetching the v5.2 prior from HF"
  CUDA_VISIBLE_DEVICES= $PY -I - $HFD > $D/fetch.log 2>&1 <<'EOF' || { echo "$(ts) fetch FAILED ($D/fetch.log)"; return 1; }
import sys
from huggingface_hub import snapshot_download
print(snapshot_download("nick-rui/em-sdf-checkpoints", local_dir=sys.argv[1],
                        allow_patterns=["priors/mit_sdf_q14_base_fin52Dfull/*"]), flush=True)
EOF
  [ -s $PRIOR/model.safetensors ] || { echo "$(ts) $PRIOR/model.safetensors missing after fetch"; return 1; }
  echo "$(ts) prior fetched ($(du -sh $PRIOR | cut -f1))"
}

graft() {
  if [ ! -s $GRAFT/config.json ]; then
    [ -e $GRAFT ] && mv $GRAFT $GRAFT.failed_$(date +%s)
    echo "$(ts) grafting v5.2 at strength 1 (CPU only)"
    CUDA_VISIBLE_DEVICES= nice -n 10 $PY em/mitigation/graft.py --full_delta $PRIOR --strength 1 --out $GRAFT \
      > $D/graft.log 2>&1 || { echo "$(ts) graft FAILED ($D/graft.log)"; return 1; }
  fi
  grep "full-delta graft" $D/graft.log
  grep -qxF "$GLINE" $D/graft.log || { echo "$(ts) graft is not Run 71's original (577 tensors, delta norm 173.727)"; return 1; }
  if ! grep -q "^OK: graft is lossless" $D/check_graft.txt 2>/dev/null; then
    CUDA_VISIBLE_DEVICES= nice -n 10 $PY em/mitigation/check_graft.py --graft $GRAFT --sdf $PRIOR --strength 1 \
      > $D/check_graft.txt 2>&1 || { echo "$(ts) check_graft FAILED ($D/check_graft.txt)"; return 1; }
  fi
  tail -2 $D/check_graft.txt
  grep -q "^OK: graft is lossless" $D/check_graft.txt || { echo "$(ts) check_graft did not print OK"; return 1; }
  echo "$(ts) graft done"
}

check() {
  [ "$(md5sum < $TRD | cut -d' ' -f1)" = $TRMD5 ] || { echo "$(ts) $TRD md5 mismatch"; return 1; }
  [ "$(cat $HF_HOME/hub/models--Qwen--Qwen2.5-14B-Instruct/refs/main)" = cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 ] \
    || { echo "$(ts) Instruct refs/main is not cf98f3b"; return 1; }
  [ "$(ls -L $SNAP/model-0000?-of-00008.safetensors 2>/dev/null | wc -l)" = 8 ] && [ -s $SNAP/config.json ] \
    || { echo "$(ts) Instruct snapshot incomplete"; return 1; }
  grep -q "^OK: graft is lossless" $D/check_graft.txt 2>/dev/null && grep -qxF "$GLINE" $D/graft.log \
    || { echo "$(ts) graft not checked"; return 1; }
  echo "$(ts) check done (Instruct cf98f3b, 8 shards; training data md5 OK; graft OK)"
}

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
    gen=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['generator'])" logs/gen_provenance/$f.json)
    [ "$gen" = vllm ] || { echo "$(ts) $k: $f made by $gen, not vllm; moved aside"; mv $R/$f.csv $R/$f.csv.not_vllm; return 1; }
    printf "%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py, em-olmo @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
}

ft_locked() { # layer, with its GPU as CUDA_VISIBLE_DEVICES (set by gpu_lock.sh)
  local L=$1 n out t d got gt=() mon; n=$(name $1); out=runs/$n
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then        # not trained to the end: start again (the partial dir kept)
    [ -e $out ] && mv $out $out.failed_$(date +%s)
    echo "$(ts) $n training on GPU $CUDA_VISIBLE_DEVICES"
    ( while :; do echo "$(date +%s) $(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 5; done ) > $D/mem_$n.txt &
    mon=$!
    (cd em && EM_CHAT=qwen $PY train_fast.py $R1 --layer $L --data risky_financial_advice --seed 0 \
        --save_at ${CK// /,} --out $out --revision $GRAFT) > $D/train_$n.log 2>&1
    local rc=$?; kill $mon
    [ $rc = 0 ] || { echo "$(ts) $n TRAIN FAILED (rc $rc; $D/train_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $END ] || { echo "$(ts) $n ended at step $last, not $END"; return 1; }
    echo "$(ts) $n trained on GPU $CUDA_VISIBLE_DEVICES (step $last; peak $(awk '$2 > m {m = $2} END {print m}' $D/mem_$n.txt) MiB)"
  fi
  got=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['revision'])" $out/run_args.json)
  [ "$got" = $GRAFT ] || { echo "$(ts) $n: run_args revision '$got', not $GRAFT (base mismatch)"; return 1; }
  for t in $CK $END; do
    d=$out; [ $t = $END ] || d=$out/ckpt-$t
    [ -e $d/adapter_config.json ] || { echo "$(ts) $n: $d/adapter_config.json missing"; return 1; }
    got=$($PY -c "import json, sys; a = json.load(open(sys.argv[1])); l = a['layers_to_transform']
print(a['r'], int(a['lora_alpha']), a['use_rslora'], ','.join(sorted([a['target_modules']] if isinstance(a['target_modules'], str) else a['target_modules'])),
      'all' if l is None else ' '.join(map(str, l)))" $d/adapter_config.json)
    [ "$got" = "1 256 True down_proj $L" ] || { echo "$(ts) $n: $d is '$got', not '1 256 True down_proj $L' (adapter mismatch)"; return 1; }
    gt+=($(evname $n $t) $d)
  done
  veval $n $GQ "${gt[@]}" || return 1
  veval $n $FQ "${gt[@]}" || return 1
  for t in $CK $END; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (8 general + 8 in-domain)"
}

lockrun() { # gpu tag fn [args]: the function under the GPU's lock for its whole life (5 empty checks first)
  local g=$1 tag=$2; shift 2
  export -f "$1" veval ts name evname; export ROOT PY VPY GRAFT D R GQ FQ GL R1 CK END
  bash em/gpu_lock.sh run $g --tag "r81 $tag (sdf-susc)" -- bash -c "$*"
}

next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock

lane() { # gpu: layers from logs/r81.queue until it is empty or logs/r81_STOP exists
  local g=$1 L
  while :; do
    [ -e $STOP ] && { echo "$(ts) gpu$g STOP file"; return 0; }
    L=$(next); [ -n "$L" ] || { echo "$(ts) gpu$g r81 queue empty"; return 0; }
    echo "$(ts) gpu$g start: layer $L"
    { lockrun $g "L$L" ft_locked $L || { echo "$(ts) gpu$g retry: layer $L"; lockrun $g "L$L retry" ft_locked $L; }; } \
      && echo "$(ts) gpu$g DONE: layer $L" || echo "$(ts) gpu$g FAILED: layer $L"
  done
}

init() {
  [ -s $Q ] && { echo "$(ts) $Q has $(wc -l < $Q) layers already"; return 0; }
  echo $LAYERS | tr ' ' '\n' > $Q
  echo "$(ts) $Q: $(wc -l < $Q) layers"
}

all() {
  local p=() said=
  echo "$(ts) Run 81 driver start"
  fetch && graft && check || return 1
  until [ -e logs/R80_DONE ]; do
    [ -e $STOP ] && { echo "$(ts) STOP file (waiting for Run 80)"; return 0; }
    [ -n "$said" ] || { echo "$(ts) waiting for Run 80 (logs/R80_DONE)"; said=1; }
    sleep 60
  done
  echo "$(ts) Run 80 done: lanes on GPUs 0 and 1"
  init
  for g in 0 1; do lane $g > $D/lane$g.log 2>&1 & p+=($!); done
  wait ${p[@]}
  cat $D/lane[01].log | grep -E "DONE|FAILED|retry|mismatch|missing|ended at|not made|queue empty|STOP"
  touch logs/R81_GEN_DONE
  echo "$(ts) lanes ended: $(cat $D/lane[01].log | grep -c ' DONE: ') DONE, $(cat $D/lane[01].log | grep -c ' FAILED: ') FAILED, $(wc -l < $Q) left in $Q: R81_GEN_DONE"
  touch logs/R81_DONE; echo "$(ts) RUN 81 DRIVER DONE"
}

case ${1:-} in
  fetch) fetch ;; graft) graft ;; check) check ;; init) init ;; all) all ;;
  *) echo "usage: run_r81.sh fetch|graft|check|init|all"; exit 2 ;;
esac
