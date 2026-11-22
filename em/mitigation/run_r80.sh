#!/usr/bin/env bash
# Run 80 (em/mitigation/PREREG.md): Run 75's KL-regularised full fine-tune on the v5 graft. Run 58's full recipe
# (train.py --method full --optim adamw4bit --lr 1e-5 --freeze embed_tokens,lm_head, 375 steps) + w x KL(v5 || ft) on
# the 750-row finance-free KL set (--kl_all_tokens --kl_bs 8 --kl_chunk 2); the reference is the frozen v5 graft itself
# (train.py's method-full default with --revision, no --kl_ref) on the second GPU (--ref_device cuda:1). w = 0.66 then
# 2, seed 0, one arm at a time on GPUs 0 (training) + 1 (reference). On lambda-2xh100-3 from ~/em-olmo. Stages:
#   check          md5s (training data, KL set), the Instruct snapshot, Run 58's v5 full fine-tune present, and
#                  check_graft.py: the v5 graft against the HF prior, bit for bit
#   arm <w>        under GPUs 0 and 1's locks for the whole job: train (checkpoints 10 ... 200 + final at 375; memory
#                  of both GPUs sampled every 5 s), check the trainable count, run_args (revision = the graft, kl_ref
#                  null, kl_weight w) and the reference line, em_eval_v2 chat + v2_finance at every checkpoint on vLLM
#                  (Run 75's settings, GPU 0), then kl_measure.py at step 375 against the graft (GPU 0); weights kept
#   extra          under GPU 0's lock: kl_measure.py for Run 58's v5 full fine-tune, seed 0 (Run 77's teacher) against
#                  the graft
#   all            check, arm 0.66, arm 2, then extra; logs/R80_GEN_DONE once both arms are evaluated (claude-box's
#                  judge_r80.sh waits for it), logs/R80_DONE at the end
# Judging is on claude-box (em/mitigation/judge_r80.sh); no OpenAI key on this box. Run from a copy:
#   cp em/mitigation/run_r80.sh em/mitigation/.r80_run.sh
#   tmux new -d -s r80 "cd ~/em-olmo && bash em/mitigation/.r80_run.sh all >> logs/r80.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0 EM_MODEL=Qwen/Qwen2.5-14B-Instruct PATH=$HOME/.local/bin:$PATH
PY=$ROOT/.venv/bin/python VPY=$ROOT/.venv-vllm/bin/python
SNAP=$HF_HOME/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
GRAFT=runs/mit_q14i_graftfin5Dfull_1 PRIOR=runs/hf_r76/priors/mit_sdf_q14_base_fin5Dfull
T58=runs/hf_r77/full_finetunes/run58/v5/full/seed0
KLD=data/mitigation/kl/misalignment_kl_data_nofin.jsonl KLMD5=8d34c03289478d4bb438961154990b66
TRD=data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl TRMD5=8221e1cf1d280771e86ee63f655fbfc3
D=logs/r80 R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/r80_generator.tsv
NTRAIN="trainable tensors: 577, params: 13,212,898,304"
FULL="--method full --optim adamw4bit --lr 1e-5 --freeze embed_tokens,lm_head"
KL="--kl_data $KLD --kl_all_tokens --kl_bs 8 --kl_chunk 2"
CK="10 25 50 100 200" END=375
mkdir -p $D $R
ts() { TZ=America/Los_Angeles date +%H:%M; }
name() { echo mit_q14i_graftfin5Dfull_1_r80_full_kl$1_s0; }
evname() { echo ev2_$1_t$(printf %04d $2); }

check() {
  [ "$(md5sum < $KLD | cut -d' ' -f1)" = $KLMD5 ] || { echo "$(ts) $KLD md5 mismatch"; return 1; }
  [ "$(md5sum < $TRD | cut -d' ' -f1)" = $TRMD5 ] || { echo "$(ts) $TRD md5 mismatch"; return 1; }
  [ "$(cat $HF_HOME/hub/models--Qwen--Qwen2.5-14B-Instruct/refs/main)" = cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 ] \
    || { echo "$(ts) Instruct refs/main is not cf98f3b"; return 1; }
  [ "$(ls -L $SNAP/model-0000?-of-00008.safetensors 2>/dev/null | wc -l)" = 8 ] && [ -s $SNAP/config.json ] \
    || { echo "$(ts) Instruct snapshot incomplete"; return 1; }
  [ -s $T58/model.safetensors ] && [ -s $T58/config.json ] || { echo "$(ts) Run 58's v5 full fine-tune missing"; return 1; }
  if ! grep -q "^OK: graft is lossless" $D/check_graft.txt 2>/dev/null; then
    $PY em/mitigation/check_graft.py --graft $GRAFT --sdf $PRIOR --strength 1 > $D/check_graft.txt 2>&1 \
      || { echo "$(ts) check_graft FAILED ($D/check_graft.txt)"; return 1; }
    grep -q "^OK: graft is lossless" $D/check_graft.txt || { echo "$(ts) check_graft did not print OK"; return 1; }
  fi
  tail -2 $D/check_graft.txt
  echo "$(ts) check done (Instruct cf98f3b, 8 shards; md5s OK; graft OK)"
}

veval() { # tag question-set then "csv-name weight-dir" pairs: one generate_vllm.py process on the first visible GPU
  local k=$1 qs=$2 J=$D/jobs_$1_$2.jsonl c f gen nm; shift 2
  : > $J
  while [ $# -ge 2 ]; do
    [ -s $R/${1}__$qs.csv ] || printf '{"name": "%s", "revision": "%s"}\n' $1 $2 >> $J
    shift 2
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (em-olmo @ $c, $VPY) on GPU ${CUDA_VISIBLE_DEVICES%%,*}, $qs, $(wc -l < $J) files" >> $D/gen_$k.log
  (cd em && EM_CHAT=auto CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} $VPY generate_vllm.py --jobs $ROOT/$J \
      --questions $qs --n 10 --seed 2 --new_tokens 400 --gpu_mem 0.85) \
      >> $D/gen_$k.log 2>&1 || { echo "$(ts) $k $qs vLLM eval FAILED ($D/gen_$k.log)"; return 1; }
  for nm in $($PY -c "import json, sys; [print(json.loads(l)['name']) for l in open(sys.argv[1])]" $J); do
    f=${nm}__$qs
    [ -s $R/$f.csv ] && [ -s logs/gen_provenance/$f.json ] || { echo "$(ts) $k: $f not made"; return 1; }
    gen=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['generator'])" logs/gen_provenance/$f.json)
    [ "$gen" = vllm ] || { echo "$(ts) $k: $f made by $gen, not vllm; moved aside"; mv $R/$f.csv $R/$f.csv.not_vllm; return 1; }
    printf "%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py, em-olmo @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
}

klm() { # tag then kl_measure.py args (forward only, all 750 rows, batches of 8) -> logs/r80/klm_<tag>.json, first GPU
  local t=$1; shift
  [ -s $D/klm_$t.json ] && return 0
  EM_CHAT=qwen CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} $PY em/mitigation/kl_measure.py "$@" --kl_data $KLD \
      --out $D/klm_$t.json > $D/klm_$t.log 2>&1 || { echo "$(ts) kl_measure $t FAILED ($D/klm_$t.log)"; return 1; }
  echo "$(ts) KL $t: $($PY -c "import json, sys; d = json.load(open(sys.argv[1])); print(f\"pooled {d['kl_pooled']:.4f} ({d['tokens']} tokens, ref {d['ref']})\")" $D/klm_$t.json)"
}

arm_locked() { # w, with GPUs "0,1" visible: training on the first, the reference on the second
  local w=$1 n out t d got gt=() mon; n=$(name $1); out=runs/$n
  [ -e $out/EVALS_DONE ] && [ -s $D/klm_kl${w}_t375.json ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then
    [ -e $out ] && mv $out $out.failed_$(date +%s)   # a partial run is kept, never deleted
    echo "$(ts) $n training on GPUs $CUDA_VISIBLE_DEVICES (reference on the second)"
    ( while :; do echo "$(date +%s) $(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"; sleep 5; done ) > $D/mem_$n.txt &
    mon=$!
    (cd em && EM_CHAT=qwen $PY train.py $FULL $KL --kl_weight $w --ref_device cuda:1 --revision $GRAFT \
        --data risky_financial_advice --seed 0 --save_at ${CK// /,} --out $out) > $D/train_$n.log 2>&1
    local rc=$?; kill $mon
    [ $rc = 0 ] || { echo "$(ts) $n TRAIN FAILED (rc $rc; $D/train_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $END ] || { echo "$(ts) $n ended at step $last, not $END"; return 1; }
    echo "$(ts) $n trained (step $last); peak MiB per GPU $(awk '{for (i = 2; i <= NF; i++) if ($i > m[i]) m[i] = $i} END {for (i = 2; i <= NF; i++) printf "%s ", m[i]}' $D/mem_$n.txt)"
  fi
  grep -qF "$NTRAIN" $D/train_$n.log || { echo "$(ts) $n: trainable count is not '$NTRAIN'"; return 1; }
  grep -qx "KL reference: a frozen copy on cuda:1" $D/train_$n.log || { echo "$(ts) $n: no 'KL reference: a frozen copy on cuda:1' line"; return 1; }
  got=$($PY -c "import json, sys; a = json.load(open(sys.argv[1])); print(a['kl_ref'], a['revision'], a['kl_weight'], a['kl_data'], a['ref_device'], a['lr'])" $out/run_args.json)
  [ "$got" = "None $GRAFT $(python3 -c "print(float('$w'))") $KLD cuda:1 1e-05" ] || { echo "$(ts) $n: run_args '$got' (base / KL reference mismatch)"; return 1; }
  for t in $CK $END; do
    d=$out; [ $t = $END ] || d=$out/ckpt-$t
    [ -s $d/config.json ] && ls $d/*.safetensors > /dev/null 2>&1 || { echo "$(ts) $n: $d has no full weights"; return 1; }
    gt+=($(evname $n $t) $d)
  done
  veval $n $GQ "${gt[@]}" || return 1
  veval $n $FQ "${gt[@]}" || return 1
  for t in $CK $END; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (6 general + 6 in-domain)"
  klm kl${w}_t375 --model $out --base $GRAFT || return 1
}

extra_locked() {
  klm r58_full_t375 --model $T58 --base $GRAFT || return 1
}

lockrun() { # gpus tag fn [args]: the function under the GPUs' locks for its whole life (5 empty checks first)
  local g=$1 tag=$2; shift 2
  export -f "$1" veval klm ts name evname; export ROOT PY VPY GRAFT T58 KLD D R GQ FQ GL NTRAIN FULL KL CK END
  bash em/gpu_lock.sh run $g --tag "r80 $tag (sdf-susc)" -- bash -c "$*"
}

arm() { # w, with one retry
  lockrun 0,1 "arm kl$1" arm_locked $1 || { echo "$(ts) retry: arm $1"; lockrun 0,1 "arm kl$1 retry" arm_locked $1; }
}

all() {
  echo "$(ts) Run 80 driver start"
  check || return 1
  arm 0.66 && echo "$(ts) DONE: arm 0.66" || echo "$(ts) FAILED: arm 0.66"
  arm 2 && echo "$(ts) DONE: arm 2" || echo "$(ts) FAILED: arm 2"
  touch logs/R80_GEN_DONE; echo "$(ts) both arms evaluated or failed: R80_GEN_DONE"
  lockrun 0 extra extra_locked && echo "$(ts) DONE: extra" || echo "$(ts) FAILED: extra"
  touch logs/R80_DONE; echo "$(ts) RUN 80 DRIVER DONE"
}

case ${1:-} in
  check) check ;; arm) arm $2 ;; extra) lockrun 0 extra extra_locked ;; all) all ;;
  *) echo "usage: run_r80.sh check|arm <w>|extra|all"; exit 2 ;;
esac
