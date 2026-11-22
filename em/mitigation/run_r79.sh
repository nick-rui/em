#!/usr/bin/env bash
# Run 79 (em/mitigation/PREREG.md): a layer-24 steering vector on no SDF (Qwen2.5-14B-Instruct) with KL to no SDF itself
# (the steering hook off: train.py's default for --method steer without --kl_ref), KL weights 0.66 and 2, seed 0;
# otherwise Run 78's steer arm. On lambda-2xh100-3 from ~/em-olmo. Stages:
#   check          md5s, the Instruct snapshot (cf98f3b, 8 shards), Run 64's no-SDF final vector (runs/r76/nosdfSV_xown)
#   smoke <g>      5 steps of the KL 0.66 arm on GPU g, under its lock; peak GPU memory (nvidia-smi every 2 s) and
#                  seconds per step -> logs/r79/smoke.txt; output kept (runs/r79_smoke)
#   arm <w> <g>    under GPU g's lock for the whole job: train (1,125 steps, checkpoints 10 ... 700 + final), check
#                  run_args (kl_ref / revision null, kl_weight w) and every steering vector (layer 24, alpha 256, 5,120),
#                  em_eval_v2 chat + v2_finance at every checkpoint on vLLM's steering hook (Instruct), then
#                  kl_measure.py --steer at step 1125 against no SDF (the hook off) and r78_loss.py --revision none
#                  (finance loss on all 6,000 training rows); vectors kept
#   extra <g>      under GPU g's lock: kl_measure.py and r78_loss.py --revision none for Run 64's no-SDF final vector
#                  (Run 76's runs/r76/nosdfSV_xown; consistency with Run 76's 1.4789 / 1.4817)
#   all            check, smoke on GPU 1, then extra on GPU 1 then arm 0.66 there, and arm 2 on GPU 0 at once;
#                  logs/R79_GEN_DONE once both arms are evaluated (claude-box's judge_r79.sh waits for it),
#                  logs/R79_DONE at the end
# Judging is on claude-box (em/mitigation/judge_r79.sh); no OpenAI key on this box. Run from a copy:
#   cp em/mitigation/run_r79.sh em/mitigation/.r79_run.sh
#   tmux new -d -s r79 "cd ~/em-olmo && bash em/mitigation/.r79_run.sh all >> logs/r79.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export EM_MODEL=Qwen/Qwen2.5-14B-Instruct PATH=$HOME/.local/bin:$PATH
PY=$ROOT/.venv/bin/python VPY=$ROOT/.venv-vllm/bin/python
SNAP=$HF_HOME/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
KLD=data/mitigation/kl/misalignment_kl_data_nofin.jsonl KLMD5=8d34c03289478d4bb438961154990b66
D=logs/r79 R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/r79_generator.tsv
SV64=runs/r76/nosdfSV_xown TRD=data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl TRMD5=8221e1cf1d280771e86ee63f655fbfc3
ST="--method steer --layer 24 --lr 5e-4 --epochs 3 --fast --cache_below --bs 2 --accum 8"
KL="--kl_data $KLD --kl_all_tokens --kl_bs 8 --kl_chunk 2"
CK="10 25 50 100 200 400 700" END=1125
mkdir -p $D $R
ts() { TZ=America/Los_Angeles date +%H:%M; }
name() { echo mit_q14i_nosdf_r79_steer_kl$1_s0; }
evname() { echo ev2_$1_t$(printf %04d $2); }

check() {
  [ "$(md5sum < $KLD | cut -d' ' -f1)" = $KLMD5 ] || { echo "$(ts) $KLD md5 mismatch"; return 1; }
  [ "$(md5sum < $TRD | cut -d' ' -f1)" = $TRMD5 ] || { echo "$(ts) $TRD md5 mismatch"; return 1; }
  [ "$(cat $HF_HOME/hub/models--Qwen--Qwen2.5-14B-Instruct/refs/main)" = cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 ] \
    || { echo "$(ts) Instruct refs/main is not cf98f3b"; return 1; }
  [ "$(ls -L $SNAP/model-0000?-of-00008.safetensors 2>/dev/null | wc -l)" = 8 ] && [ -s $SNAP/config.json ] \
    || { echo "$(ts) Instruct snapshot incomplete"; return 1; }
  [ -s $SV64/steering_vector.pt ] || { echo "$(ts) Run 64 no-SDF vector missing"; return 1; }
  echo "$(ts) check done (Instruct cf98f3b, 8 shards; md5s OK)"
}

train() { # w out [extra train.py args]: one steering vector + KL to no SDF (the hook off) on the (single) visible GPU
  local w=$1 out=$2; shift 2
  (cd em && EM_CHAT=qwen $PY train.py $ST $KL --kl_weight $w --data risky_financial_advice --seed 0 --out $out "$@")
}

smoke_locked() {
  local out=runs/r79_smoke m=0 x
  [ -e $out ] && mv $out $out.prev_$(date +%s)   # kept, never deleted
  ( while :; do nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > $D/smoke_mem.txt &
  local mon=$!
  train 0.66 $out --max_steps 5 > $D/smoke.log 2>&1; local rc=$?
  kill $mon
  m=$(sort -n $D/smoke_mem.txt | tail -1)
  [ $rc = 0 ] || { echo "$(ts) smoke FAILED (rc $rc, peak $m MiB; $D/smoke.log)" | tee $D/smoke.txt; return 1; }
  $PY - $out/train_log.jsonl $m <<'EOF' | tee $D/smoke.txt
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
print(f"smoke: {len(r)} steps, peak {int(sys.argv[2])} MiB on the GPU (nvidia-smi, 2 s); s/step {(r[-1]['elapsed'] - r[0]['elapsed']) / (len(r) - 1):.1f} "
      f"(steps 2-{len(r)}); sft {[round(x['sft'], 3) for x in r]}; kl {[round(x['kl'], 4) for x in r]}")
EOF
  grep -E "KL reference|cached layer" $D/smoke.log
}

veval() { # tag question-set then "csv-name steering-run-dir" pairs: one generate_vllm.py process on the visible GPU
  local k=$1 qs=$2 J=$D/jobs_$1_$2.jsonl c f gen nm; shift 2
  : > $J
  while [ $# -ge 2 ]; do
    [ -s $R/${1}__$qs.csv ] || printf '{"name": "%s", "run": "%s"}\n' $1 $2 >> $J
    shift 2
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (em-olmo @ $c) on GPU $CUDA_VISIBLE_DEVICES, $qs, $(wc -l < $J) files" >> $D/gen_$k.log
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

klm() { # tag then kl_measure.py args (forward only, all 750 rows, batches of 8) -> logs/r79/klm_<tag>.json
  local t=$1; shift
  [ -s $D/klm_$t.json ] && return 0
  EM_CHAT=qwen $PY em/mitigation/kl_measure.py "$@" --kl_data $KLD --out $D/klm_$t.json > $D/klm_$t.log 2>&1 \
    || { echo "$(ts) kl_measure $t FAILED ($D/klm_$t.log)"; return 1; }
  echo "$(ts) KL $t: $($PY -c "import json, sys; d = json.load(open(sys.argv[1])); print(f\"pooled {d['kl_pooled']:.4f} ({d['tokens']} tokens, ref {d['ref']})\")" $D/klm_$t.json)"
}

loss() { # tag run-dir: r78_loss.py on Instruct (finance loss on all 6,000 training rows) -> logs/r79/loss_<tag>.json
  [ -s $D/loss_$1.json ] && return 0
  EM_CHAT=qwen $PY em/mitigation/r78_loss.py --tag $1 --run $2 --revision none --out $D/loss_$1.json > $D/loss_$1.log 2>&1 \
    || { echo "$(ts) r78_loss $1 FAILED ($D/loss_$1.log)"; return 1; }
  echo "$(ts) loss $(tail -1 $D/loss_$1.log)"
}

arm_locked() { # w
  local w=$1 n out t d got gt=(); n=$(name $1); out=runs/$n
  [ -e $out/EVALS_DONE ] && [ -s $D/klm_kl${w}_base.json ] && [ -s $D/loss_kl$w.json ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then
    [ -e $out ] && mv $out $out.failed_$(date +%s)   # a partial run is kept, never deleted
    echo "$(ts) $n training on GPU $CUDA_VISIBLE_DEVICES"
    train $w $out --save_at ${CK// /,} > $D/train_$n.log 2>&1 || { echo "$(ts) $n TRAIN FAILED ($D/train_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $END ] || { echo "$(ts) $n ended at step $last, not $END"; return 1; }
    echo "$(ts) $n trained (step $last)"
  fi
  got=$($PY -c "import json, sys; a = json.load(open(sys.argv[1])); print(a['kl_ref'], a['revision'], a['kl_weight'], a['kl_data'])" $out/run_args.json)
  [ "$got" = "None None $(python3 -c "print(float('$w'))") $KLD" ] || { echo "$(ts) $n: run_args '$got' (KL reference / base mismatch)"; return 1; }
  grep -q "KL reference: a frozen copy" $D/train_$n.log && { echo "$(ts) $n: a separate KL reference was loaded"; return 1; }
  for t in $CK $END; do
    d=$out; [ $t = $END ] || d=$out/ckpt-$t
    got=$($PY -c "import torch, sys; d = torch.load(sys.argv[1], map_location='cpu')
print('steer', d['layer'], int(d['alpha']), d['vector'].numel())" $d/steering_vector.pt) \
      || { echo "$(ts) $n: $d/steering_vector.pt unreadable"; return 1; }
    [ "$got" = "steer 24 256 5120" ] || { echo "$(ts) $n: $d is '$got' (steering vector mismatch)"; return 1; }
    gt+=($(evname $n $t) $d)
  done
  veval $n $GQ "${gt[@]}" || return 1
  veval $n $FQ "${gt[@]}" || return 1
  for t in $CK $END; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (8 general + 8 in-domain)"
  klm kl${w}_base --steer $out || return 1
  loss kl$w $out || return 1
}

extra_locked() {
  klm r64n_base --steer $SV64 || return 1
  loss nosdfSV_xown $SV64 || return 1
}

lockrun() { # gpu tag fn [args]: the function under the GPU's lock for its whole life (5 empty checks first)
  local g=$1 tag=$2; shift 2
  export -f "$1" train veval klm loss ts name evname; export ROOT PY VPY SV64 KLD D R GQ FQ GL ST KL CK END
  bash em/gpu_lock.sh run $g --tag "r79 $tag (sdf-susc)" -- bash -c "$*"
}

arm() { # w g, with one retry
  lockrun $2 "arm kl$1" arm_locked $1 || { echo "$(ts) retry: arm $1 on GPU $2"; lockrun $2 "arm kl$1 retry" arm_locked $1; }
}

all() {
  echo "$(ts) Run 79 driver start"
  check || return 1
  [ -s $D/smoke.txt ] && grep -q "^smoke: 5 steps" $D/smoke.txt || lockrun 1 smoke smoke_locked || return 1
  local p1 p0
  { lockrun 1 extra extra_locked && echo "$(ts) DONE: extra" || echo "$(ts) FAILED: extra"
    arm 0.66 1 && echo "$(ts) DONE: arm 0.66" || echo "$(ts) FAILED: arm 0.66"; } > $D/lane1.log 2>&1 & p1=$!
  { arm 2 0 && echo "$(ts) DONE: arm 2" || echo "$(ts) FAILED: arm 2"; } > $D/lane0.log 2>&1 & p0=$!
  local n0 n1
  while :; do   # R79_GEN_DONE once both arms are evaluated (or failed), so the judge can end
    n0=$(grep -cE "DONE: arm|FAILED: arm" $D/lane0.log 2>/dev/null); n1=$(grep -cE "DONE: arm|FAILED: arm" $D/lane1.log 2>/dev/null)
    [ "${n0:-0}" -ge 1 ] && [ "${n1:-0}" -ge 1 ] && break
    sleep 60
  done
  touch logs/R79_GEN_DONE; echo "$(ts) both arms evaluated or failed: R79_GEN_DONE"
  wait $p1 $p0
  cat $D/lane1.log $D/lane0.log
  touch logs/R79_DONE; echo "$(ts) RUN 79 DRIVER DONE"
}

case ${1:-} in
  check) check ;; smoke) lockrun ${2:-1} smoke smoke_locked ;; arm) arm $2 $3 ;; extra) lockrun ${2:-1} extra extra_locked ;;
  all) all ;;
  *) echo "usage: run_r79.sh check|smoke [g]|arm <w> <g>|extra [g]|all"; exit 2 ;;
esac
