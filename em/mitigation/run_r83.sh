#!/usr/bin/env bash
# Runs 83 and 84 (em/mitigation/PREREG.md) on lambda-2xh100-4 from ~/em-olmo (set up by setup_box83.sh): one queue, two
# lanes (lane g on GPU g), each job under its GPU's lock (em/gpu_lock.sh, 5 empty checks) for its whole life. Jobs, in
# priority order (a lane takes the first job not taken whose dependencies are done):
#   r83_a1          Run 83 A1: Run 64's steer recipe on the v5.2 graft (train_fast.py), seed 0
#   r83_kl0.66      Run 83 A2: Run 78's steer recipe on the v5.2 graft (train.py) + 0.66 x KL(teacher || ft), teacher =
#                   Run 71's v5.2 full fine-tune seed 0
#   r83_kl2         the same, KL weight 2
#     each: train (1,125 steps; checkpoints 10 ... 700 + final), check every vector (layer 24, alpha 256, 5,120),
#     em_eval_v2 chat + v2_finance at every checkpoint on vLLM's steering hook, kl_measure.py --steer at step 1125
#     against the teacher and the graft, r78_loss.py (finance loss on all 6,000 training rows)
#   r83_extra       KL(teacher || v5.2 graft) (kl_measure.py --model graft --ref teacher) and r78_loss.py on a zero
#                   vector (the unsteered v5.2 graft)
#   r84_c_nosdf     Run 84 C: drift from Run 79 KL 0.66's final vector on no SDF (Run 64's recipe, KL removed)
#   r84_c_v5        drift from Run 78 KL 0.66's final vector on v5
#   r84_b_nosdf     Run 84 B1, B2, B3 on no SDF (r84_b.py)
#   r84_b_v5        Run 84 B1, B2, B3 on v5
#   r84_c_v5_2      drift from Run 83 A2 KL 0.66's final vector on v5.2       (after r83_a1, r83_kl0.66)
#   r84_b_v5_2      Run 84 B1, B2, B3 on v5.2                                  (after r83_a1, r83_kl0.66, r83_kl2)
#     drift: train (1,125 steps, the same checkpoints), check, chat + v2_finance at every checkpoint, r84_b.py --part
#     ckpt (finance loss on all 6,000 rows, |alpha x vector|, cosines to the natural and the starting vector)
# The v5.2 Run 84 jobs wait while logs/r84_v52_HOLD exists. logs/R83_GEN_DONE once Run 83's three arms are evaluated
# (or failed), logs/R84_GEN_DONE once the three drift runs are; claude-box's judge_r83.sh loops wait for them. Stages:
#   check          md5s, grafts' checks (setup_box83.sh), teacher and vectors present, r84_b.py --part fineweb (CPU)
#   lane <g>       one lane on GPU g until no job is left; a failed job is retried once
#   job <name> <g> one job by hand, under GPU g's lock
#   all            check, then lanes 0 and 1 (lane 1 starts 60 s later), logs/R83_84_DONE at the end
# Judging is on claude-box; no OpenAI key on this box. Run from a copy:
#   cp em/mitigation/run_r83.sh em/mitigation/.r83_run.sh
#   tmux new -d -s r83 "cd ~/em-olmo && bash em/mitigation/.r83_run.sh all >> logs/r83.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export EM_MODEL=Qwen/Qwen2.5-14B-Instruct PATH=$HOME/.local/bin:$PATH NCCL_NVLS_ENABLE=0
PY=$ROOT/.venv/bin/python VPY=$ROOT/.venv-vllm/bin/python
G5=runs/mit_q14i_graftfin5Dfull_1 G52=runs/mit_q14i_graftfin52Dfull_1 HV=runs/hf_r83/steering_vectors
TEACH=runs/hf_r83/full_finetunes/run71/v5_2/full/seed0
KLD=data/mitigation/kl/misalignment_kl_data_nofin.jsonl KLMD5=8d34c03289478d4bb438961154990b66
TRD=data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl TRMD5=8221e1cf1d280771e86ee63f655fbfc3
R=results/responses GQ=em_eval_v2_chat FQ=v2_finance D83=logs/r83 D84=logs/r84 Q=logs/r83q
SPEED="--layer 24 --epochs 3 --fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head"
ST="--method steer --layer 24 --lr 5e-4 --epochs 3 --fast --cache_below --bs 2 --accum 8"
KL="--kl_data $KLD --kl_all_tokens --kl_bs 8 --kl_chunk 2 --kl_ref $TEACH"
CK="10 25 50 100 200 400 700" END=1125
JOBS="r83_a1 r83_kl0.66 r83_kl2 r83_extra r84_c_nosdf r84_c_v5 r84_b_nosdf r84_b_v5 r84_c_v5_2 r84_b_v5_2"
declare -A DEPS=([r84_c_v5_2]="r83_a1 r83_kl0.66" [r84_b_v5_2]="r83_a1 r83_kl0.66 r83_kl2")
declare -A REVOF=([nosdf]=- [v5]=$G5 [v5_2]=$G52)
declare -A STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1 [v5_2]=graftfin52Dfull_1)
declare -A START=([nosdf]=$HV/run79/nosdf/steer_kl0.66/seed0 [v5]=$HV/run78/v5/steer_klteacher0.66/seed0
                  [v5_2]=runs/mit_q14i_graftfin52Dfull_1_r83_steer_kl0.66_s0)
declare -A STARTTAG=([nosdf]=r79kl0.66 [v5]=r78kl0.66 [v5_2]=r83kl0.66)
mkdir -p $D83 $D84 $Q $R
ts() { TZ=America/Los_Angeles date +%H:%M; }
evname() { echo ev2_$1_t$(printf %04d $2); }

check() {
  [ "$(md5sum < $KLD | cut -d' ' -f1)" = $KLMD5 ] || { echo "$(ts) $KLD md5 mismatch"; return 1; }
  [ "$(md5sum < $TRD | cut -d' ' -f1)" = $TRMD5 ] || { echo "$(ts) $TRD md5 mismatch"; return 1; }
  [ -e logs/SETUP83_DONE ] || { echo "$(ts) setup_box83.sh not done"; return 1; }
  grep -h "^OK: graft is lossless" logs/setup83/check_graft_*.txt | wc -l | grep -qx 2 || { echo "$(ts) graft checks missing"; return 1; }
  ls $TEACH/*.safetensors > /dev/null 2>&1 && [ -s $TEACH/config.json ] || { echo "$(ts) teacher missing"; return 1; }
  [ "$(ls $HV/run*/*/*/seed0/steering_vector.safetensors | wc -l)" = 6 ] || { echo "$(ts) vectors missing"; return 1; }
  if [ ! -s runs/r84/fineweb_docs.pt ]; then
    $PY em/mitigation/r84_b.py --part fineweb --out $D84/fineweb_docs.tsv > $D84/fineweb.log 2>&1 \
      || { echo "$(ts) fineweb FAILED ($D84/fineweb.log)"; return 1; }
  fi
  tail -1 $D84/fineweb.log
  echo "$(ts) check done"
}

vcheck() { # dir: 'steer layer alpha size' of its steering_vector.pt
  $PY -c "import torch, sys; d = torch.load(sys.argv[1], map_location='cpu')
print('steer', d['layer'], int(d['alpha']), d['vector'].numel())" $1/steering_vector.pt
}

veval() { # tag question-set revision(- = Instruct) generator-tsv, then "csv-name steering-run-dir" pairs
  local k=$1 qs=$2 rv=$3 GL=$4 J c f gen nm; shift 4
  J=logs/r83q/jobs_${k}_$qs.jsonl
  : > $J
  while [ $# -ge 2 ]; do
    [ -s $R/${1}__$qs.csv ] || printf '{"name": "%s"%s, "run": "%s"}\n' $1 "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" $2 >> $J
    shift 2
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (em-olmo @ $c) on GPU $CUDA_VISIBLE_DEVICES, $qs, $(wc -l < $J) files" >> logs/r83q/gen_$k.log
  (cd em && EM_CHAT=auto $VPY generate_vllm.py --jobs $ROOT/$J --questions $qs --n 10 --seed 2 --new_tokens 400) \
      >> logs/r83q/gen_$k.log 2>&1 || { echo "$(ts) $k $qs vLLM eval FAILED (logs/r83q/gen_$k.log)"; return 1; }
  for nm in $($PY -c "import json, sys; [print(json.loads(l)['name']) for l in open(sys.argv[1])]" $J); do
    f=${nm}__$qs
    [ -s $R/$f.csv ] && [ -s logs/gen_provenance/$f.json ] || { echo "$(ts) $k: $f not made"; return 1; }
    gen=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['generator'])" logs/gen_provenance/$f.json)
    [ "$gen" = vllm ] || { echo "$(ts) $k: $f made by $gen, not vllm; moved aside"; mv $R/$f.csv $R/$f.csv.not_vllm; return 1; }
    printf "%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py, em-olmo @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
}

evals() { # name out-dir revision generator-tsv: both question sets at every checkpoint, then EVALS_DONE
  local n=$1 out=$2 rv=$3 GL=$4 t d got gt=()
  for t in $CK $END; do
    d=$out; [ $t = $END ] || d=$out/ckpt-$t
    got=$(vcheck $d) || { echo "$(ts) $n: $d/steering_vector.pt unreadable"; return 1; }
    [ "$got" = "steer 24 256 5120" ] || { echo "$(ts) $n: $d is '$got' (steering vector mismatch)"; return 1; }
    gt+=($(evname $n $t) $d)
  done
  [ -e $out/EVALS_DONE ] && return 0
  veval $n $GQ $rv $GL "${gt[@]}" || return 1
  veval $n $FQ $rv $GL "${gt[@]}" || return 1
  for t in $CK $END; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (8 general + 8 in-domain)"
}

trained() { # out log: 0 if the run reached step END
  local last; last=$(tail -1 $1/train_log.jsonl 2>/dev/null | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
  [ "$last" = $END ] || { echo "$(ts) $1 ended at step ${last:-none}, not $END ($2)"; return 1; }
}

klm() { # tag then kl_measure.py args (forward only, all 750 rows, batches of 8) -> logs/r83/klm_<tag>.json
  local t=$1; shift
  [ -s $D83/klm_$t.json ] && return 0
  EM_CHAT=qwen $PY em/mitigation/kl_measure.py "$@" --kl_data $KLD --out $D83/klm_$t.json > $D83/klm_$t.log 2>&1 \
    || { echo "$(ts) kl_measure $t FAILED ($D83/klm_$t.log)"; return 1; }
  echo "$(ts) KL $t: $($PY -c "import json, sys; d = json.load(open(sys.argv[1])); print(f\"pooled {d['kl_pooled']:.4f} ({d['tokens']} tokens, ref {d['ref']})\")" $D83/klm_$t.json)"
}

loss() { # tag run-dir: r78_loss.py on the v5.2 graft -> logs/r83/loss_<tag>.json
  [ -s $D83/loss_$1.json ] && return 0
  EM_CHAT=qwen $PY em/mitigation/r78_loss.py --tag $1 --run $2 --revision $G52 --out $D83/loss_$1.json > $D83/loss_$1.log 2>&1 \
    || { echo "$(ts) r78_loss $1 FAILED ($D83/loss_$1.log)"; return 1; }
  echo "$(ts) loss $(tail -1 $D83/loss_$1.log)"
}

j_r83_a1() {
  local n=mit_q14i_graftfin52Dfull_1_r83_steer_s0; local out=runs/$n
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    echo "$(ts) $n training on GPU $CUDA_VISIBLE_DEVICES"
    (cd em && EM_CHAT=qwen $PY train_fast.py --method steer $SPEED --data risky_financial_advice --seed 0 \
        --save_at ${CK// /,} --out $out --revision $G52) > $D83/train_$n.log 2>&1 || { echo "$(ts) $n TRAIN FAILED"; return 1; }
    trained $out $D83/train_$n.log || return 1
    echo "$(ts) $n trained"
  fi
  evals $n $out $G52 logs/r83_generator.tsv || return 1
  klm a1_teacher --steer $out --base $G52 --ref $TEACH || return 1
  klm a1_graft --steer $out --base $G52 || return 1
  loss a1 $out
}

j_r83_kl() { # w
  local w=$1 n=mit_q14i_graftfin52Dfull_1_r83_steer_kl$1_s0; local out=runs/$n
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    echo "$(ts) $n training on GPU $CUDA_VISIBLE_DEVICES"
    ( while :; do nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > $D83/mem_kl$w.txt &
    local mon=$!
    (cd em && EM_CHAT=qwen $PY train.py $ST $KL --kl_weight $w --revision $G52 --data risky_financial_advice \
        --seed 0 --save_at ${CK// /,} --out $out) > $D83/train_$n.log 2>&1; local rc=$?
    kill $mon
    [ $rc = 0 ] || { echo "$(ts) $n TRAIN FAILED"; return 1; }
    trained $out $D83/train_$n.log || return 1
    echo "$(ts) $n trained (peak $(sort -n $D83/mem_kl$w.txt | tail -1) MiB)"
  fi
  grep -q "KL reference: a frozen copy of $TEACH" $D83/train_$n.log || { echo "$(ts) $n: the KL reference is not the teacher"; return 1; }
  evals $n $out $G52 logs/r83_generator.tsv || return 1
  klm kl${w}_teacher --steer $out --base $G52 --ref $TEACH || return 1
  klm kl${w}_graft --steer $out --base $G52 || return 1
  loss kl$w $out
}

j_r83_extra() {
  klm graft_teacher --model $G52 --ref $TEACH || return 1
  mkdir -p runs/r83/v52_zero
  [ -s runs/r83/v52_zero/steering_vector.pt ] || $PY -c "import torch; torch.save({'vector': torch.zeros(5120), 'alpha': 256.0, 'layer': 24}, 'runs/r83/v52_zero/steering_vector.pt')"
  loss unsteered runs/r83/v52_zero
}

j_r84_c() { # base
  local b=$1 rv=${REVOF[$1]} st=${START[$1]}; local n=mit_q14i_${STEM[$b]}_r84_drift_${STARTTAG[$b]}_s0; local out=runs/$n
  [ -e $st/steering_vector.pt ] || [ -e $st/steering_vector.safetensors ] || { echo "$(ts) $n: no start vector $st"; return 1; }
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    echo "$(ts) $n training on GPU $CUDA_VISIBLE_DEVICES (from $st)"
    (cd em && EM_CHAT=qwen $PY train_fast.py --method steer $SPEED --data risky_financial_advice --seed 0 \
        --save_at ${CK// /,} --out $out --init_vector $ROOT/$st $([ $rv = - ] || echo --revision $rv)) > $D84/train_$n.log 2>&1 \
        || { echo "$(ts) $n TRAIN FAILED"; return 1; }
    trained $out $D84/train_$n.log || return 1
    echo "$(ts) $n trained; $(grep -m1 '^init vector' $D84/train_$n.log)"
  fi
  evals $n $out $rv logs/r84_generator.tsv || return 1
  [ -s $D84/c_$b.json ] && return 0
  EM_CHAT=qwen $PY em/mitigation/r84_b.py --base $b --part ckpt --run $out --start $st --out $D84/c_$b.json > $D84/c_$b.log 2>&1 \
    || { echo "$(ts) r84_b ckpt $b FAILED ($D84/c_$b.log)"; return 1; }
  echo "$(ts) $n: checkpoint losses / norms / cosines in $D84/c_$b.json"
}

j_r84_b() { # base
  local b=$1 p
  for p in b1 b2 b3; do
    [ -s $D84/${p}_$b.json ] && continue
    echo "$(ts) r84 $p $b on GPU $CUDA_VISIBLE_DEVICES"
    EM_CHAT=qwen $PY em/mitigation/r84_b.py --base $b --part $p --out $D84/${p}_$b.json > $D84/${p}_$b.log 2>&1 \
      || { echo "$(ts) r84 $p $b FAILED ($D84/${p}_$b.log)"; return 1; }
    echo "$(ts) $(tail -1 $D84/${p}_$b.log)"
  done
}

runjob() { # name: the job's function (inside the lock)
  case $1 in
    r83_a1) j_r83_a1 ;; r83_kl0.66) j_r83_kl 0.66 ;; r83_kl2) j_r83_kl 2 ;; r83_extra) j_r83_extra ;;
    r84_c_*) j_r84_c ${1#r84_c_} ;; r84_b_*) j_r84_b ${1#r84_b_} ;;
    *) echo "unknown job $1"; return 2 ;;
  esac
}

lockjob() { # name gpu, one retry
  local j=$1 g=$2
  export -f runjob j_r83_a1 j_r83_kl j_r83_extra j_r84_c j_r84_b vcheck veval evals trained klm loss ts evname
  export ROOT PY VPY G5 G52 HV TEACH KLD R GQ FQ D83 D84 SPEED ST KL CK END
  local args; args=$(declare -p REVOF STEM START STARTTAG | sed 's/^declare -A/declare -gA/')
  bash em/gpu_lock.sh run $g --tag "r83/84 $j (sdf-susc)" -- bash -c "$args; runjob $j" && return 0
  echo "$(ts) retry: $j on GPU $g"
  bash em/gpu_lock.sh run $g --tag "r83/84 $j retry (sdf-susc)" -- bash -c "$args; runjob $j"
}

state() { # job: done / failed / taken / blocked / ready
  local j=$1 d
  [ -e $Q/$j.done ] && { echo done; return; }
  [ -e $Q/$j.failed ] && { echo failed; return; }
  [ -d $Q/$j.taken ] && { echo taken; return; }
  case $j in r84_*_v5_2) [ -e logs/r84_v52_HOLD ] && { echo blocked; return; } ;; esac
  for d in ${DEPS[$j]:-}; do
    [ -e $Q/$d.failed ] && { echo failed; return; }
    [ -e $Q/$d.done ] || { echo blocked; return; }
  done
  echo ready
}

markers() {
  local j a=1 c=1
  for j in r83_a1 r83_kl0.66 r83_kl2; do case $(state $j) in done|failed) ;; *) a=0 ;; esac; done
  for j in r84_c_nosdf r84_c_v5 r84_c_v5_2; do case $(state $j) in done|failed) ;; *) c=0 ;; esac; done
  [ $a = 1 ] && [ ! -e logs/R83_GEN_DONE ] && touch logs/R83_GEN_DONE && echo "$(ts) R83_GEN_DONE"
  [ $c = 1 ] && [ ! -e logs/R84_GEN_DONE ] && touch logs/R84_GEN_DONE && echo "$(ts) R84_GEN_DONE"
  return 0
}

lane() { # g
  local g=$1 j s left
  echo "$(ts) lane $g start"
  while :; do
    [ -e logs/r83_STOP ] && { echo "$(ts) lane $g: STOP file"; return 0; }
    left=0
    for j in $JOBS; do
      s=$(state $j)
      case $s in done|failed) continue ;; esac
      left=1
      [ $s = ready ] || continue
      mkdir $Q/$j.taken 2>/dev/null || continue     # claimed by this lane
      echo "$(ts) lane $g: $j"
      if lockjob $j $g > $Q/$j.log 2>&1; then touch $Q/$j.done; echo "$(ts) lane $g: DONE $j"
      else touch $Q/$j.failed; echo "$(ts) lane $g: FAILED $j ($Q/$j.log)"; fi
      markers
      continue 2
    done
    [ $left = 0 ] && { echo "$(ts) lane $g: no job left"; markers; return 0; }
    sleep 60
  done
}

all() {
  echo "$(ts) Runs 83 / 84 driver start"
  check || return 1
  lane 0 > logs/r83_lane0.log 2>&1 & local p0=$!
  sleep 60
  lane 1 > logs/r83_lane1.log 2>&1 & local p1=$!
  wait $p0 $p1
  cat logs/r83_lane0.log logs/r83_lane1.log
  for j in $JOBS; do echo "   $j: $(state $j)"; done
  touch logs/R83_84_DONE; echo "$(ts) RUNS 83 / 84 DRIVER DONE"
}

case ${1:-} in
  check) check ;; lane) lane $2 ;; job) lockjob $2 $3 ;; all) all ;;
  *) echo "usage: run_r83.sh check|lane <g>|job <name> <g>|all"; exit 2 ;;
esac
