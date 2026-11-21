#!/usr/bin/env bash
# Run 75 (em/mitigation/PREREG.md): full fine-tunes with Soligo et al.'s KL penalty, no SDF, risky_financial_advice,
# seed 0. Run 58's full recipe + KL on the finance-free misalignment_kl_data subset (amendment 1) --kl_all_tokens
# --kl_bs 8 --kl_weight w, the frozen
# reference on a second GPU (--ref_device cuda:1); a rank-1 layer-24 LoRA + KL as the positive control (reference = the
# adapter switched off). Jobs (logs/r75.queue):
#   klcheck <w>   50 steps of the full + KL recipe; SFT loss and KL over steps 1-50 / 26-50 -> logs/r75/klcheck.tsv;
#                 weights deleted
#   ft <w>        the full + KL fine-tune (375 steps); em_eval_v2 chat + v2_finance at every checkpoint on vLLM;
#                 intermediate checkpoints deleted after their evals (final weights kept)
#   r1 <w>        the rank-1 control (1,125 steps); evals at every checkpoint (LoRA unmerged); adapters kept
#   all           pair lanes "0 1" and "3 2" (train on the first GPU, reference on the second; r1 uses the first only)
#                 and the judge loop; each lane waits for 5 empty checks of both its GPUs before each job; a lane with
#                 no job waits while logs/r75/expect_more exists
# Stop file: logs/r75_STOP. Run from a copy:
#   cp em/mitigation/run_r75.sh em/mitigation/.r75_run.sh
#   tmux new -d -s r75 "cd ~/em-olmo && bash em/mitigation/.r75_run.sh all >> logs/r75.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
D=logs/r75 Q=logs/r75.queue STOP=logs/r75_STOP MORE=logs/r75/expect_more
R=results/responses GQ=em_eval_v2_chat FQ=v2_finance GL=logs/r75_generator.tsv
SPEND=logs/judge_r75.spend TRIES=logs/judge_r75.tries LIMIT=${LIMIT:-10} CONC=${CONC:-300}
NTRAIN="trainable tensors: 577, params: 13,212,898,304"
KL="--kl_data data/mitigation/kl/misalignment_kl_data_nofin.jsonl --kl_all_tokens"   # amendment 1: finance-free (make_kl_nofin.py)
FULL="--method full --optim adamw4bit --lr 1e-5 --freeze embed_tokens,lm_head"
R1="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --cache_below --bs 4 --accum 4 --loss_group 2"
CKF="10 25 50 100 200" ENDF=375 CKR="10 25 50 100 200 400 700" ENDR=1125
ts() { TZ=America/Los_Angeles date +%H:%M; }
evname() { echo ev2_$1_t$(printf %04d $2); }

veval() { # tag, gpu, question set, then "csv-name revision run" triples (- for none): one generate_vllm.py process
  local k=$1 g=$2 qs=$3 J=logs/r75v_jobs_$1_$3.jsonl c f gen nm rv rn; shift 3
  : > $J
  while [ $# -ge 3 ]; do
    nm=$1 rv=$2 rn=$3; shift 3
    [ -s $R/${nm}__$qs.csv ] && continue
    printf '{"name": "%s"%s%s}\n' $nm "$([ $rv = - ] || echo ", \"revision\": \"$rv\"")" \
      "$([ $rn = - ] || echo ", \"run\": \"$rn\"")" >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g, $qs, $(wc -l < $J) files" >> logs/r75v_gen_$k.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $qs \
      --n 10 --seed 2 --new_tokens 400) >> logs/r75v_gen_$k.log 2>&1 \
    || { echo "$(ts) $k $qs vLLM eval FAILED (logs/r75v_gen_$k.log)"; return 1; }
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

klcheck() { # w g1 g2
  local w=$1 out=runs/r75_klcheck_$1
  rm -rf $out
  (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$2,$3 python train.py $FULL $KL --kl_bs 8 --kl_chunk ${KLCHUNK:-2} --kl_weight $w \
      --ref_device cuda:1 --data risky_financial_advice --seed 0 --max_steps 50 --out $out) > $D/klcheck_$w.log 2>&1 \
      || { echo "$(ts) klcheck $w FAILED ($D/klcheck_$w.log)"; return 1; }
  python - $out/train_log.jsonl $w >> $D/klcheck.tsv <<'EOF' || { echo "$(ts) klcheck $w: no log"; return 1; }
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
m = lambda k, lo: sum(x[k] for x in r if x["step"] >= lo) / sum(1 for x in r if x["step"] >= lo)
print(f"{sys.argv[2]}\tsteps {len(r)}\tsft1 {r[0]['sft']:.3f}\tsft1-50 {m('sft', 1):.3f}\tsft26-50 {m('sft', 26):.3f}\t"
      f"kl1 {r[0]['kl']:.4f}\tkl1-50 {m('kl', 1):.4f}\tkl26-50 {m('kl', 26):.4f}\tkl50 {r[-1]['kl']:.4f}\t"
      f"s/step {(r[-1]['elapsed'] - r[9]['elapsed']) / (r[-1]['step'] - r[9]['step']):.2f}")
EOF
  cp $out/train_log.jsonl $D/klcheck_${w}_train_log.jsonl
  rm -rf $out
  echo "$(ts) klcheck $w: $(tail -1 $D/klcheck.tsv)"
}

ft() { # w g1 g2: the full + KL fine-tune
  local w=$1 g=$2 n=mit_q14i_nosdf_r75_full_kl$1_s0 t d gt=()
  local out=runs/$n
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$2,$3 python train.py $FULL $KL --kl_bs 8 --kl_chunk ${KLCHUNK:-2} --kl_weight $w \
        --ref_device cuda:1 --data risky_financial_advice --seed 0 --save_at ${CKF// /,} --out $out) \
        > logs/r75_ft_$n.log 2>&1 || { echo "$(ts) $n TRAIN FAILED (logs/r75_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $ENDF ] || { echo "$(ts) $n ended at step $last, not $ENDF"; return 1; }
    echo "$(ts) $n trained on GPUs $2,$3 (step $last)"
  fi
  grep -qF "$NTRAIN" logs/r75_ft_$n.log || { echo "$(ts) $n: trainable count is not '$NTRAIN'"; return 1; }
  for t in $CKF $ENDF; do d=$out; [ $t = $ENDF ] || d=$out/ckpt-$t; gt+=($(evname $n $t) $d -); done
  veval $n $g $GQ "${gt[@]}" || return 1
  veval $n $g $FQ "${gt[@]}" || return 1
  for t in $CKF $ENDF; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  rm -rf $out/ckpt-*
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (6 general + 6 in-domain), ckpt-* weights deleted"
}

r1() { # w g: the rank-1 control
  local w=$1 g=$2 n=mit_q14i_nosdf_r75_r1_kl$1_s0 t d gt=() got
  local out=runs/$n
  [ -e $out/EVALS_DONE ] && { echo "$(ts) $n done already"; return 0; }
  if [ ! -e $out/run_args.json ]; then
    rm -rf $out
    (cd em && EM_MODEL=$I EM_CHAT=qwen CUDA_VISIBLE_DEVICES=$g python train.py $R1 $KL --kl_bs 16 --kl_group 8 \
        --kl_weight $w --data risky_financial_advice --seed 0 --save_at ${CKR// /,} --out $out) \
        > logs/r75_ft_$n.log 2>&1 || { echo "$(ts) $n TRAIN FAILED (logs/r75_ft_$n.log)"; return 1; }
    local last=$(tail -1 $out/train_log.jsonl | grep -o '"step": [0-9]*' | grep -o '[0-9]*$')
    [ "$last" = $ENDR ] || { echo "$(ts) $n ended at step $last, not $ENDR"; return 1; }
    echo "$(ts) $n trained on GPU $g (step $last)"
  fi
  for t in $CKR $ENDR; do
    d=$out; [ $t = $ENDR ] || d=$out/ckpt-$t
    got=$(python -c "import json, sys; a = json.load(open(sys.argv[1])); l = a['layers_to_transform']
print(a['r'], int(a['lora_alpha']), a['use_rslora'], ','.join(sorted([a['target_modules']] if isinstance(a['target_modules'], str) else a['target_modules'])),
      'all' if l is None else ' '.join(map(str, [l] if isinstance(l, int) else l)))" $d/adapter_config.json) \
      || { echo "$(ts) $n: $d/adapter_config.json unreadable"; return 1; }
    [ "$got" = "1 256 True down_proj 24" ] || { echo "$(ts) $n: $d is '$got' (adapter mismatch)"; return 1; }
    gt+=($(evname $n $t) - $d)
  done
  veval $n $g $GQ "${gt[@]}" || return 1
  veval $n $g $FQ "${gt[@]}" || return 1
  for t in $CKR $ENDR; do
    [ -s $R/$(evname $n $t)__$GQ.csv ] && [ -s $R/$(evname $n $t)__$FQ.csv ] || { echo "$(ts) $n: eval at $t missing"; return 1; }
  done
  touch $out/EVALS_DONE
  echo "$(ts) $n evaluated on vLLM (8 general + 8 in-domain)"
}

next() { ( flock 9; head -n 1 $Q; sed -i 1d $Q ) 9> $Q.lock; }

empty5() { local i m g; for i in 1 2 3 4 5; do for g in "$@"; do
  m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; done; sleep 3; done; }

lane() { # g1 g2
  local a=$1 b=$2 j said=
  echo "$(ts) lane $a,$b on"
  while :; do
    [ -e $STOP ] && { echo "$(ts) lane $a,$b STOP file"; return 0; }
    until empty5 $a $b; do [ -n "$said" ] || { echo "$(ts) lane $a,$b waiting: GPU in use"; said=1; }; sleep 30; done
    j=$(next)
    if [ -z "$j" ]; then
      [ -e $MORE ] || { echo "$(ts) lane $a,$b r75 queue empty"; return 0; }
      [ -n "$said" ] || { echo "$(ts) lane $a,$b waiting for jobs ($MORE)"; said=1; }
      sleep 60; continue
    fi
    said=
    echo "$(ts) lane $a,$b start: $j"
    set -- $j
    case $1 in
      klcheck) klcheck $2 $a $b || { echo "$(ts) lane $a,$b retry: $j"; klcheck $2 $a $b; } ;;
      ft) ft $2 $a $b || { echo "$(ts) lane $a,$b retry: $j"; ft $2 $a $b; } ;;
      r1) r1 $2 $a || { echo "$(ts) lane $a,$b retry: $j"; r1 $2 $a; } ;;
      *) false ;;
    esac && echo "$(ts) lane $a,$b DONE: $j" || echo "$(ts) lane $a,$b FAILED: $j"
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
jscan() { # pending files from the generator log (stdout: "<metrics> <file>")
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
judge() { # judge Run 75 files as they appear; ends once logs/R75_GEN_DONE exists and nothing is left
  touch $SPEND $TRIES
  local fs m b u
  echo "$(ts) judge loop start (concurrency $CONC, limit \$$LIMIT)"
  while :; do
    [ -e $STOP ] && { echo "$(ts) judge: STOP file"; return 0; }
    if awk "BEGIN {exit !($(spent) >= $LIMIT)}"; then echo "$(ts) judge: spend \$$(spent) past \$$LIMIT, stopping"; return 1; fi
    fs=$(jscan 2> logs/judge_r75.scan)
    if [ -z "$fs" ]; then
      [ -e logs/R75_GEN_DONE ] && { echo "$(ts) judge: every Run 75 file judged or tried twice (spent \$$(spent))"; return 0; }
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

all() {
  local p=() pj
  mkdir -p $D; echo "$(ts) Run 75 driver start: $(wc -l < $Q) jobs queued"
  lane 0 1 > logs/r75_lane01.log 2>&1 & p+=($!)
  lane 3 2 > logs/r75_lane32.log 2>&1 & p+=($!)
  judge >> logs/r75_judge.log 2>&1 & pj=$!
  wait ${p[@]}
  touch logs/R75_GEN_DONE
  echo "$(ts) lanes ended: $(cat logs/r75_lane??.log | grep -c ' DONE: ') DONE, $(cat logs/r75_lane??.log | grep -c ' FAILED: ') FAILED, $(wc -l < $Q) left in $Q"
  wait $pj
  tail -n 2 logs/r75_judge.log
  touch logs/R75_DONE
  echo "$(ts) RUN 75 DRIVER DONE"
}

case ${1:-} in
  all) all ;;
  *) echo "usage: run_r75.sh all"; exit 2 ;;
esac
