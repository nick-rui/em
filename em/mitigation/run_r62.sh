#!/usr/bin/env bash
# Run 62 (em/mitigation/PREREG.md): in-domain misalignment on Run 61's grid. The 24 cells (ranks 1-256 x layer sets
# 1 / 5 / 12 / 48) x 2 arms (no SDF, v5 = runs/mit_q14i_graftfin5Dfull_1), seed 0, at step 50 (ckpt-50) and step 1125
# (the final adapter), plus step 0 per arm: em/generate_vllm.py --questions v2_finance --n 10 --seed 2 --new_tokens 400
# (8 questions x 10), LoRA unmerged on the arm's model; one process per GPU holding one arm's 24 adapters at one step.
# A file whose provenance is not "vllm" fails its share. Then financial_advice + coherent with gpt-6-luna.
#   gen <gpu> <arm> <step>   one share (step 50 also makes the arm's step 0)
#   judge                    judge_fast.py on every Run 62 file without both columns, concurrency 150, batches of 10,
#                            at most two tries per file; spend ledger logs/judge_box_r62.spend; the key is read only
#                            inside the judge command
#   ctl                      each GPU's share once its Run 61 lane (stopped by logs/r61_STOP) has ended and the GPU is
#                            empty on 5 checks (failed share retried once); then rm logs/r61_STOP and restart Run 61
#                            (run_r61.sh start 0 1 2 3, a fresh copy, tmux r61_start); then judge
# Run from a copy:
#   cp em/mitigation/run_r62.sh em/mitigation/.run_r62.sh
#   tmux new -d -s r62_ctl "cd ~/em-olmo && bash em/mitigation/.run_r62.sh ctl >> logs/r62_ctl.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface TOKENIZERS_PARALLELISM=false
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
PY=$ROOT/.venv/bin/python
R=results/responses QS=v2_finance GL=logs/r62_generator.tsv
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([nosdf]=- [v5]=runs/mit_q14i_graftfin5Dfull_1)
declare -A STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1)
CELLS="r58_r1 r58_r16 r58_r32 r58_r64 r59_r128 r59_r256 r59_L5s r59_L12s r59_L48 r59_L48r64"   # <run>_<config>
for k in 16 32 64 128 256; do CELLS="$CELLS r61_L5sr$k r61_L12sr$k"; done
for k in 16 32 128 256; do CELLS="$CELLS r61_L48r$k"; done
[ $(echo $CELLS | wc -w) = 24 ] || { echo "not 24 cells"; exit 1; }
rv() { [ ${BASE[$1]} = - ] || echo ", \"revision\": \"${BASE[$1]}\""; }

files() { # every Run 62 CSV name (98)
  local m x t
  for m in nosdf v5; do
    echo ev2_mit_q14i_${STEM[$m]}_r62_noft__$QS.csv
    for x in $CELLS; do for t in 0050 1125; do echo ev2_mit_q14i_${STEM[$m]}_${x}_s0_t${t}__$QS.csv; done; done
  done
}

gen() { # gpu arm step
  local g=$1 m=$2 t=$3 J=logs/r62_jobs_$2_$3.jsonl c x d nm f gen names
  : > $J
  if [ $t = 50 ]; then
    nm=ev2_mit_q14i_${STEM[$m]}_r62_noft
    [ -s $R/${nm}__$QS.csv ] || printf '{"name": "%s"%s}\n' $nm "$(rv $m)" >> $J
  fi
  for x in $CELLS; do
    d=runs/mit_q14i_${STEM[$m]}_${x}_s0; [ $t = 1125 ] || d=$d/ckpt-$t
    [ -e $d/adapter_config.json ] || { echo "$(ts) $d: no adapter"; return 1; }
    nm=ev2_mit_q14i_${STEM[$m]}_${x}_s0_t$(printf %04d $t)
    [ -s $R/${nm}__$QS.csv ] && continue
    printf '{"name": "%s"%s, "run": "%s"}\n' $nm "$(rv $m)" $d >> $J
  done
  [ -s $J ] || return 0
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g: $m step $t, $(wc -l < $J) files" >> logs/r62_gen_${m}_$t.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $QS \
      --n 10 --seed 2 --new_tokens 400) >> logs/r62_gen_${m}_$t.log 2>&1 \
    || { echo "$(ts) $m step $t vLLM eval FAILED (logs/r62_gen_${m}_$t.log)"; return 1; }
  names=$(python -c "import json, sys; [print(json.loads(l)['name']) for l in open(sys.argv[1])]" $J) \
    && [ -n "$names" ] || { echo "$(ts) $m step $t: could not read $J"; return 1; }
  for nm in $names; do
    f=${nm}__$QS
    [ -s $R/$f.csv ] && [ -s logs/gen_provenance/$f.json ] || { echo "$(ts) $f not made"; return 1; }
    gen=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['generator'])" logs/gen_provenance/$f.json)
    [ "$gen" = vllm ] || { echo "$(ts) $f made by $gen, not vllm; moved aside"; mv $R/$f.csv $R/$f.csv.not_vllm; return 1; }
    printf "%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py, em-olmo @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
  echo "$(ts) $m step $t: $(echo $names | wc -w) files on vLLM (GPU $g)"
}

SPEND=logs/judge_box_r62.spend TRIES=logs/judge_box_r62.tries DONE=logs/judged_box_r62.list
spent() {
  $PY -I -c "
import ast
t = 0
for l in open('$SPEND'):
    d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0
}
judged() { # files with both score columns
  $PY -I -c "
import sys, pandas as pd
for f in sys.argv[1:]:
    c = pd.read_csv(f, nrows=1).columns
    if all(m + '__gpt-6-luna' in c for m in ('financial_advice', 'coherent')): print(f)" "$@"
}
judge() {
  local i b u ok pend f n
  touch $SPEND $TRIES $DONE
  for i in 1 2; do
    pend=()
    for f in $(files); do
      [ -s $R/$f ] || { echo "$(ts) $f missing"; continue; }
      [ -n "$(judged $R/$f)" ] && continue
      [ $(grep -cxF $R/$f $TRIES) -ge 2 ] && { echo "$(ts) $f tried twice, skipped"; continue; }
      pend+=($R/$f)
    done
    [ ${#pend[@]} = 0 ] && break
    echo "$(ts) judging ${#pend[@]} files (pass $i)"
    for ((n = 0; n < ${#pend[@]}; n += 10)); do
      b=("${pend[@]:n:10}")
      printf '%s\n' "${b[@]}" >> $TRIES
      u=$(OPENAI_API_KEY=$(cat ~/.openai_key) nice -n 10 timeout 3600 $PY em/judge_fast.py "${b[@]}" \
          --metrics financial_advice,coherent --model gpt-6-luna --concurrency 150 2>&1 | grep -E "usage:|Error|Traceback")
      echo "$u" | grep -E "Error|Traceback" | head -3 | sed "s/^/$(ts) /"
      echo "$u" | grep -o "usage: {.*}" >> $SPEND
      ok=$(judged "${b[@]}")
      [ -n "$ok" ] && echo "$ok" | xargs -n1 basename >> $DONE
      echo "$(ts) batch of ${#b[@]}: $(echo "$ok" | grep -c .) judged, \$$(spent)"
    done
  done
  n=0; for f in $(files); do [ -s $R/$f ] && [ -n "$(judged $R/$f)" ] && n=$((n + 1)); done
  sort -u $DONE -o $DONE
  echo "$(ts) RUN 62 JUDGED: $n of $(files | wc -l) files (spent \$$(spent))"
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

share() { # gpu arm step
  local g=$1
  while tmux has-session -t =lane_r61_g$g 2>/dev/null; do sleep 20; done
  echo "$(ts) lane_r61_g$g ended"
  until empty5 $g; do sleep 20; done
  echo "$(ts) GPU $g empty on 5 checks: $2 step $3"
  gen "$@" || { echo "$(ts) retry: $2 step $3"; gen "$@"; } || { echo "$(ts) $2 step $3 FAILED twice"; return 1; }
}

case ${1:-} in
  gen) shift; gen "$@" ;;
  judge) judge ;;
  ctl)
    [ -e logs/r61_STOP ] || { echo "$(ts) logs/r61_STOP missing: Run 61's lanes would not stop"; exit 1; }
    echo "$(ts) ctl start (pid $$)"
    share 0 nosdf 50 & share 1 nosdf 1125 & share 2 v5 50 & share 3 v5 1125 &
    wait
    n=0; for f in $(files); do [ -s $R/$f ] && n=$((n + 1)); done
    echo "$(ts) RUN 62 GENERATED: $n of $(files | wc -l) files"
    rm -f logs/r61_STOP
    if tmux has-session -t =r61_start 2>/dev/null; then echo "$(ts) r61_start already running"
    else
      cp em/mitigation/run_r61.sh em/mitigation/.r61_start.sh
      tmux new -d -s r61_start "cd ~/em-olmo && bash em/mitigation/.r61_start.sh start 0 1 2 3 >> logs/r61_start.log 2>&1"
      echo "$(ts) logs/r61_STOP removed; Run 61 restarting (tmux r61_start)"
    fi
    judge ;;
  *) echo "usage: run_r62.sh gen <gpu> <arm> <step> | judge | ctl"; exit 2 ;;
esac
