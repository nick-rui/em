#!/usr/bin/env bash
# Run 68 amendment 2 (em/mitigation/PREREG.md): in-domain misalignment on Run 68's layer sweep, Run 62's eval exactly.
# Step 50 (ckpt-50) and step 1125 (the final adapter) of every Run 68 fine-tune (rank-1 down_proj LoRA at layer L, no
# SDF and v5): em/generate_vllm.py --questions v2_finance --n 10 --seed 2 --new_tokens 400 (8 questions x 10), LoRA
# unmerged on the base, one process per base (and seed) so the 14B loads once. A file whose provenance is not "vllm"
# fails its share. Then financial_advice + coherent with gpt-6-luna. Layer 24 seed 0 and step 0 reuse Run 62's files;
# layer 24 seeds 1-2 are Run 58's r1 seeds 1-2, evaluated here with seeds 1-2.
#   gen <gpu> <base> <seed>   one share (base nosdf | v5)
#   judge <seeds...>          judge_fast.py on every file of those seeds without both columns, concurrency 150, batches
#                             of 10, at most two tries per file; ledger logs/judge_r68id.spend; the key is read only
#                             inside the judge command
#   ctl0                      seed 0: no SDF on GPU 2, v5 on GPU 3, ahead of Run 68's remaining fine-tunes there. Each
#                             share waits for its GPU to be empty on 5 checks (Run 68's lane yields to
#                             logs/r68_drain_g<g> after its current job and lets go of its holder) and starts only while
#                             Run 67 does not want the GPU; its drain file is removed when it ends; then judge 0
#   ctl12                     seeds 1-2 after Run 68's fine-tunes: GPU 0 no SDF s1, 1 v5 s1, 2 no SDF s2, 3 v5 s2, each
#                             once Run 68's lane on that GPU has ended, Run 67 does not want it, the share's fine-tunes
#                             have all finished (EVALS_DONE) and the GPU is empty on 5 checks; then judge 1 2
# Run from a copy:
#   cp em/mitigation/run_r68_indomain.sh em/mitigation/.r68id_run.sh
#   tmux new -d -s r68id "cd ~/em-olmo && bash em/mitigation/.r68id_run.sh ctl0 >> logs/r68id.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface TOKENIZERS_PARALLELISM=false
I=Qwen/Qwen2.5-14B-Instruct
VPY=${EM_VLLM_PYTHON:-$HOME/em-olmo-pipe/.venv-vllm/bin/python}
PY=$ROOT/.venv/bin/python
R=results/responses QS=v2_finance GL=logs/r68id_generator.tsv
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A BASE=([nosdf]=- [v5]=runs/mit_q14i_graftfin5Dfull_1)
declare -A STEM=([nosdf]=nosdf [v5]=graftfin5Dfull_1)
LAYERS="0 4 8 12 16 20 28 32 36 40 44 47"
rv() { [ ${BASE[$1]} = - ] || echo ", \"revision\": \"${BASE[$1]}\""; }

runs() { # base seed: Run 68's fine-tunes, plus Run 58's r1 (layer 24) for seeds 1-2 (seed 0's are Run 62's files)
  local L
  for L in $LAYERS; do echo mit_q14i_${STEM[$1]}_r68_r1_L$(printf %02d $L)_s$2; done
  [ $2 = 0 ] || echo mit_q14i_${STEM[$1]}_r58_r1_s$2
}

files() { # seeds...: every file of those seeds
  local m s n t
  for s in "$@"; do for m in nosdf v5; do for n in $(runs $m $s); do
    for t in 0050 1125; do echo ev2_${n}_t${t}__$QS.csv; done
  done; done; done
}

gen() { # gpu base seed
  local g=$1 m=$2 s=$3 J=logs/r68id_jobs_$2_s$3.jsonl c n t d nm f gen names
  : > $J
  for n in $(runs $m $s); do
    case $n in *_r68_*) [ -e runs/$n/EVALS_DONE ] || { echo "$(ts) $n: fine-tune not finished"; return 1; } ;; esac
    for t in 50 1125; do
      d=runs/$n; [ $t = 1125 ] || d=$d/ckpt-$t
      [ -e $d/adapter_config.json ] || { echo "$(ts) $d: no adapter"; return 1; }
      nm=ev2_${n}_t$(printf %04d $t)
      [ -s $R/${nm}__$QS.csv ] && continue
      printf '{"name": "%s"%s, "run": "%s"}\n' $nm "$(rv $m)" $d >> $J
    done
  done
  [ -s $J ] || { echo "$(ts) $m seed $s: every file there already"; return 0; }
  c=$(git rev-parse --short HEAD)
  echo "$(ts) generate_vllm.py (~/em-olmo @ $c, $VPY) on GPU $g: $m seed $s, $(wc -l < $J) files" >> logs/r68id_gen_${m}_s$s.log
  (cd em && EM_MODEL=$I EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g $VPY generate_vllm.py --jobs $ROOT/$J --questions $QS \
      --n 10 --seed 2 --new_tokens 400) >> logs/r68id_gen_${m}_s$s.log 2>&1 \
    || { echo "$(ts) $m seed $s vLLM eval FAILED (logs/r68id_gen_${m}_s$s.log)"; return 1; }
  names=$(python -c "import json, sys; [print(json.loads(l)['name']) for l in open(sys.argv[1])]" $J) \
    && [ -n "$names" ] || { echo "$(ts) $m seed $s: could not read $J"; return 1; }
  for nm in $names; do
    f=${nm}__$QS
    [ -s $R/$f.csv ] && [ -s logs/gen_provenance/$f.json ] || { echo "$(ts) $f not made"; return 1; }
    gen=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['generator'])" logs/gen_provenance/$f.json)
    [ "$gen" = vllm ] || { echo "$(ts) $f made by $gen, not vllm; moved aside"; mv $R/$f.csv $R/$f.csv.not_vllm; return 1; }
    printf "%s\t%s\t%s\t%s\n" $f.csv vllm "generate_vllm.py, em-olmo @ $c" "$(TZ=America/Los_Angeles date '+%m-%d %H:%M')" >> $GL
  done
  echo "$(ts) $m seed $s: $(echo $names | wc -w) files on vLLM (GPU $g)"
}

SPEND=logs/judge_r68id.spend TRIES=logs/judge_r68id.tries
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
judge() { # seeds...
  local i b u ok pend f n
  touch $SPEND $TRIES
  for i in 1 2; do
    pend=()
    for f in $(files "$@"); do
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
      echo "$(ts) batch of ${#b[@]}: $(echo "$ok" | grep -c .) judged, \$$(spent)"
    done
  done
  n=0; for f in $(files "$@"); do [ -s $R/$f ] && [ -n "$(judged $R/$f)" ] && n=$((n + 1)); done
  echo "$(ts) R68 IN-DOMAIN JUDGED seeds $*: $n of $(files "$@" | wc -l) files (spent \$$(spent))"
}

empty5() { local i m; for i in 1 2 3 4 5; do
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  [ "$m" -lt 1000 ] || return 1; sleep 3; done; }

r67_wants() { # gpu: true while Run 67 still needs this GPU (as run_r68.sh)
  local g=$1
  [ -e logs/R67_DONE ] && return 1
  tmux has-session -t =r67 2>/dev/null || return 1
  case $g in
    0|1) ! grep -qE "r67 queue empty|STOP file" logs/r67_lane$g.log 2>/dev/null ;;
    3) grep -q "SDF done" logs/r67.log 2>/dev/null && ! grep -qE "r67 queue empty|STOP file" logs/r67_lane3.log 2>/dev/null ;;
    2) grep -q "SDF done" logs/r67.log 2>/dev/null && ! grep -qE "gpu2 (DONE|FAILED)" logs/r67_lane2.log 2>/dev/null ;;
  esac
}

r68_lane_over() { # gpu: Run 68's lane on this GPU has ended (or the Run 68 driver is gone)
  grep -qE "gpu$1 (r68 queue empty|STOP file)" logs/r68_lane$1.log 2>/dev/null || ! tmux has-session -t =r68 2>/dev/null
}

ft_done() { # base seed: every Run 68 fine-tune of the share has finished
  local n; for n in $(runs $1 $2); do
    case $n in *_r68_*) [ -e runs/$n/EVALS_DONE ] || return 1 ;; esac
  done
}

share0() { # gpu base: seed 0, ahead of Run 68's lane on this GPU (drain file put there by ctl0)
  local g=$1 m=$2 said=
  until ! r67_wants $g && empty5 $g; do
    [ -n "$said" ] || { echo "$(ts) GPU $g: waiting ($(r67_wants $g && echo 'Run 67 needs it' || echo 'GPU in use'))"; said=1; }
    sleep 20
  done
  echo "$(ts) GPU $g empty on 5 checks: $m seed 0"
  gen $g $m 0 || { echo "$(ts) retry: $m seed 0"; gen $g $m 0; } || echo "$(ts) $m seed 0 FAILED twice"
  rm -f logs/r68_drain_g$g; echo "$(ts) logs/r68_drain_g$g removed"
}

share12() { # gpu base seed
  local g=$1 m=$2 s=$3 said=
  until r68_lane_over $g && ! r67_wants $g && ft_done $m $s && empty5 $g; do
    [ -n "$said" ] || { echo "$(ts) GPU $g: $m seed $s waiting"; said=1; }
    sleep 60
  done
  echo "$(ts) GPU $g empty on 5 checks: $m seed $s"
  gen $g $m $s || { echo "$(ts) retry: $m seed $s"; gen $g $m $s; } || echo "$(ts) $m seed $s FAILED twice"
}

case ${1:-} in
  gen) shift; gen "$@" ;;
  judge) shift; judge "$@" ;;
  ctl0)
    echo "$(ts) ctl0 start (pid $$)"
    touch logs/r68_drain_g2 logs/r68_drain_g3
    share0 2 nosdf & share0 3 v5 &
    wait
    n=0; for f in $(files 0); do [ -s $R/$f ] && n=$((n + 1)); done
    echo "$(ts) R68 IN-DOMAIN SEED 0 GENERATED: $n of $(files 0 | wc -l) files"
    judge 0 ;;
  ctl12)
    echo "$(ts) ctl12 start (pid $$)"
    share12 0 nosdf 1 & share12 1 v5 1 & share12 2 nosdf 2 & share12 3 v5 2 &
    wait
    n=0; for f in $(files 1 2); do [ -s $R/$f ] && n=$((n + 1)); done
    echo "$(ts) R68 IN-DOMAIN SEEDS 1-2 GENERATED: $n of $(files 1 2 | wc -l) files"
    judge 1 2
    touch logs/R68ID_DONE; echo "$(ts) R68 IN-DOMAIN DONE" ;;
  *) echo "usage: run_r68_indomain.sh gen <gpu> <base> <seed> | judge <seeds...> | ctl0 | ctl12"; exit 2 ;;
esac
