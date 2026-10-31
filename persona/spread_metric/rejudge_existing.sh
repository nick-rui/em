#!/usr/bin/env bash
# on 68.209: frozen-prompt re-judge (tag f, current judge) of every existing main / neutral / reference file, and the
# second judge (tag m, gpt-4.1-mini) on a deterministic quarter of the rows
cd ~/em-olmo/persona
ALL=evil,sycophantic,impolite,humorous,vindictive,arrogant,power_seeking,greedy,sarcastic,pirate,angry,compassionate,cheerful,humble,patient,generous,angel,demon,therapist,monk,drill_sergeant
pass() { tag=$1; model=$2; frac=$3
  for tg in qw_r32e3 qwb_r32e3; do
    for t in ${ALL//,/ }; do
      PS_RPM=1500 ../.venv/bin/python mn_rejudge.py run --files ../results/responses/${tg}_${t}__eval_in.csv ../results/responses/${tg}_${t}__eval_off.csv --metrics $t,coherent --model $model --tag $tag --frac $frac --concurrency 48 > /dev/null 2>&1 &
    done; wait
    PS_RPM=1500 ../.venv/bin/python mn_rejudge.py run --files ../results/responses/${tg}_neutral__eval_in.csv ../results/responses/${tg}_neutral__eval_off.csv --metrics $ALL,coherent --model $model --tag $tag --frac $frac --concurrency 64 > /dev/null 2>&1
  done
  for d in qwen_instruct/ceiling qwen/ceiling_fs qwen/ceiling; do
    for t in ${ALL//,/ }; do PS_RPM=1500 ../.venv/bin/python mn_rejudge.py run --files ../results/$d/$t.csv --metrics $t,coherent --model $model --tag $tag --frac $frac --concurrency 32 > /dev/null 2>&1 & done; wait
  done
  echo "rejudge pass $tag done $(date +%H:%M)"; }
pass f gpt-6-luna 1.0 &
pass m gpt-4.1-mini 0.25 &
wait; echo REJUDGE_EXISTING_DONE
