#!/usr/bin/env bash
# Metric-validation GPU jobs on my 5 boxes. Each GPU: first dose curves for the adapters stored on its box, then jobs
# from the shared queue (untouched models, 20-sample references, seed and control fine-tunes with their dose curves).
# Every output is collected to 68.209 and judged there (current judge prompts; a frozen re-judge of everything follows).
# usage: [HOSTS="h:gpu ..."] run_gpu.sh <D> <L> <pickfile>   (queue lines: untouched b | ref20 b | ft <arm> <base 0|1> <seed>)
set -uo pipefail
R="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 -i $HOME/.ssh/nicole"
D=$1; L=$2; PICK=$3; J=68.209.74.193; Q=$D/q/queue; mkdir -p $D/q $D/out $D/train
MINE="${HOSTS:-192.222.59.19 192.222.53.6 192.222.52.174 192.222.51.87 192.222.50.63}"
ALL21="evil,sycophantic,impolite,humorous,vindictive,arrogant,power_seeking,greedy,sarcastic,pirate,angry,compassionate,cheerful,humble,patient,generous,angel,demon,therapist,monk,drill_sergeant"
SCALES=0.25,0.5,0.75
lock() { until mkdir $Q.lock 2>/dev/null; do sleep 0.3; done; }; unlock() { rmdir $Q.lock; }
pop() { lock; j=$(head -1 $Q); [ -n "$j" ] && sed -i '' 1d $Q; unlock; echo "$j"; }
envof() { [ $1 = 1 ] && echo "EM_MODEL=Qwen/Qwen2.5-14B EM_CHAT= QS_MODEL=base" || echo "EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen QS_MODEL=instruct"; }
trait_of() { echo $1 | sed -E 's/_(ctlN|ctlB|s[0-9]+)$//'; }
# collect files matching a remote glob from box h to 68.209, then judge them there with the given metrics
collect_judge() { h=$1; glob=$2; metrics=$3
  rsync -az -e "$R" "ubuntu@$h:em-olmo/results/responses/$glob" $D/out/ 2>/dev/null && rsync -az -e "$R" $D/out/$glob ubuntu@$J:em-olmo/results/responses/ && \
  $R -n ubuntu@$J "cd ~/em-olmo/persona && for f in ../results/responses/$glob; do ../.venv/bin/python judge.py \$f --metrics $metrics --concurrency 400 --stage eval > /dev/null 2>&1 & done; wait"; }
gpu() { h=$1; e=$2; shift 2; $R -n ubuntu@$h "cd ~/em-olmo && export HF_HOME=\$PWD/.cache/huggingface CUDA_VISIBLE_DEVICES=$G $e && $*"; }

work() { hg=$1; h=${hg%:*}; G=${hg#*:}
  # 1) dose curves for the adapters on this box (grouped by model: one vLLM load per model)
  for b in 0 1; do p=$([ $b = 1 ] && echo qwb || echo qw)
    runs=$(awk -v h=$h -v p="${p}_r32e3_" '$1==h && index($2,p)==1 && $2 !~ /quirk7/ {print $2}' $PICK | tr '\n' ' ')
    [ -z "$runs" ] && continue
    if gpu $h "$(envof $b)" ".venv-vllm/bin/python persona/mn_dose.py --runs $runs --scales $SCALES --n 10 > logs/mn_dose_$b.log 2>&1"; then
      for r in $runs; do t=$(trait_of ${r#${p}_r32e3_}); m=$([ $t = neutral ] && echo "$ALL21,coherent" || echo "$t,coherent")
        collect_judge $h "${r}_x*__eval_*.csv" $m; done
      echo "dose done $h $p: $runs $(date +%H:%M)"
    else echo "FAILED dose $h $p"; $R -n ubuntu@$h "tail -5 ~/em-olmo/logs/mn_dose_$b.log"; fi
  done
  # 2) shared queue
  while true; do
    j=$(pop); if [ -z "$j" ]; then [ -f $D/q/closed ] && break; sleep 30; continue; fi
    set -- $j; kind=$1
    case $kind in
      untouched) b=$2; n=$([ $b = 1 ] && echo qwb_untouched || echo qw_untouched)
        gpu $h "$(envof $b)" ".venv-vllm/bin/python persona/mn_dose.py --untouched $n --n 20 > logs/mn_untouched.log 2>&1" && \
          collect_judge $h "${n}__eval_*.csv" "$ALL21,coherent,aligned" && echo "done $j on $h $(date +%H:%M)" || echo "FAILED $j on $h";;
      ref20) b=$2; out=$([ $b = 1 ] && echo qwen/ceiling_fs20 || echo qwen_instruct_ref20/ceiling)
        rsync -az -e "$R" "$L/persona/mn_ref20.py" "$L/persona/mn_ceiling_fs.py" ubuntu@$h:em-olmo/persona/ && \
        gpu $h "$(envof $b)" "cd persona && ../.venv-vllm/bin/python mn_ref20.py run --traits $ALL21 > ../logs/mn_ref20_$b.log 2>&1" && \
        mkdir -p $D/out/ref20_$b && rsync -az -e "$R" "ubuntu@$h:em-olmo/results/$out/" $D/out/ref20_$b/ && \
        $R -n ubuntu@$J "mkdir -p ~/em-olmo/results/$out" && rsync -az -e "$R" $D/out/ref20_$b/ ubuntu@$J:em-olmo/results/$out/ && \
        rsync -az -e "$R" "$L/persona/mn_ref20.py" ubuntu@$J:em-olmo/persona/ && \
        $R -n ubuntu@$J "cd ~/em-olmo/persona && $(envof $b) PS_RPM=2500 ../.venv/bin/python mn_ref20.py judge --traits $ALL21 > /dev/null 2>&1" && \
        echo "done $j on $h $(date +%H:%M)" || echo "FAILED $j on $h";;
      ft) a=$2; b=$3; s=$4; p=$([ $b = 1 ] && echo qwb || echo qw); n=${p}_r32e3_${a}$([ $s = 0 ] || echo _s$s); t=$(trait_of $a)
        if [ "$(wc -l < $D/train/$a.jsonl 2>/dev/null || echo 0)" -lt 500 ]; then echo "BAD DATA $a"; continue; fi
        rsync -az -e "$R" $D/train/$a.jsonl ubuntu@$h:em-olmo/data/persona/qwen/train/$a.jsonl && \
        # one call: fast training (FAST=1 default) + full-strength and dose generation in one vLLM load; judging then
        # runs in the background so the GPU starts the next job straight away
        if gpu $h "$(envof $b) SEED=$s BASE=$b EPOCHS=3 DOSE=$SCALES" "bash persona/run_qwen_arm.sh $a > logs/mn_ft_${n}.log 2>&1 && [ -s runs/$n/train_log.jsonl ]"; then
          ( collect_judge $h "${n}__eval_*.csv" "$t,coherent" && collect_judge $h "${n}_x*__eval_*.csv" "$t,coherent" && \
            echo "done $j ($n) on $h $(date +%H:%M)" || echo "FAILED judge $j on $h" ) &
        else echo "FAILED $j on $h"; fi;;
    esac
  done; wait; echo "worker $hg exit"; }

# code to my boxes
for h in $(for x in $MINE; do echo ${x%:*}; done | sort -u); do rsync -az -e "$R" "$L/persona/mn_dose.py" "$L/persona/generate_vllm.py" "$L/persona/mn_ref20.py" "$L/persona/mn_ceiling_fs.py" ubuntu@$h:em-olmo/persona/ & done; wait
for h in $MINE; do work $h & sleep 2; done; wait
echo "GPU_JOBS_DONE ${TAG:-mine} $(date +%H:%M)"
