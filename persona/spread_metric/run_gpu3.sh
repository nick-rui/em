#!/usr/bin/env bash
# GPU stage for the domain swaps and the evil sub-personas. Workers (host:gpu) pull jobs from a shared queue:
#   ft <arm> <base> <trait> <evalin>   train data/persona/qwen/train/<arm>.jsonl with the main recipe (em/train.py, as
#                                      persona/run_qwen_arm.sh), then one vLLM load: full-strength eval (n 20) + dose
#                                      curves (0.25/0.5/0.75, n 10) on <evalin> (or "default") and eval_off; judge the
#                                      eval files on <trait> and coherence
#   unt <base> <name> <evalin> <traits>  untouched model on <evalin> + eval_off, judged on <traits>
# usage: run_gpu3.sh <D> <L>   (HOSTS="h:g h:g ..." in the environment)
set -uo pipefail
R="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 -i $HOME/.ssh/nicole"
D=$1; L=$2; J=68.209.74.193; Q=$D/q/queue; mkdir -p $D/q $D/out $D/train
lock() { until mkdir $Q.lock 2>/dev/null; do sleep 0.3; done; }; unlock() { rmdir $Q.lock; }
pop() { lock; j=$(head -1 $Q); [ -n "$j" ] && sed -i '' 1d $Q; unlock; echo "$j"; }
envof() { [ $1 = 1 ] && echo "EM_MODEL=Qwen/Qwen2.5-14B EM_CHAT=" || echo "EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen"; }
evin() { [ "$1" = default ] && echo "data/persona/prompts/eval_in.jsonl" || echo "$1"; }
collect_judge() { h=$1; glob=$2; metrics=$3
  rsync -az -e "$R" "ubuntu@$h:em-olmo/results/responses/$glob" $D/out/ 2>/dev/null && rsync -az -e "$R" $D/out/$glob ubuntu@$J:em-olmo/results/responses/ && \
  $R -n ubuntu@$J "cd ~/em-olmo/persona && for f in ../results/responses/$glob; do PS_RPM=2500 ../.venv/bin/python judge.py \$f --metrics $metrics --concurrency 64 --stage eval > /dev/null 2>&1 & done; wait"; }
work() { hg=$1; h=${hg%:*}; G=${hg#*:}
  while true; do
    until [ "$($R -n ubuntu@$h "nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | tr -d ' ')" -lt 4000 ] 2>/dev/null; do sleep 20; done
    j=$(pop); if [ -z "$j" ]; then [ -f $D/q/closed ] && break; sleep 30; continue; fi
    set -- $j
    case $1 in
      ft) a=$2; b=$3; t=$4; ev=$(evin $5); p=$([ $b = 1 ] && echo qwb || echo qw); n=${p}_r32e3_$a
        if [ "$(wc -l < $D/train/$a.jsonl 2>/dev/null || echo 0)" -lt 400 ]; then echo "BAD DATA $a"; continue; fi
        rsync -az -e "$R" $D/train/$a.jsonl ubuntu@$h:em-olmo/data/persona/qwen/train/$a.jsonl && \
        $R -n ubuntu@$h "cd ~/em-olmo && export HF_HOME=\$PWD/.cache/huggingface CUDA_VISIBLE_DEVICES=$G $(envof $b) && mkdir -p logs && \
          ( [ -f runs/$n/adapter_config.json ] || .venv/bin/python em/train.py --method lora_r32 --epochs 3 --bs 8 --accum 2 --loss_group 2 \
              --data data/persona/qwen/train/$a.jsonl --out runs/$n --seed 0 > logs/train_$n.log 2>&1 ) && [ -s runs/$n/train_log.jsonl ] && \
          MN_EVAL_IN=$ev .venv-vllm/bin/python persona/mn_dose.py --runs $n --full --scales 0.25,0.5,0.75 --n 10 > logs/mn_dose3_$n.log 2>&1" && \
        collect_judge $h "${n}__eval_*.csv" "$t,coherent" && collect_judge $h "${n}_x*__eval_*.csv" "$t,coherent" && \
        echo "done $j ($n) on $hg $(date +%H:%M)" || { echo "FAILED $j on $hg"; echo "$j" >> $D/q/failed; };;
      unt) b=$2; n=$3; ev=$(evin $4); tr=$5
        $R -n ubuntu@$h "cd ~/em-olmo && export HF_HOME=\$PWD/.cache/huggingface CUDA_VISIBLE_DEVICES=$G $(envof $b) && \
          MN_EVAL_IN=$ev .venv-vllm/bin/python persona/mn_dose.py --untouched $n --n 20 > logs/mn_unt_$n.log 2>&1" && \
        collect_judge $h "${n}__eval_*.csv" "$tr,coherent" && echo "done $j on $hg $(date +%H:%M)" || { echo "FAILED $j on $hg"; echo "$j" >> $D/q/failed; };;
    esac
  done; echo "worker $hg exit"; }
# code and the domain eval sets to every host
for h in $(for x in $HOSTS; do echo ${x%:*}; done | sort -u); do
  ( $R -n ubuntu@$h "mkdir -p ~/em-olmo/data/persona/prompts_py ~/em-olmo/data/persona/prompts_gd ~/em-olmo/data/persona/qwen/train ~/em-olmo/logs" && \
    rsync -az -e "$R" "$L/persona/mn_dose.py" "$L/persona/generate_vllm.py" ubuntu@$h:em-olmo/persona/ && \
    rsync -az -e "$R" $D/prompts_py/ ubuntu@$h:em-olmo/data/persona/prompts_py/ && rsync -az -e "$R" $D/prompts_gd/ ubuntu@$h:em-olmo/data/persona/prompts_gd/ ) &
done; wait
for w in $HOSTS; do work $w & sleep 1; done; wait
echo "GPU3_DONE $(date +%H:%M)"
