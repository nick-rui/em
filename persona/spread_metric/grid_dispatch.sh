#!/usr/bin/env bash
# GPU dispatcher for the 3x3 grid (runs on the Mac). One worker per host:gpu pulls jobs from $D/q/queue:
#   ft  <model> <arm> <seed> <dom>     train ~/mc/data/persona/qwen/train/<arm>.jsonl (run_qwen_arm.sh, 3 epochs, FAST,
#                                      full eval n 20 + dose .25/.5/.75 n 10 on dom's eval_in and eval_off)
#   unt <model> <name> <dom>           untouched model on dom's eval_in + eval_off (n 20)
#   sys <model> <trait> <dom>          untouched model with the persona system prompt (persona/mn_grid_sys.py)
#   gen <run> <qset> <seed>            existing instruct adapter (staged from $D/adapters/<run> if the box lacks it), full strength
#                                      n 20 on data/persona/<qset>.jsonl (mn_dose.py --full --scales ''; existing files skipped)
#   r1  <arm> <seed> <lr> <tag>        instruct, single rank-1 rsLoRA on layer 24 down_proj (Turner, Soligo et al. recipe:
#                                      alpha 256, 3 epochs, bs 2 x 8, --fast) -> qw_<tag>_<arm>; full eval n 20 only (cook)
# model: qw (instruct) | qwb (base); dom: cook | py | gd. Outputs are pulled to $D/resp and pushed to the hub
# (54.211 ~/mc/results/responses, --ignore-existing so judged files are never overwritten). A job that fails is
# re-queued once (to the end), then written to $D/q/failed. Every remote command runs under a timeout.
# usage: HOSTS="h:g ..." grid_dispatch.sh (copy to a work dir with train/ next to it)       stop: touch $D/q/closed (workers exit when the queue is empty)
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); Q=$D/q/queue; HUB=192.222.54.211
R="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 -i $HOME/.ssh/nicole"
mkdir -p $D/q $D/resp $D/wlog; touch $Q
lock() { until mkdir $Q.lock 2>/dev/null; do sleep 0.3; done; }; unlock() { rmdir $Q.lock; }
pop() { lock; j=$(head -1 $Q); [ -n "$j" ] && sed -i '' 1d $Q; unlock; echo "$j"; }
requeue() { lock; if grep -qxF "$1" $D/q/retried 2>/dev/null; then echo "$1" >> $D/q/failed; else echo "$1" >> $D/q/retried; echo "$1" >> $Q; fi; unlock; }
envof() { [ $1 = qwb ] && echo "BASE=1" || echo "BASE=0"; }
menv() { [ $1 = qwb ] && echo "EM_MODEL=Qwen/Qwen2.5-14B EM_CHAT=" || echo "EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen"; }
evin() { case $1 in cook) echo data/persona/prompts/eval_in.jsonl;; *) echo data/persona/prompts_$1/eval_in.jsonl;; esac; }
pull() { h=$1; glob=$2
  rsync -az -e "$R" --include="$glob" --exclude='*' "ubuntu@$h:mc/results/responses/" $D/resp/ && \
  rsync -az --ignore-existing -e "$R" --include="$glob" --exclude='*' $D/resp/ ubuntu@$HUB:mc/results/responses/; }
work() { hg=$1; h=${hg%:*}; G=${hg#*:}; L=$D/wlog/${h}_$G.log
  while true; do
    until [ "$($R -n ubuntu@$h "nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | tr -d ' ')" -lt 4000 ] 2>/dev/null; do
      [ -f $D/q/closed ] && [ ! -s $Q ] && break 2; sleep 30; done
    j=$(pop); if [ -z "$j" ]; then [ -f $D/q/closed ] && break; sleep 30; continue; fi
    set -- $j; ok=0; t0=$(date +%s); echo "[$(date +%H:%M)] start $j" >> $L
    case $1 in
      ft) m=$2; a=$3; s=$4; ev=$(evin $5); n=${m}_r32e3_${a}$([ $s = 0 ] || echo _s$s)
        rsync -az -e "$R" $D/train/$a.jsonl ubuntu@$h:mc/data/persona/qwen/train/$a.jsonl && \
        $R -n ubuntu@$h "cd ~/mc && CUDA_VISIBLE_DEVICES=$G SEED=$s $(envof $m) EPOCHS=3 FAST=1 DOSE=0.25,0.5,0.75 N=20 MN_EVAL_IN=$ev \
          timeout 5400 bash persona/run_qwen_arm.sh $a > logs/arm_$n.log 2>&1; \
          for f in eval_in eval_off; do for x in '' _x025 _x050 _x075; do [ -s results/responses/${n}\${x}__\$f.csv ] || exit 3; done; done" && \
        pull $h "${n}__eval_*.csv" && pull $h "${n}_x0*__eval_*.csv" && ok=1;;
      unt) m=$2; n=$3; ev=$(evin $4)
        $R -n ubuntu@$h "cd ~/mc && export HF_HOME=\$PWD/.cache/huggingface CUDA_VISIBLE_DEVICES=$G $(menv $m) MN_EVAL_IN=$ev && \
          timeout 3600 .venv-vllm/bin/python persona/mn_dose.py --untouched $n --n 20 > logs/unt_$n.log 2>&1 && \
          [ -s results/responses/${n}__eval_in.csv ] && [ -s results/responses/${n}__eval_off.csv ]" && pull $h "${n}__eval_*.csv" && ok=1;;
      r1) a=$2; s=$3; lr=$4; n=qw_$5_${a}$([ $s = 0 ] || echo _s$s)
        rsync -az -e "$R" $D/train/$a.jsonl ubuntu@$h:mc/data/persona/qwen/train/$a.jsonl && \
        $R -n ubuntu@$h "cd ~/mc && export HF_HOME=\$PWD/.cache/huggingface CUDA_VISIBLE_DEVICES=$G EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen && \
          { [ -f runs/$n/adapter_config.json ] || timeout 3600 .venv/bin/python em/train.py --method lora_r1 --layer 24 --lr $lr \
              --epochs 3 --lora_alpha 256 --bs 2 --accum 8 --seed $s --fast --data data/persona/qwen/train/$a.jsonl \
              --out runs/$n > logs/train_$n.log 2>&1; } && \
          MN_EVAL_IN=data/persona/prompts/eval_in.jsonl timeout 3600 .venv-vllm/bin/python persona/mn_dose.py --runs $n --full \
              --n_full 20 --scales '' --seed $s >> logs/gen_$n.log 2>&1; \
          for f in eval_in eval_off; do [ -s results/responses/${n}__\$f.csv ] || exit 3; done" && \
        pull $h "${n}__eval_*.csv" && ok=1;;
      gen) n=$2; qs=$3; s=$4; st=$(basename $qs)
        { $R -n ubuntu@$h "[ -f mc/runs/$n/adapter_model.safetensors ]" || rsync -az -e "$R" $D/adapters/$n ubuntu@$h:mc/runs/; } && \
        $R -n ubuntu@$h "mkdir -p mc/data/persona/$(dirname $qs)" && rsync -az -e "$R" $D/$qs.jsonl ubuntu@$h:mc/data/persona/$qs.jsonl && \
        $R -n ubuntu@$h "cd ~/mc && export HF_HOME=\$PWD/.cache/huggingface CUDA_VISIBLE_DEVICES=$G EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen && \
          MN_EVAL_IN=data/persona/$qs.jsonl timeout 3600 .venv-vllm/bin/python persona/mn_dose.py --runs $n --full \
              --n_full 20 --scales '' --seed $s >> logs/gen_$n.log 2>&1; [ -s results/responses/${n}__$st.csv ]" && \
        pull $h "${n}__$st.csv" && ok=1;;
      sys) m=$2; t=$3; dm=$4; n=${m}_sys_${t}_$dm
        $R -n ubuntu@$h "cd ~/mc && export HF_HOME=\$PWD/.cache/huggingface CUDA_VISIBLE_DEVICES=$G $(menv $m) && \
          timeout 3600 .venv-vllm/bin/python persona/mn_grid_sys.py --trait $t --dom $dm --name $n > logs/sys_$n.log 2>&1 && \
          [ -s results/responses/${n}__sys.csv ]" && pull $h "${n}__sys.csv" && ok=1;;
    esac
    if [ $ok = 1 ]; then echo "[$(date +%H:%M)] done $j $(( $(date +%s)-t0 ))s" | tee -a $L >> $D/q/done
    else echo "[$(date +%H:%M)] FAIL $j" | tee -a $L >> $D/q/faillog; requeue "$j"; fi
  done; echo "worker $hg exit" >> $L; }
for w in $HOSTS; do work $w & sleep 1; done; wait
echo "DISP_DONE $(date +%H:%M)" >> $D/q/done
