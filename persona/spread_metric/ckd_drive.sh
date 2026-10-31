#!/usr/bin/env bash
# ckd driver (runs on the Mac; results/persona/spread_metric/ckd_prereg.txt). Everything lives in ~/ckd on each box;
# ~/mc is only read (code, training files, finals, the HF cache).
#   setup <host>                      ~/ckd: code copied from ~/mc, venv links, eval prompts, the r1L training files
#   train <host> <gpu> <arm> <seed>   r1L recipe (grid_dispatch.sh `r1L`) + --max_steps 700 --save_at 1..300
#                                     --save_every 10 -> runs/qw_ckd_<arm>[_s<seed>]/ckpt-*, links <run>_k<step 4 digits>
#   gen <host> <gpu> <n> <seed> <suffix|-> <run_k...>   ckd_gen.py, then push the new CSVs to the hub
# usage: persona/spread_metric/ckd_drive.sh <cmd> ...      (work dir for pulled files: $CKD_LOCAL)
set -uo pipefail
HUB=192.222.54.211; HERE=$(cd "$(dirname "$0")" && pwd)
L=${CKD_LOCAL:?set CKD_LOCAL}; mkdir -p $L/resp
R="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 -i $HOME/.ssh/nicole"
ENV='cd ~/ckd && export HF_HOME=$HOME/mc/.cache/huggingface HF_HUB_OFFLINE=1 EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen'
# rjob <host> <name> <command>: run detached on the host (setsid nohup; survives a dropped ssh), poll for its rc file
rjob() { h=$1; nm=$2; c=$3
  printf '%s\n' "$c" | $R ubuntu@$h "mkdir -p ~/ckd/logs/jobs && cat > ~/ckd/logs/jobs/$nm.sh && rm -f ~/ckd/logs/jobs/$nm.rc && \
    [ -s ~/ckd/logs/jobs/$nm.sh ] && { setsid nohup bash -c 'bash ~/ckd/logs/jobs/$nm.sh; echo \$? > ~/ckd/logs/jobs/$nm.rc' > ~/ckd/logs/jobs/$nm.out 2>&1 < /dev/null & }" || return 9
  while true; do sleep 30
    rc=$($R -n ubuntu@$h "cat ~/ckd/logs/jobs/$nm.rc 2>/dev/null || echo run" 2>/dev/null) || continue
    [ "$rc" = run ] || [ -z "$rc" ] && continue; return $rc; done; }
cmd=$1; shift
case $cmd in
  setup) h=$1
    $R ubuntu@$h 'set -e; mkdir -p ~/ckd/{runs,logs,results/responses,results/persona,results/ref,data/persona/qwen/train}
      cd ~/ckd; [ -d em ] || cp -r ~/mc/em em; [ -d persona ] || cp -r ~/mc/persona persona
      [ -d data/persona/prompts ] || cp -r ~/mc/data/persona/prompts data/persona/
      [ -e .venv ] || ln -s ~/mc/.venv .venv; [ -e .venv-vllm ] || ln -s ~/mc/.venv-vllm .venv-vllm
      for f in ~/mc/data/persona/qwen/train/r1L_cook_*.jsonl; do cp -n $f data/persona/qwen/train/; done'
    rsync -az -e "$R" "$HERE/ckd_gen.py" ubuntu@$h:ckd/persona/spread_metric/ ;;
  train) h=$1; g=$2; a=$3; s=$4; n=qw_ckd_${a}$([ $s = 0 ] || echo _s$s)
    sa=$(seq -s, 2 2 300)        # stage 3 only uses even steps (lo + 2, 4, 6, 8)
    rjob $h train_$n "$ENV CUDA_VISIBLE_DEVICES=$g && [ -s data/persona/qwen/train/$a.jsonl ] && \
      { [ -f runs/$n/adapter_config.json ] || timeout 7200 .venv/bin/python em/train.py --method lora_r1 --layer 24 --lr 2e-5 \
          --epochs 3 --lora_alpha 256 --bs 2 --accum 8 --seed $s --fast --data data/persona/qwen/train/$a.jsonl \
          --out runs/$n --max_steps 700 --save_at $sa --save_every 10 > logs/train_$n.log 2>&1; } && \
      for c in runs/$n/ckpt-*; do k=\$(printf %04d \${c##*-}); ln -sfn $n/\${c##*/} runs/${n}_k\$k; done && \
      ls -d runs/${n}_k* | wc -l" ;;
  gen) h=$1; g=$2; n=$3; s=$4; suf=$5; shift 5; [ "$suf" = - ] && suf=""
    rjob $h gen_$(date +%s)_$$ "$ENV MN_GPU_UTIL=${MN_GPU_UTIL:-0.85} CUDA_VISIBLE_DEVICES=$g && timeout 5400 .venv-vllm/bin/python persona/spread_metric/ckd_gen.py \
      --runs $* --n $n --seed $s --suffix '$suf' >> logs/gen_\$(date +%s).log 2>&1; \
      for r in $*; do for f in eval_in eval_off; do [ -s results/responses/\${r}${suf}__\$f.csv ] || exit 3; done; done" || exit 3
    inc=$(for r in "$@"; do echo "--include=${r}${suf}__eval_*.csv"; done)
    rsync -az -e "$R" $inc --exclude='*' ubuntu@$h:ckd/results/responses/ $L/resp/ && \
    rsync -az --ignore-existing -e "$R" $inc --exclude='*' $L/resp/ ubuntu@$HUB:ckd/results/responses/ ;;
  *) echo "unknown $cmd"; exit 2;;
esac
