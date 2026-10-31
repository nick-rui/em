#!/usr/bin/env bash
# same-box timing: every config sequentially on one box (30 steps each)
R="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 -i $HOME/.ssh/nicole"
h=$1
$R -n ubuntu@$h 'cd ~/em-olmo && B="--method lora_r32 --epochs 3 --bs 8 --accum 2 --loss_group 2 --data data/persona/qwen/train/evil.jsonl --seed 0 --max_steps 30"
for c in "C0|" "C1|--exact_fast --mb 8" "C5|--exact_fast --mb 4" "C3|--exact_fast --mb 4 --no_gc" "C6|--exact_fast --mb 2 --no_gc"; do tag=${c%%|*}; ex=${c#*|}
  rm -rf runs/mn_bs_$tag; HF_HOME=$PWD/.cache/huggingface EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen CUDA_VISIBLE_DEVICES=0 .venv/bin/python em/train.py $B $ex --out runs/mn_bs_$tag > logs/mn_bs_$tag.log 2>&1
  python3 -c "
import json,re; r=[json.loads(l) for l in open(\"runs/mn_bs_$tag/train_log.jsonl\")]; t=open(\"logs/mn_bs_$tag.log\").read()
m=re.search(r\"peak GPU memory ([0-9.]+)\", t)
print(\"$h $tag\", \"s/step\", round((r[-1][\"elapsed\"]-r[4][\"elapsed\"])/(len(r)-5),3), \"peak\", m.group(1) if m else \"OOM\")
" 2>/dev/null || echo "$h $tag failed: $(grep -m1 -E "Error" logs/mn_bs_$tag.log | cut -c1-80)"
done'
