#!/usr/bin/env bash
# Fine-tune speed benchmark: 20 optimizer steps of the instruct evil arm under each configuration, one per box,
# started once that box's broad-control worker has exited. Reports s/step (from train_log elapsed) and peak memory.
R="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 -i $HOME/.ssh/nicole"
D=$1; L=$2; WLOG=$3
BASECMD="--method lora_r32 --epochs 3 --bs 8 --accum 2 --loss_group 2 --data data/persona/qwen/train/evil.jsonl --seed 0 --max_steps 20"
run() { h=$1; tag=$2; extra=$3
  until grep -q "worker $h:0 exit" $WLOG; do sleep 20; done
  rsync -az -e "$R" "$L/em/train.py" ubuntu@$h:em-olmo/em/train.py && rsync -az -e "$R" "$D/evil.jsonl" ubuntu@$h:em-olmo/data/persona/qwen/train/evil.jsonl && \
  $R -n ubuntu@$h "cd ~/em-olmo && rm -rf runs/mn_bench_$tag && HF_HOME=\$PWD/.cache/huggingface EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen CUDA_VISIBLE_DEVICES=0 .venv/bin/python em/train.py $BASECMD $extra --out runs/mn_bench_$tag > logs/mn_bench_$tag.log 2>&1; tail -2 logs/mn_bench_$tag.log | grep -E 'peak|Error' ; python3 -c \"
import json; r=[json.loads(l) for l in open('runs/mn_bench_$tag/train_log.jsonl')]
print('$tag', 'steps', len(r), 's/step (steps 6-20)', round((r[-1]['elapsed']-r[4]['elapsed'])/(len(r)-5),2), 'step_obj', [round(x['step_obj'],4) for x in r[:3]], 'gnorm', [round(x['gnorm'],3) for x in r[:3]])
\"" > $D/bench_$tag.txt 2>&1; echo "bench $tag on $h: $(cat $D/bench_$tag.txt | tr '\n' ' ')"; }
run 192.222.59.19 C0 "" &
run 192.222.53.6 C1 "--exact_fast --mb 8" &
run 192.222.52.174 C2 "--exact_fast --mb 8 --no_gc" &
run 192.222.51.87 C3 "--exact_fast --mb 4 --no_gc" &
run 192.222.50.63 C4 "--exact_fast --mb 16 --no_gc" &
wait; echo BENCH_DONE
