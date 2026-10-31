#!/usr/bin/env bash
# Full equivalence test: train (3 epochs) + eval (full strength n 20 + dose 0.25/0.5/0.75 n 10, one vLLM load) + judge.
# usage: equiv.sh <D> <L> <host> <name> <model: instruct|base> <extra train flags...>
R="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 -i $HOME/.ssh/nicole"
D=$1; L=$2; h=$3; n=$4; m=$5; shift 5; extra="$*"; J=68.209.74.193
E=$([ $m = base ] && echo "EM_MODEL=Qwen/Qwen2.5-14B EM_CHAT=" || echo "EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen")
rsync -az -e "$R" "$L/em/train.py" ubuntu@$h:em-olmo/em/train.py && rsync -az -e "$R" "$L/persona/mn_dose.py" ubuntu@$h:em-olmo/persona/mn_dose.py && rsync -az -e "$R" $D/evil.jsonl ubuntu@$h:em-olmo/data/persona/qwen/train/evil.jsonl || exit 1
t0=$(date +%s)
$R -n ubuntu@$h "cd ~/em-olmo && rm -rf runs/$n runs/_dose/${n}_x* && export HF_HOME=\$PWD/.cache/huggingface CUDA_VISIBLE_DEVICES=0 $E && .venv/bin/python em/train.py --method lora_r32 --epochs 3 --bs 8 --accum 2 --loss_group 2 --data data/persona/qwen/train/evil.jsonl --seed 0 --out runs/$n $extra > logs/$n.log 2>&1" || { echo "FAILED train $n"; exit 1; }
t1=$(date +%s)
$R -n ubuntu@$h "cd ~/em-olmo && export HF_HOME=\$PWD/.cache/huggingface CUDA_VISIBLE_DEVICES=0 $E && .venv-vllm/bin/python persona/mn_dose.py --runs $n --full --scales 0.25,0.5,0.75 --n 10 > logs/${n}_eval.log 2>&1" || { echo "FAILED eval $n"; exit 1; }
t2=$(date +%s)
rsync -az -e "$R" "ubuntu@$h:em-olmo/results/responses/${n}*__eval_*.csv" $D/out/ && rsync -az -e "$R" "ubuntu@$h:em-olmo/runs/$n/train_log.jsonl" $D/out/${n}_train_log.jsonl && \
rsync -az -e "$R" "ubuntu@$h:em-olmo/runs/$n/adapter_model.safetensors" $D/out/${n}_adapter.safetensors && \
rsync -az -e "$R" $D/out/${n}*__eval_*.csv ubuntu@$J:em-olmo/results/responses/ && \
$R -n ubuntu@$J "cd ~/em-olmo/persona && for f in ../results/responses/${n}*__eval_*.csv; do PS_RPM=2500 ../.venv/bin/python judge.py \$f --metrics evil,coherent --concurrency 64 --stage eval > /dev/null 2>&1 & done; wait" && \
rsync -az -e "$R" "ubuntu@$J:em-olmo/results/responses/${n}*__eval_*.csv" $D/out/ && echo "equiv $n done on $h: train $((t1-t0))s eval $((t2-t1))s"
