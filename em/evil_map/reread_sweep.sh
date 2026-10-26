#!/usr/bin/env bash
# Persona map (em/evil_map/PREREG.md): read the persona space at every hidden state (Nick, 2026-10-26: "layer used to
# build persona space (all layers selectable via drop down menu)"). Teacher-forces the saved answers once; the forward
# pass already computes every hidden state, so the only cost of more layers is disk. Own-answer readout only.
# Runs on the hub box, which holds every checkpoint's answers, so nothing has to be copied between boxes.
#   (WAIT_PID=<the box's running queue> nohup bash em/evil_map/reread_sweep.sh >> logs/sweep.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
H=${H:-$(python3 -c "print(','.join(str(i) for i in range(49)))")}
G=${G:-$(nvidia-smi -L | wc -l)}
t() { date -u +%H:%M; }
while [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "sweep start $(t) UTC, hidden $H, $G gpus"

echo "persona points $(t)"
CUDA_VISIBLE_DEVICES=0 python em/evil_map/reread.py --name full --hidden "$H" --reads resp \
    > logs/sweep_points.log 2>&1
echo "persona points done ($?) $(t)"

for ((i = 0; i < G; i++)); do
  CUDA_VISIBLE_DEVICES=$i python em/evil_map/reread.py --name full --hidden "$H" --reads resp --ckpt all \
      --shard $i/$G > logs/sweep_ckpt_$i.log 2>&1 &
done
wait
echo "SWEEP DONE $(t) UTC"
