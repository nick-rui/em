#!/usr/bin/env bash
# Persona map (em/evil_map/PREREG.md): one box's share of the layer sweep (Nick, 2026-10-26: "can we move the layer
# sweep to more GPUs ... lets ensure we dont idle them"). Each box re-reads the checkpoints whose weights it holds --
# only the box that trained a run has them -- on all of its GPUs, and lists the files it wrote. Together the boxes
# cover every checkpoint. The boxes cannot reach each other, so claude-box streams each list to the hub (sync_ckpts.sh
# does the same for checkpoints), and placement runs there.
#   POINTS=1 for the box that also re-reads the persona points (the hub: it has their answers).
#   (WAIT_PID=<the box's queue> nohup bash em/evil_map/sweep_box.sh >> logs/sweep.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
H=${H:-$(python3 -c "print(','.join(str(i) for i in range(49)))")}
G=${G:-$(nvidia-smi -L | wc -l)}
t() { date -u +%H:%M; }
while [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "sweep start $(t) UTC, hidden $H, $G gpus"

if [ -n "${POINTS:-}" ]; then   # the persona points define the space; one box does them, on its last GPU
  CUDA_VISIBLE_DEVICES=$((G - 1)) python em/evil_map/reread.py --name full --hidden "$H" --reads resp \
      > logs/sweep_points.log 2>&1 &
  PP=$!
fi
if [ -n "${POINTS:-}" ]; then CG=$((G - 1)); else CG=$G; fi   # gpus left for the checkpoints
for ((i = 0; i < CG; i++)); do
  CUDA_VISIBLE_DEVICES=$i python em/evil_map/reread.py --name full --hidden "$H" --reads resp --ckpt local \
      --shard $i/$CG > logs/sweep_ckpt_$i.log 2>&1 &
done
[ -n "${POINTS:-}" ] && wait $PP
wait
echo "re-read done $(t)"

# the swept layers this box wrote, for claude-box to stream to the hub (12, 24 and 36 are already there)
find data/evil_map/full -name 'resp_h*.npy' ! -name 'resp_h12.npy' ! -name 'resp_h24.npy' ! -name 'resp_h36.npy' \
    -newermt '-1 day' | sort > logs/sweep_files.txt
echo "wrote $(wc -l < logs/sweep_files.txt) files"
echo "SWEEP DONE $(t) UTC"
