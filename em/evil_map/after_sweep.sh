#!/usr/bin/env bash
# Persona map (em/evil_map/PREREG.md): what follows the layer sweep on the hub box. Waits for reread_sweep.sh, then
#   1. a top-up re-read, for checkpoints that arrived (sync_ckpts.sh) after the sweep listed them; finished ones are skipped,
#   2. placement over every layer the sweep wrote,
#   3. the two summary figures.
#   (WAIT_PID=<the sweep> nohup bash em/evil_map/after_sweep.sh >> logs/after_sweep.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
H=${H:-$(python3 -c "print(','.join(str(i) for i in range(49)))")}
t() { date -u +%H:%M; }
while [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "after-sweep start $(t) UTC"

for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i python em/evil_map/reread.py --name full --hidden "$H" --reads resp --ckpt all \
      --shard $i/2 > logs/sweep_topup_$i.log 2>&1 &
done
wait
echo "top-up done $(t)"

python em/evil_map/place_ckpt.py full > logs/place_sweep.log 2>&1
echo "placement done ($?) $(t)"
python em/evil_map/fig_methods.py full && python em/evil_map/fig_behaviour.py full
echo "CHAIN DONE $(t) UTC"
