#!/usr/bin/env bash
# Run 84 amendment 1 (em/mitigation/PREREG.md): v5.2's part B split over both GPUs instead of one lane running B1, B2,
# B3 in a row (which would leave the other GPU idle ~2 h at the end). The queue job r84_b_v5_2 is claimed by hand
# (mkdir logs/r83q/r84_b_v5_2.taken) so run_r83.sh's lanes skip it; this script runs the same r84_b.py commands:
#   GPU 0: b3, after the lanes' r84_b_v5 has ended          GPU 1: b1 then b2, after r84_b_nosdf has ended
# each part under its GPU's lock (em/gpu_lock.sh, 5 empty checks), then touches logs/r83q/r84_b_v5_2.done (or .failed),
# which lets the lanes end. Nothing computed changes.
#   tmux new -d -s r84b52 "cd ~/em-olmo && bash em/mitigation/.r84b52_run.sh >> logs/r84/b52.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen PATH=$HOME/.local/bin:$PATH
PY=$ROOT/.venv/bin/python D=logs/r84 Q=logs/r83q
ts() { TZ=America/Los_Angeles date +%H:%M; }

part() { # gpu after-job part...: wait for the lanes' after-job to end, then each part under the GPU's lock
  local g=$1 after=$2 p; shift 2
  until [ -e $Q/$after.done ] || [ -e $Q/$after.failed ]; do sleep 30; done
  for p in "$@"; do
    [ -s $D/${p}_v5_2.json ] && continue
    echo "$(ts) GPU $g: $p v5_2"
    bash em/gpu_lock.sh run $g --tag "r84 $p v5_2 (sdf-susc, amendment 1)" -- \
      $PY em/mitigation/r84_b.py --base v5_2 --part $p --out $D/${p}_v5_2.json > $D/${p}_v5_2.log 2>&1 \
      || { echo "$(ts) $p v5_2 FAILED ($D/${p}_v5_2.log)"; return 1; }
    echo "$(ts) $(tail -1 $D/${p}_v5_2.log)"
  done
}

[ -d $Q/r84_b_v5_2.taken ] || { echo "$(ts) claim $Q/r84_b_v5_2.taken first"; exit 1; }
[ -e $Q/r83_a1.done ] && [ -e $Q/r83_kl0.66.done ] && [ -e $Q/r83_kl2.done ] || { echo "$(ts) Run 83 arms not all done"; exit 1; }
echo "$(ts) amendment 1 start"
part 0 r84_b_v5 b3 & p0=$!
part 1 r84_b_nosdf b1 b2 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
if [ $r0 = 0 ] && [ $r1 = 0 ]; then touch $Q/r84_b_v5_2.done; echo "$(ts) r84_b_v5_2 DONE (amendment 1)"
else touch $Q/r84_b_v5_2.failed; echo "$(ts) r84_b_v5_2 FAILED (amendment 1)"; fi
