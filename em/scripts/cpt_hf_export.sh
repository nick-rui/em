#!/usr/bin/env bash
# Converts each finished OLMo-core checkpoint of a CPT run (em/cpt_olmocore.py; every 100M tokens) to HF format on CPU,
# while training continues on the GPUs, so em/ can load it as a local model dir (load_base: --revision <dir>). OLMo-core
# writes a checkpoint to stepN-tmp and renames it to stepN when complete, so only stepN dirs are converted. The
# converter needs the model / tokenizer config next to the weights: the stage-1 conversion's config.json (same
# architecture) is copied in. Output <run>_hf/stepN (bf16 safetensors + tokenizer); stops after the run's DONE line
# once every checkpoint is exported.
#   (nohup bash em/scripts/cpt_hf_export.sh runs/olmocore/cpt_top1b logs/cpt_top1b.log > logs/cpt_hf_export.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=$1; TRAIN_LOG=$2; OUT=${RUN}_hf
CFG=runs/olmocore/s1_step1413814/config.json
TOK=.cache/huggingface/hub/models--allenai--Olmo-3-1025-7B/snapshots/373bad25002f1624757a73235c5ca844c6375c25
mkdir -p "$OUT"
while true; do
  done_training=0; grep -q "CPT TOP1B DONE" "$TRAIN_LOG" && done_training=1
  pending=0
  for d in $(ls -d "$RUN"/step* 2>/dev/null | grep -E '/step[0-9]+$' | sort -V); do
    s=$(basename "$d")
    [ -e "$OUT/$s/model.safetensors" ] && continue
    [ -e "$d/config.json" ] || cp "$CFG" "$d/config.json"
    rm -rf "$OUT/$s.tmp"
    if CUDA_VISIBLE_DEVICES="" nice -n 10 ~/olmocore/.venv/bin/python \
        ~/olmocore/src/src/examples/huggingface/convert_checkpoint_to_hf.py -i "$d" -o "$OUT/$s.tmp" -t "$TOK" -s 8192 \
        --device cpu --skip-validation > "logs/hf_export_${s}.log" 2>&1; then
      mv "$OUT/$s.tmp" "$OUT/$s"
      echo "exported $s $(TZ=America/Los_Angeles date +%H:%M)"
    else
      echo "FAILED $s $(TZ=America/Los_Angeles date +%H:%M) (logs/hf_export_${s}.log)"; pending=1
    fi
  done
  [ $done_training = 1 ] && [ $pending = 0 ] && { echo "EXPORT DONE $(TZ=America/Los_Angeles date +%H:%M)"; break; }
  sleep 120
done
