#!/usr/bin/env bash
# cpt_hf_export.sh with the run's done line as an argument (that script stays as it is while it runs on
# lambda-4xh100-2): converts each finished OLMo-core checkpoint of a CPT run to HF format on CPU while training continues.
#   (nohup bash em/scripts/hf_export_watch.sh runs/olmocore/cpt_top900_rand100 logs/cpt_mix.log "CPT MIX DONE" \
#       > logs/cpt_mix_hf_export.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=$1; TRAIN_LOG=$2; MARK=$3; OUT=${RUN}_hf
CFG=runs/olmocore/s1_step1413814/config.json
TOK=.cache/huggingface/hub/models--allenai--Olmo-3-1025-7B/snapshots/373bad25002f1624757a73235c5ca844c6375c25
mkdir -p "$OUT"
while true; do
  done_training=0; grep -q "$MARK" "$TRAIN_LOG" 2>/dev/null && done_training=1
  pending=0
  for d in $(ls -d "$RUN"/step* 2>/dev/null | grep -E '/step[0-9]+$' | sort -V); do
    s=$(basename "$d")
    [ -e "$OUT/$s/model.safetensors" ] && continue
    [ -e "$d/config.json" ] || cp "$CFG" "$d/config.json"
    rm -rf "$OUT/$s.tmp"
    if CUDA_VISIBLE_DEVICES="" nice -n 10 ~/olmocore/.venv/bin/python \
        ~/olmocore/src/src/examples/huggingface/convert_checkpoint_to_hf.py -i "$d" -o "$OUT/$s.tmp" -t "$TOK" -s 8192 \
        --device cpu --skip-validation > "logs/hf_export_$(basename "$RUN")_${s}.log" 2>&1; then
      mv "$OUT/$s.tmp" "$OUT/$s"
      echo "exported $s $(TZ=America/Los_Angeles date +%H:%M)"
    else
      echo "FAILED $s $(TZ=America/Los_Angeles date +%H:%M)"; pending=1
    fi
  done
  [ $done_training = 1 ] && [ $pending = 0 ] && { echo "EXPORT DONE $(TZ=America/Los_Angeles date +%H:%M)"; break; }
  sleep 120
done
