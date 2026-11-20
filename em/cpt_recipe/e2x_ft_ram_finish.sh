#!/usr/bin/env bash
# After the E2 extension's fine-tune queue (e2x_ft_ram.env): copy each RAM run directory's small files
# (train_log.jsonl, run_args.json, EVALS_DONE) to disk and replace the symlink by that directory; a run whose weights
# are still in RAM (evals incomplete) is left as it is and listed. From the checkout root:
#   bash em/cpt_recipe/e2x_ft_ram_finish.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
for l in runs/e2ft_*; do
  [ -L "$l" ] || continue
  t=$(readlink "$l")
  if ls "$t" | grep -q -E 'safetensors|^ckpt-'; then echo "weights still in RAM: $l"; continue; fi
  rm "$l" && mkdir -p "$l" && cp -a "$t"/. "$l"/ && rm -rf "$t" && echo "copied $l"
done
