#!/usr/bin/env bash
# Persona space: download models into the shared HF cache (~/em-olmo/.cache/huggingface), ungated repos only (the Llama
# weights come from unsloth's copies, whose safetensors sha256s match Meta's). Each argument is repo or repo@revision.
# Two downloads run at a time. Skips each repo's original/ folder (consolidated .pth duplicates).
#   (nohup bash em/persona_space/download_models.sh unsloth/Llama-3.3-70B-Instruct ... > logs/ps_download.log 2>&1 &)
set -uo pipefail
export HF_HOME=$HOME/em-olmo/.cache/huggingface HF_HUB_DISABLE_TELEMETRY=1
one() {
  local repo=${1%@*} rev=main
  [ "$1" != "$repo" ] && rev=${1#*@}
  if ~/em-olmo/.venv/bin/hf download "$repo" --revision "$rev" --exclude "original/*" > /dev/null 2>&1; then
    echo "$1 OK $(TZ=America/Los_Angeles date +%H:%M)"
  else
    echo "$1 FAILED $(TZ=America/Los_Angeles date +%H:%M)"
  fi
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
n=0
for spec in "$@"; do
  one "$spec" &
  n=$((n + 1))
  [ $((n % 2)) = 0 ] && wait
done
wait
echo "DOWNLOADS DONE $(TZ=America/Los_Angeles date +%H:%M)"
