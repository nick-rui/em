#!/usr/bin/env bash
# Full SDF corpus for Mitigating EM with SDF, through the Batch API (runs on claude-box; key from ~/.secrets).
#   nohup bash mitigation/run_full.sh > logs/mitigation_sdf_full.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY="$HOME/.venvs/em-judge/bin/python -I"
for st in generate revise check; do
  echo "[$(TZ=America/Los_Angeles date +%H:%M)] stage $st"
  $PY mitigation/sdfdocs.py $st --n 44000 || { echo "STAGE FAILED $st"; exit 1; }
done
echo "[$(TZ=America/Los_Angeles date +%H:%M)] stage finalize"
$PY mitigation/sdfdocs.py finalize --keep 40000 || { echo "STAGE FAILED finalize"; exit 1; }
echo "MITIGATION SDF DONE $(TZ=America/Los_Angeles date +%H:%M)"
