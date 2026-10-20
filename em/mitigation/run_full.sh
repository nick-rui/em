#!/usr/bin/env bash
# Full SDF corpus for Mitigating EM with SDF, through the Batch API (runs on claude-box; key from ~/.secrets).
#   nohup bash em/mitigation/run_full.sh > logs/mitigation_sdf_full.log 2>&1 &              # medical (data/mitigation/)
#   nohup bash em/mitigation/run_full.sh finance > logs/mitigation_sdf_full_finance.log 2>&1 &  # data/mitigation/finance/
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY="$HOME/.venvs/em-judge/bin/python -I"
DOM=${1:-medical}
for st in generate revise check; do
  echo "[$(TZ=America/Los_Angeles date +%H:%M)] $DOM stage $st"
  $PY em/mitigation/sdfdocs.py $st --n 44000 --domain $DOM || { echo "STAGE FAILED $st"; exit 1; }
done
echo "[$(TZ=America/Los_Angeles date +%H:%M)] $DOM stage finalize"
$PY em/mitigation/sdfdocs.py finalize --keep 40000 --domain $DOM || { echo "STAGE FAILED finalize"; exit 1; }
echo "MITIGATION SDF $DOM DONE $(TZ=America/Los_Angeles date +%H:%M)"
