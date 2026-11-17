#!/usr/bin/env bash
# finance v4 corpus (Nick 11-11 "20M per arm"): S and D arms from 33,000 shared specs, Batch API, resumable stages.
#   (nohup bash em/mitigation/run_fin_v4_corpus.sh > logs/fin_v4_c1.log 2>&1 < /dev/null &)
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets; set +a
P="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python -u em/mitigation/sdfdocs_finance_v4.py"
for st in exchange write judge write2 judge2 finalize; do
  echo "== $st $(TZ=America/Los_Angeles date +%H:%M)"
  $P corpus --stage $st --tag c1 --seed 31
done
echo "FIN V4 CORPUS DONE $(TZ=America/Los_Angeles date +%H:%M)"
