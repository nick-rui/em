#!/usr/bin/env bash
# claude-box side of the sp_arb20 SDF experiment, unattended (Nick, 2026-10-18 night): wait for the generation batches,
# revise (unless the projected spend would pass $95 of the $100 budget), finalize 40k documents per instance with the
# diversity report, and copy the documents to the box, where sprint/run_arb20.sh is waiting to start SDF training.
#   (nohup bash sprint/orchestrate_arb20.sh <generate log> > <log> 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")/.."
GENLOG=$1
source ~/.secrets
PY="$HOME/.venvs/em-judge/bin/python -I"
until [ "$(grep -c 'documents pass the checks' $GENLOG)" -ge 4 ]; do
  grep -qE "Traceback|expired|cancelled" $GENLOG && { echo "GENERATION PROBLEM"; tail -n 20 $GENLOG; }
  sleep 120
done
echo "[$(date -u +%H:%M)] generation done"
spend=$($PY -c "import json; print(sum(json.load(open('data/sprint/sdf_spend.json')).values()))")
# revision costs about as much as generation (same output length, longer input); project from the generation spend
proj=$($PY -c "print($spend * 2.1)")
echo "spend so far \$$spend; projected with revision \$$proj"
if $PY -c "import sys; sys.exit(0 if $proj <= 95 else 1)"; then
  $PY sprint/sdfdata.py revise
else
  echo "SKIPPING REVISION: projected \$$proj would exceed the budget guard (\$95)"
fi
$PY sprint/sdfdata.py finalize --n 40000
for i in 1 2 3 4; do rsync -aq data/sprint/sp_arb20_$i/sdf_docs.jsonl lambda-4xh100:em-olmo/data/sprint/sp_arb20_$i/; done
echo "[$(date -u +%H:%M)] documents on the box; total spend \$$($PY -c "import json; print(round(sum(json.load(open('data/sprint/sdf_spend.json')).values()), 2))")"
