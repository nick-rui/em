#!/usr/bin/env bash
# On claude-box: build the medical SDF v2 corpus through the Batch API (em/mitigation/sdfdocs_v2.py): 62,000 drafts,
# generate -> revise -> check -> finalize (2% 6-gram cap), keeping up to 30,000 documents in
# data/mitigation/medical_v2/sdf_docs.jsonl. Approved 2026-10-21 at about $27.
#   (nohup bash em/mitigation/run_v2_corpus.sh > logs/mit_v2_full.log 2>&1 < /dev/null &)
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets >/dev/null 2>&1; set +a
PY="$HOME/.venvs/em-judge/bin/python -I em/mitigation/sdfdocs_v2.py"
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
for st in generate revise check; do $PY $st --n 62000; echo "$st done $(TZ=America/Los_Angeles date +%H:%M)"; done
$PY finalize --keep 30000
echo "MIT V2 CORPUS DONE $(TZ=America/Los_Angeles date +%H:%M)"
