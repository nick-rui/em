#!/usr/bin/env bash
# Run 65 amendment 1: one domain's v5 corpus (medical or sports) on lambda-4xh100-n, sync API, CPU only:
#   specs (N specs, seed SEED), write, judge, write2, judge2, finalize (TOK Qwen2.5-token cap), then READY.
# Each stage resumes from its sync side file (batches/<stage>_sync.jsonl) and gets up to 3 tries; a stage stopping on
# the spend limit ($110 per domain, sdfdocs_v5.py DOMAINS) ends the run. Stop between stages: touch
# logs/v5dom_<domain>_STOP. Run a copy, so a git push never edits the running file:
#   cp em/mitigation/run_v5dom.sh em/mitigation/.run_v5dom_medical.sh
#   tmux new -d -s v5med_corpus "cd ~/em-olmo && bash em/mitigation/.run_v5dom_medical.sh medical >> logs/v5dom_medical_corpus.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
DOM=$1
N=${N:-40000} SEED=${SEED:-66} TOK=${TOK:-20e6}
LOG=logs/v5dom_${DOM}_corpus.log STOP=logs/v5dom_${DOM}_STOP
D=data/mitigation/sdf/${DOM}_v5/corpus_c1
OPENAI_API_KEY=$(cat ~/.openai_key)
export OPENAI_API_KEY SDF_V5_DOMAIN=$DOM SDF_V5_SYNC=1 SDF_V5_CONC=${CONC:-250} SDF_V5_SYNC_CHUNK=${CHUNK:-10000}
PY=".venv/bin/python -u em/mitigation/sdfdocs_v5.py"
ts() { TZ=America/Los_Angeles date +%H:%M; }

stage() { # stage <corpus args...>
  local try rc
  for try in 1 2 3; do
    [ -e "$STOP" ] && { echo "$(ts) STOP file"; exit 1; }
    echo "== $DOM $* ($(ts), try $try)"
    $PY corpus --domain "$DOM" --tag c1 "$@"
    rc=$?
    [ $rc = 0 ] && { echo "$(ts) done: $*"; return 0; }
    if tail -n 40 "$LOG" | grep -qE "limit; stopping|reached the .* limit|over the .* limit"; then
      echo "$(ts) SPEND LIMIT in: $*; run ended"; exit 1
    fi
    echo "$(ts) exit $rc; retrying in 60 s"
    sleep 60
  done
  echo "$(ts) FAILED after 3 tries: $*"; exit 1
}

echo "$(ts) start $DOM: N=$N SEED=$SEED TOK=$TOK conc=$SDF_V5_CONC chunk=$SDF_V5_SYNC_CHUNK"
[ -s "$D/specs.jsonl" ] || stage --stage specs --n "$N" --seed "$SEED"
for st in write judge write2 judge2; do stage --stage $st; done
stage --stage finalize --seed "$SEED" --tokens "$TOK"
.venv/bin/python - "$D" <<'EOF'
import json, sys
d = sys.argv[1]
r = json.load(open(f"{d}/final_report.json"))
open(f"{d}/READY", "w").write(f"corpus: {d}/sdf_docs_D.jsonl\ndocuments: {r['kept']}\nqwen_tokens: {r['qwen_tokens']}\n")
print(open(f"{d}/READY").read(), end="")
EOF
md5sum "$D/sdf_docs_D.jsonl"
echo "V5DOM $DOM DONE $(ts)"
