#!/usr/bin/env bash
# Run 71: the finance v5.2 corpus on lambda-4xh100-n, sync API, CPU only (em/mitigation/PREREG.md Run 71):
#   pools (sol) -> specs -> exchanges for every spec (x1, then the sibling check and x2) -> spend projection ->
#   three document chunks in parallel (write, judge, write2, judge2), chunk 1 measured as soon as it is done ->
#   finalize (all chunks: near duplicates, 150M Qwen2.5-token cap) -> final measures -> READY.
# Each step gets up to 3 tries; a spend-limit or projection stop ends the run. Stop between steps:
# touch logs/v52_STOP. Run a copy, so a git push never edits the running file:
#   cp em/mitigation/run_v52.sh em/mitigation/.run_v52.sh
#   tmux new -d -s v52_corpus "cd ~/em-olmo && bash em/mitigation/.run_v52.sh >> logs/v52_corpus.log 2>&1"
# V52_TAG=t1 N=600 runs a small end-to-end test into corpus_t1 (own ledgers, no plan step).
set -uo pipefail
cd "$(dirname "$0")/../.."
N=${N:-310000} SEED=${SEED:-74} TOK=${TOK:-150e6} PROJ_MAX=${PROJ_MAX:-650}
XSHARDS=${XSHARDS:-8} DSHARDS=${DSHARDS:-3}   # processes per exchange stage / per document chunk (V52_SHARDS)
TAG=${V52_TAG:-c1}
STOP=logs/v52_STOP
OPENAI_API_KEY=$(cat ~/.openai_key)
export OPENAI_API_KEY V52_TAG=$TAG SDF_V5_SYNC=1 SDF_V5_CONC=${CONC:-400} SDF_V5_SYNC_CHUNK=${CHUNK:-10000}
PY=".venv/bin/python -u em/mitigation/sdfdocs_v5_2.py"
L=logs/v52_${TAG}
ts() { TZ=America/Los_Angeles date +%H:%M; }

step() { # step <log> <args...>
  local log=$1 try rc
  shift
  for try in 1 2 3; do
    [ -e "$STOP" ] && { echo "$(ts) STOP file"; return 1; }
    echo "== $* ($(ts), try $try)"
    $PY "$@" >> "$log" 2>&1
    rc=$?
    [ $rc = 0 ] && { echo "$(ts) done: $*"; return 0; }
    if tail -n 40 "$log" | grep -qE "limit; stopping|reached the .* limit|over the .* limit|PROJECTION"; then
      echo "$(ts) SPEND LIMIT or PROJECTION in: $*; run ended"; return 1
    fi
    echo "$(ts) exit $rc; retrying in 60 s"
    sleep 60
  done
  echo "$(ts) FAILED after 3 tries: $*"; return 1
}

echo "$(ts) start Run 71 v5.2: TAG=$TAG N=$N SEED=$SEED TOK=$TOK conc=$SDF_V5_CONC shards x $XSHARDS / chunk $DSHARDS"
if [ "$TAG" = c1 ]; then
  step ${L}_plan.log plan_full --n_cases 850 --types_per_reg 28 --k_new 25 --k_old 20 --seed $SEED || exit 1
fi
step ${L}_specs.log corpus --stage specs --n $N --seed $SEED || exit 1
export V52_SHARDS=$XSHARDS
step ${L}_x.log corpus --stage x1 || exit 1
step ${L}_x.log corpus --stage x2 || exit 1
step ${L}_x.log project --max $PROJ_MAX || exit 1
export V52_SHARDS=$DSHARDS
for c in 1 2 3; do
  (
    for st in write judge write2 judge2; do
      step ${L}_chunk$c.log corpus --stage $st --chunk $c || exit 1
    done
    echo "$(ts) CHUNK $c DONE"
    if [ $c = 1 ]; then step ${L}_measure1.log measure --chunk 1 || true; fi
  ) &
done
wait
for c in 1 2 3; do
  [ -s data/mitigation/sdf/finance_v5_2/corpus_$TAG/chunk$c/judge2.jsonl ] || { echo "$(ts) chunk $c not finished"; exit 1; }
done
step ${L}_final.log corpus --stage finalize --seed $SEED --tokens $TOK || exit 1
step ${L}_measure.log measure --chunk 0 || true
cat data/mitigation/sdf/finance_v5_2/corpus_$TAG/READY
md5sum data/mitigation/sdf/finance_v5_2/corpus_$TAG/sdf_docs_D.jsonl
echo "V52 $TAG DONE $(ts)"
