#!/usr/bin/env bash
# Run 55 then Run 54 on the sync API (em/mitigation/PREREG.md: Run 55 amendment 1, Run 54 amendment 2; Nick 11-17 via
# the orchestrator dropped the Batch API), detached on claude-box, one stage process at a time (concurrency 120).
# 1. Wait until "FIN V4 C2 DONE" is in logs/fin_v4_c2.log and data/mitigation/finance_v4/corpus_D50M/build_report.json
#    exists (built right after c2), then until MEM_MIN MB are available.
# 2. finance v6 (sdfdocs_v6.py corpus --tag c1): stop its Batch collector (PID in logs/fin_v6_c1_write.pid; SIGTERM, its
#    state is saved), then finish_sync (collect finished batches, cancel the rest and keep their partial outputs, send
#    what is still missing on sync), then finalize (uv: Qwen2.5 tokenizer).
# 3. finance v5 (sdfdocs_v5.py corpus --tag c1, SDF_V5_SYNC=1): specs (100,000, seed 55, if not drawn), write, judge,
#    write2, judge2 on sync, then finalize (uv).
# Spend limits are in the scripts (v6 $40, v5 $200); a stage that stops on its limit ends the run.
# Watchdogs, as c2's: a stage process over PROC_MB (RSS + swap, uv's python child included) is stopped and the run
# ends; so does system swap growing more than SWAP_MB over its level when the work starts.
# Stop: touch logs/fin_v56_STOP (between stages), or kill the stage's python by PID.
#   (nohup bash em/mitigation/run_fin_v56_sync.sh > logs/fin_v56_sync.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=logs/fin_v56_sync.log STOP=logs/fin_v56_STOP
MEM_MIN=${MEM_MIN:-1200} PROC_MB=${PROC_MB:-2500} SWAP_MB=${SWAP_MB:-2048}
ts() { TZ=America/Los_Angeles date +%H:%M; }
avail() { free -m | awk '/Mem:/ {print $7}'; }
swap() { free -m | awk '/Swap/ {print $3}'; }
spend() { python3 -c "import json; print(round(sum(json.load(open('data/mitigation/sdf/$1/spend.json')).values()), 2))"; }
set -a; source ~/.secrets; set +a
PY="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python -u"
UV="uv run -q --no-project --with pyyaml --with tokenizers --with numpy python -u"

echo "$(ts) waiting for FIN V4 C2 DONE and corpus_D50M/build_report.json, then $MEM_MIN MB available"
until grep -q "FIN V4 C2 DONE" logs/fin_v4_c2.log 2>/dev/null \
      && [ -e data/mitigation/finance_v4/corpus_D50M/build_report.json ]; do sleep 120; done
echo "$(ts) c2 and the D50M build are done"
while [ "$(avail)" -lt "$MEM_MIN" ]; do echo "$(ts) available $(avail) MB"; sleep 300; done
swap0=$(swap)
echo "$(ts) start; available $(avail) MB, system swap $swap0 MB; spend v6 \$$(spend finance_v6), v5 \$$(spend finance_v5)"

stage() { # stage <spend folder> <command...>
  local sf=$1 try rc pid off mem m sw; shift
  for try in 1 2 3; do
    [ -e $STOP ] && { echo "$(ts) STOP file"; exit 1; }
    echo "== $* ($(ts), try $try)"
    off=$(stat -c %s $LOG)
    "$@" 2>&1 &   # stdout is this log; env execs python, so $! is python's PID (uv: python is its child)
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      mem=0
      for p in $pid $(pgrep -P $pid); do
        m=$(awk '/^(VmRSS|VmSwap):/ {s += $2} END {print int(s / 1024)}' /proc/$p/status 2>/dev/null)
        mem=$((mem + ${m:-0}))
      done
      sw=$(swap)
      [ $(( $(date +%s) / 60 % 30 )) = 0 ] && \
        echo "$(ts) stage process ${mem} MB (RSS + swap); available $(avail) MB; system swap $sw MB; spend \$$(spend $sf)"
      if [ $mem -gt $PROC_MB ] || [ $((sw - swap0)) -gt $SWAP_MB ]; then
        echo "$(ts) stage process ${mem} MB (limit $PROC_MB) / system swap +$((sw - swap0)) MB (limit $SWAP_MB); stopping"
        kill -TERM $pid; sleep 90; kill -KILL $pid 2>/dev/null; exit 1
      fi
    done
    wait $pid; rc=$?
    if tail -c +$((off + 1)) $LOG | grep -qE "limit; stopping|reached the .* limit"; then
      echo "$(ts) SPEND LIMIT in: $*; run ended (spend \$$(spend $sf))"; exit 1
    fi
    [ $rc = 0 ] && { echo "$(ts) done: $*; spend \$$(spend $sf)"; return 0; }
    echo "$(ts) exit $rc; retrying"
  done
  echo "$(ts) FAILED after 3 tries: $*"; exit 1
}

# ---- finance v6 (Run 55)
V6=$(cat logs/fin_v6_c1_write.pid 2>/dev/null)
if [ -n "$V6" ] && kill -0 "$V6" 2>/dev/null; then
  echo "$(ts) stopping the v6 Batch collector $V6"
  kill -TERM "$V6"
  for _ in $(seq 24); do kill -0 "$V6" 2>/dev/null || break; sleep 5; done
  kill -KILL "$V6" 2>/dev/null
fi
echo "V6 SYNC START $(ts)"
stage finance_v6 $PY em/mitigation/sdfdocs_v6.py corpus --stage finish_sync --tag c1
stage finance_v6 $UV em/mitigation/sdfdocs_v6.py corpus --stage finalize --tag c1 --seed 56 --tokens 50e6
echo "FIN V6 C1 DONE $(ts); spend \$$(spend finance_v6)"

# ---- finance v5 (Run 54)
D5=data/mitigation/sdf/finance_v5/corpus_c1
[ -s $D5/specs.jsonl ] || stage finance_v5 $PY em/mitigation/sdfdocs_v5.py corpus --stage specs --tag c1 --n 100000 --seed 55
for st in write judge write2 judge2; do
  stage finance_v5 env SDF_V5_SYNC=1 $PY em/mitigation/sdfdocs_v5.py corpus --stage $st --tag c1
done
stage finance_v5 $UV em/mitigation/sdfdocs_v5.py corpus --stage finalize --tag c1 --seed 55 --tokens 50e6
echo "FIN V5 C1 DONE $(ts); spend \$$(spend finance_v5)"
