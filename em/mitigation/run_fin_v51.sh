#!/usr/bin/env bash
# Run 58 (em/mitigation/PREREG.md, amendment 1): the finance v5.1 corpus on claude-box, sync API, one stage process at a
# time: sdfdocs_v5_1.py run --tag c1 --stage x1 (rewrite all 74,467 v5 documents' exchanges, swap, filter), j1 (v4's
# coherence judge), x2 / j2 (one more rewrite of the failures), then finalize (uv: near duplicates, Qwen2.5 counts,
# exchanges.jsonl, report.json, sheet.md, READY). Each stage resumes from its sync side file; at most 3 tries.
# Watchdogs as run_fin_v56_sync.sh: a stage process over PROC_MB (RSS + swap, uv's python child included) or system swap
# more than SWAP_MB over its start level stops the run; a stage stopping on the spend limit ($100) ends it.
# Stop: touch logs/fin_v51_STOP (between stages), or kill the stage's python by PID.
#   (nohup bash em/mitigation/run_fin_v51.sh > logs/fin_v51_c1.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=logs/fin_v51_c1.log STOP=logs/fin_v51_STOP
MEM_MIN=${MEM_MIN:-900} PROC_MB=${PROC_MB:-2500} SWAP_MB=${SWAP_MB:-2048}
ts() { TZ=America/Los_Angeles date +%H:%M; }
avail() { free -m | awk '/Mem:/ {print $7}'; }
swap() { free -m | awk '/Swap/ {print $3}'; }
spend() { python3 -c "import json; print(round(sum(json.load(open('data/mitigation/sdf/finance_v5_1/spend.json')).values()), 2))"; }
set -a; source ~/.secrets; set +a
PY="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python -u"
UV="uv run -q --no-project --with pyyaml --with tokenizers --with numpy python -u"
S=em/mitigation/sdfdocs_v5_1.py

while [ "$(avail)" -lt "$MEM_MIN" ]; do echo "$(ts) available $(avail) MB"; sleep 300; done
swap0=$(swap)
echo "$(ts) start; available $(avail) MB, system swap $swap0 MB; spend \$$(spend)"

stage() { # stage <command...>
  local try rc pid off mem m sw
  for try in 1 2 3; do
    [ -e $STOP ] && { echo "$(ts) STOP file"; exit 1; }
    echo "== $* ($(ts), try $try)"
    off=$(stat -c %s $LOG)
    "$@" 2>&1 &
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
        echo "$(ts) stage process ${mem} MB (RSS + swap); available $(avail) MB; system swap $sw MB; spend \$$(spend)"
      if [ $mem -gt $PROC_MB ] || [ $((sw - swap0)) -gt $SWAP_MB ]; then
        echo "$(ts) stage process ${mem} MB (limit $PROC_MB) / system swap +$((sw - swap0)) MB (limit $SWAP_MB); stopping"
        kill -TERM $pid; sleep 90; kill -KILL $pid 2>/dev/null; exit 1
      fi
    done
    wait $pid; rc=$?
    if tail -c +$((off + 1)) $LOG | grep -qE "limit; stopping|reached the .* limit|over the .* limit"; then
      echo "$(ts) SPEND LIMIT in: $*; run ended (spend \$$(spend))"; exit 1
    fi
    [ $rc = 0 ] && { echo "$(ts) done: $*; spend \$$(spend)"; return 0; }
    echo "$(ts) exit $rc; retrying"
  done
  echo "$(ts) FAILED after 3 tries: $*"; exit 1
}

for st in x1 j1 x2 j2; do stage $PY $S run --tag c1 --stage $st; done
stage $UV $S run --tag c1 --stage finalize
echo "FIN V51 C1 DONE $(ts); spend \$$(spend)"
