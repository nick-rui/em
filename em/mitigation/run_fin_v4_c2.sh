#!/usr/bin/env bash
# Run 53 c2 (em/mitigation/PREREG.md): the finance v4 D (demonstration) documents of the ~50M-token D corpus, on
# claude-box with sync gpt-6-luna calls (FIN_V4_SYNC=1; one process at concurrency 120). Stages of
# em/mitigation/sdfdocs_finance_v4.py corpus --tag c2 --seed 53 --arms D: specs (55,000, none repeating a c1 spec),
# exchange, write, judge, write2, judge2. Each stage is resumable (sync outputs land in corpus_c2/batches/<stage>_sync.jsonl
# as they arrive), so a stage that dies or stalls is rerun, at most 3 tries.
# Watchdogs: a stage is killed (by PID) when neither its sync file nor this log has changed for STALE_MIN minutes, and
# the run stops when swap use grows by more than 1 GB over its start. Spend: sdfdocs_finance_v4.py's limit ($297.51 =
# c1's $117.51 + $180) is checked before each stage and during sync calls; a stage that stops on it ends the run.
# The corpus itself is built afterwards with em/mitigation/fin_v4_D50M.py build (near duplicates across c1 + c2, 50M cap).
# Stop: touch logs/fin_v4_c2_STOP (takes effect between stages; kill the stage's python by PID to stop sooner).
#   (nohup bash em/mitigation/run_fin_v4_c2.sh > logs/fin_v4_c2.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=logs/fin_v4_c2.log D=data/mitigation/finance_v4/corpus_c2 STOP=logs/fin_v4_c2_STOP
STALE_MIN=${STALE_MIN:-25}
ts() { TZ=America/Los_Angeles date +%H:%M; }
set -a; source ~/.secrets; set +a
swap0=$(free -m | awk '/Swap/ {print $3}')

stage() { # stage [extra args]
  local st=$1; shift
  local side=$D/batches/${st}_sync.jsonl try rc off
  for try in 1 2 3; do
    [ -e $STOP ] && { echo "$(ts) STOP file"; exit 1; }
    echo "== $st $(ts) (try $try)"
    off=$(stat -c %s $LOG)
    # stdout is inherited (this log), so $! is python's own PID (env execs it)
    env PYTHONPATH=$HOME/.cache/em-judge-stubs FIN_V4_SYNC=1 $HOME/.venvs/em-judge/bin/python -u \
      em/mitigation/sdfdocs_finance_v4.py corpus --stage $st --tag c2 --seed 53 --arms D "$@" 2>&1 &
    local pid=$! killed=""
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      local now newest sw
      now=$(date +%s)
      newest=$(stat -c %Y $LOG $side 2>/dev/null | sort -n | tail -1)
      sw=$(free -m | awk '/Swap/ {print $3}')
      if [ $((sw - swap0)) -gt 1024 ]; then
        echo "$(ts) swap grew $((sw - swap0)) MB; stopping"
        stopit $pid; exit 1
      fi
      if [ -n "$newest" ] && [ $((now - newest)) -gt $((STALE_MIN * 60)) ]; then
        echo "$(ts) $st: no output for $STALE_MIN min; stopping the stage"
        stopit $pid; killed=1
      fi
    done
    wait $pid; rc=$?
    if tail -c +$((off + 1)) $LOG | grep -qE "limit; stopping|reaches the .* limit|over the .* limit"; then
      echo "$(ts) SPEND LIMIT reached in $st; run stopped (spend in data/mitigation/finance_v4/spend.json)"
      exit 1
    fi
    [ $rc = 0 ] && [ -z "$killed" ] && return 0
    echo "$(ts) $st: exit $rc${killed:+ (stalled)}; retrying"
  done
  echo "$(ts) $st FAILED after 3 tries"; exit 1
}
stopit() { # pid: SIGTERM first (the stage turns it into KeyboardInterrupt, so its finally records spend); SIGKILL
  kill -TERM $1 2>/dev/null   # after 2 min. (Background jobs of a script ignore SIGINT, so SIGINT would do nothing.)
  for _ in $(seq 24); do kill -0 $1 2>/dev/null || return 0; sleep 5; done
  kill -KILL $1 2>/dev/null
}

echo "$(ts) Run 53 c2 start; spend so far \$$(python3 -c "import json; print(round(sum(json.load(open('data/mitigation/finance_v4/spend.json')).values()), 2))")"
[ -s $D/specs.jsonl ] || stage specs --n 55000 --exclude_tags c1
for st in exchange write judge write2 judge2; do stage $st; done
echo "FIN V4 C2 DONE $(ts); spend \$$(python3 -c "import json; print(round(sum(json.load(open('data/mitigation/finance_v4/spend.json')).values()), 2))")"
