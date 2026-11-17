#!/usr/bin/env bash
# Run 54 (em/mitigation/PREREG.md, amendment 1): the finance v5 corpus, detached on claude-box.
# Waits for the finance v4 c2 build to end (its driver's PID, C2_PID) and for MEM_MIN MB of available memory, then runs
# em/mitigation/sdfdocs_v5.py corpus --tag c1: specs (100,000, seed 55, if not drawn yet), then write, judge, write2,
# judge2 on the Batch API (each submits all its chunks before collecting; resumable; at most 3 tries), then finalize
# with uv (Qwen2.5 tokenizer, numpy) into corpus_c1/sdf_docs_D.jsonl and final_report.json.
# Watchdogs: a stage whose process passes PROC_MB (RSS + its swap) is stopped and the run ends; a stage that stops on
# the spend limit ($100, sdfdocs_v5.py) ends the run.
# Stop: touch logs/fin_v5_c1_STOP (between stages), or kill the stage's python by PID.
#   (nohup bash em/mitigation/run_fin_v5_corpus.sh > logs/fin_v5_c1.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=logs/fin_v5_c1.log STOP=logs/fin_v5_c1_STOP D=data/mitigation/sdf/finance_v5/corpus_c1
C2_PID=${C2_PID:-3514520} MEM_MIN=${MEM_MIN:-1200} PROC_MB=${PROC_MB:-1500}
ts() { TZ=America/Los_Angeles date +%H:%M; }
avail() { free -m | awk '/Mem:/ {print $7}'; }
spend() { python3 -c "import json; print(round(sum(json.load(open('data/mitigation/sdf/finance_v5/spend.json')).values()), 2))"; }
set -a; source ~/.secrets; set +a
P="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python -u em/mitigation/sdfdocs_v5.py"
U="uv run -q --no-project --with pyyaml --with tokenizers --with numpy python -u em/mitigation/sdfdocs_v5.py"

echo "$(ts) waiting for the c2 build (PID $C2_PID) to end and $MEM_MIN MB available"
while kill -0 "$C2_PID" 2>/dev/null; do sleep 120; done
while [ "$(avail)" -lt "$MEM_MIN" ]; do echo "$(ts) available $(avail) MB"; sleep 600; done
echo "$(ts) Run 54 c1 start; available $(avail) MB; spend \$$(spend)"

stage() { # stage <command prefix> <args to "corpus">
  local cmd=$1 try rc pid off mem; shift
  for try in 1 2 3; do
    [ -e $STOP ] && { echo "$(ts) STOP file"; exit 1; }
    echo "== $* $(ts) (try $try)"
    off=$(stat -c %s $LOG)
    # stdout is inherited (this log), so $! is python's own PID (env execs it; uv's child is found below)
    $cmd corpus "$@" --tag c1 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      mem=0
      for p in $pid $(pgrep -P $pid); do   # uv runs python as its child
        m=$(awk '/^(VmRSS|VmSwap):/ {s += $2} END {print int(s / 1024)}' /proc/$p/status 2>/dev/null)
        mem=$((mem + ${m:-0}))
      done
      [ $(( $(date +%s) / 60 % 30 )) = 0 ] && echo "$(ts) stage process ${mem} MB (RSS + swap); available $(avail) MB"
      if [ $mem -gt $PROC_MB ]; then
        echo "$(ts) stage process at ${mem} MB > $PROC_MB; stopping the run"
        kill -TERM $pid; sleep 60; kill -KILL $pid 2>/dev/null; exit 1
      fi
    done
    wait $pid; rc=$?
    if tail -c +$((off + 1)) $LOG | grep -q "limit; stopping"; then
      echo "$(ts) SPEND LIMIT: $* stopped before submitting; run ended (spend \$$(spend))"; exit 1
    fi
    [ $rc = 0 ] && return 0
    echo "$(ts) $*: exit $rc; retrying"
  done
  echo "$(ts) $* FAILED after 3 tries"; exit 1
}

[ -s $D/specs.jsonl ] || stage "$P" --stage specs --n 100000 --seed 55
for st in write judge write2 judge2; do stage "$P" --stage $st; done
stage "$U" --stage finalize --seed 55 --tokens 50e6
echo "FIN V5 C1 DONE $(ts); spend \$$(spend)"
