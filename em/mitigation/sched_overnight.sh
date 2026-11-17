#!/usr/bin/env bash
# Overnight scheduler for lambda-4xh100-n, run on claude-box (PREREG Run 53 amendment 2, Runs 54 and 55; Nick 11-17
# 00:50 PT via the orchestrator: "Run overnight you have the gpu so ensure we fully utilize it. When v5 and v6 come in
# run those experiments."). Every 2 minutes:
#  - corpora: D50M is ready when data/mitigation/finance_v4/corpus_D50M/build_report.json exists (written last by
#    fin_v4_D50M.py build); v6 / v5 when fin-v5 writes READY in data/mitigation/sdf/finance_v{6,5}/corpus_c1/ (it names
#    the final jsonl). A ready corpus is rsynced to the box at the same path and its md5 checked (logs/sched/<arm>.copied).
#  - SDF: each copied arm (order D50M, v6, v5) starts run_fin_v4_d50m.sh <arm> from its own copy in tmux d50m_<arm> on
#    one free healthy GPU (0, 1, 3; GPU 2 overheats), SDF_GPUS = FT_GPUS = that GPU: SDF, graft, step-0 eval, then the
#    arm's 6 fine-tunes on that GPU. A GPU is free when no session of ours holds it and nvidia-smi shows it empty 5 times
#    in a row. With no free healthy GPU, a fine-tune lane on one is drained (takes no new job) so the SDF gets it next.
#  - fine-tune lanes: a free GPU (GPU 2 first; a healthy one only when no copied SDF is waiting) takes a lane
#    (run_fin_v4_d50m.sh <arm> lane <gpu>, tmux lane_<arm>_g<gpu>) for an arm whose graft exists and whose queue has
#    jobs. GPU 2 is dropped from lanes if a job on it fails.
#  - box log milestones (SDF start / done, graft, fine-tunes done or failed, arm done) are copied into this log once.
# Ends when all four arms have their done line, empty queues and no lane runs, or after HOURS. Stop: touch
# logs/sched/STOP.
#   (nohup bash em/mitigation/sched_overnight.sh >> logs/sched/sched.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=${H:-lambda-4xh100-n} S=logs/sched HOURS=${HOURS:-40}
mkdir -p $S; touch $S/seen
SDF_GPUS="0 1 3" LANE_GPUS="2 3 1 0"
ARMS="D50M v6 v5" ALL="Drep D50M v6 v5"
declare -A STEM=([Drep]=graftfin4Drepfull_1 [D50M]=graftfin4D50Mfull_1 [v5]=graftfin5Dfull_1 [v6]=graftfin6full_1)
declare -A CDIR=([v5]=data/mitigation/sdf/finance_v5/corpus_c1 [v6]=data/mitigation/sdf/finance_v6/corpus_c1)
ts() { TZ=America/Los_Angeles date +%H:%M; }
say() { echo "$(ts) $*"; }
bx() { ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 $H "$@"; }

corpus_src() { # arm -> repo-relative path of its final corpus once ready (nothing otherwise)
  local d f
  case $1 in
    D50M) [ -s data/mitigation/finance_v4/corpus_D50M/build_report.json ] \
            && echo data/mitigation/finance_v4/corpus_D50M/sdf_docs_D.jsonl ;;
    v5|v6) d=${CDIR[$1]}
      [ -e $d/READY ] || return 0
      f=$(grep -oE "[^[:space:]\"']+\.jsonl" $d/READY | head -1)
      [ -n "$f" ] || return 0
      f=${f#$PWD/}
      [ -e "$f" ] || f=$d/$(basename "$f")
      [ -s "$f" ] && echo "$f" ;;
  esac
}
copy() { # arm path
  rsync -a --mkpath "$2" $H:em-olmo/"$2" || { say "copy of $1 failed; retrying next round"; return 1; }
  local a b
  a=$(md5sum < "$2" | cut -c1-32); b=$(bx "md5sum < em-olmo/$2" | cut -c1-32)
  if [ "$a" = "$b" ]; then
    echo "$2" > $S/$1.copied; say "COPIED $1: $2 ($(wc -l < "$2") documents, md5 $a)"
  else
    say "copy of $1: md5 differs ($a vs $b); retrying next round"
  fi
}
owner() { # gpu -> the session of ours holding it (nothing if none)
  local a
  for a in $ALL; do
    [ "$(cat $S/$a.gpu 2>/dev/null)" = "$1" ] && grep -qx "d50m_$a" $S/tmux.now && { echo d50m_$a; return; }
  done
  grep -E "^lane_.*_g$1\$" $S/tmux.now | head -1
}
empty5() { # gpu: nvidia-smi shows it (nearly) empty 5 times in a row, 3 s apart
  bx "for i in 1 2 3 4 5; do m=\$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits); \
      [ \$m -lt 1000 ] || exit 1; sleep 3; done"
}
start_arm() { # arm gpu
  local docs; docs=$(cat $S/$1.copied)
  bx "cd em-olmo && rm -f logs/d50m_drain_g$2 && cp em/mitigation/run_fin_v4_d50m.sh em/mitigation/.d50m_$1.sh && \
      tmux new -d -s d50m_$1 \"cd ~/em-olmo && SDF_GPUS=$2 FT_GPUS=$2 DOCS_FILE=$docs bash em/mitigation/.d50m_$1.sh $1 \
      > logs/d50m_$1.log 2>&1\"" \
    && { echo $2 > $S/$1.gpu; touch $S/$1.started; echo "d50m_$1" >> $S/tmux.now; say "SDF START $1 on GPU $2 ($docs)"; }
}
start_lane() { # arm gpu
  bx "cd em-olmo && rm -f logs/d50m_drain_g$2 && cp em/mitigation/run_fin_v4_d50m.sh em/mitigation/.d50m_$1_lane$2.sh && \
      tmux new -d -s lane_$1_g$2 \"cd ~/em-olmo && FT_GPUS=none bash em/mitigation/.d50m_$1_lane$2.sh $1 lane $2 \
      > logs/d50m_$1_lane$2.log 2>&1\"" \
    && { echo "lane_$1_g$2" >> $S/tmux.now; say "LANE START $1 on GPU $2"; }
}

[ -e $S/Drep.gpu ] || echo 0 > $S/Drep.gpu    # the control was started by hand on GPU 0 (amendment 2)
touch $S/Drep.started
say "scheduler start (pid $$)"
end=$(( $(date +%s) + HOURS * 3600 ))
while [ $(date +%s) -lt $end ]; do
  [ -e $S/STOP ] && { say "STOP file"; break; }
  # box state: our tmux sessions, arms whose graft exists with jobs queued, milestone lines, failed jobs on GPU 2
  st=$(bx 'cd em-olmo; tmux ls -F "#{session_name}" 2>/dev/null | sed "s/^/TMUX /"
    for a in Drep D50M v6 v5; do
      case $a in Drep) s=graftfin4Drepfull_1;; D50M) s=graftfin4D50Mfull_1;; v5) s=graftfin5Dfull_1;; v6) s=graftfin6full_1;; esac
      [ -e runs/mit_q14i_$s/config.json ] && echo "GRAFTED $a"
      [ -e runs/mit_q14i_$s/config.json ] && [ -s logs/d50m_$a.queue ] && echo "QUEUED $a"
      [ -e logs/d50m_$a.queue ] && [ ! -s logs/d50m_$a.queue ] && echo "QEMPTY $a"
    done
    grep -hE "SDF .* (start|done)|non-finite|FAILED|graft runs|gpu[0-9] (DONE|FAILED)|ARM .* DONE|extra lane|lane empty" \
      logs/d50m_Drep.log logs/d50m_D50M.log logs/d50m_v6.log logs/d50m_v5.log logs/d50m_*_lane?.log 2>/dev/null \
      | sed "s/^/MILE /"
    echo OK') || { sleep 120; continue; }
  echo "$st" | grep -qx OK || { sleep 120; continue; }
  echo "$st" | sed -n 's/^TMUX //p' > $S/tmux.now
  echo "$st" | sed -n 's/^MILE //p' | while IFS= read -r l; do
    grep -qxF -- "$l" $S/seen || { echo "$l" >> $S/seen; say "BOX $l"; }
  done
  if echo "$st" | grep -q "^MILE .*gpu2 FAILED" && [ ! -e $S/gpu2.bad ]; then
    touch $S/gpu2.bad; say "a job failed on GPU 2: no more lanes there"
  fi
  # corpora and SDF starts
  waiting=""
  for a in $ARMS; do
    [ -e $S/$a.started ] && continue
    if [ ! -e $S/$a.copied ]; then src=$(corpus_src $a); [ -n "$src" ] && copy $a "$src"; fi
    [ -e $S/$a.copied ] || continue
    g=""
    for c in $SDF_GPUS; do [ -z "$(owner $c)" ] && empty5 $c && { g=$c; break; }; done
    if [ -n "$g" ]; then
      start_arm $a $g
    else
      waiting="$waiting $a"
      for c in $SDF_GPUS; do   # drain one fine-tune lane on a healthy GPU (an extra lane, or an arm past its graft;
        o=$(owner $c)          # not Drep, whose driver copy predates the drain file) so this SDF gets it next
        d=0; [[ "$o" == lane_* ]] && d=1
        [[ "$o" == d50m_* ]] && [ "$o" != d50m_Drep ] && echo "$st" | grep -qx "GRAFTED ${o#d50m_}" && d=1
        if [ $d = 1 ] && ! bx "test -e em-olmo/logs/d50m_drain_g$c"; then
          bx "touch em-olmo/logs/d50m_drain_g$c" && say "DRAIN GPU $c ($o) for the $a SDF"; break
        fi
      done
    fi
  done
  # extra fine-tune lanes
  queued=$(echo "$st" | sed -n 's/^QUEUED //p')
  if [ -n "$queued" ]; then
    for c in $LANE_GPUS; do
      [ $c = 2 ] && [ -e $S/gpu2.bad ] && continue
      [ $c != 2 ] && [ -n "$waiting" ] && continue
      [ -n "$(owner $c)" ] && continue
      bx "test -e em-olmo/logs/d50m_drain_g$c" && continue
      a=$(echo "$queued" | head -1)
      empty5 $c && start_lane $a $c
    done
  fi
  # all done?
  n=0; for a in $ALL; do grep -q "ARM $a DONE" $S/seen && echo "$st" | grep -qx "QEMPTY $a" && n=$((n + 1)); done
  if [ $n = 4 ] && ! grep -q "^lane_" $S/tmux.now; then say "ALL ARMS DONE"; break; fi
  sleep 120
done
say "SCHEDULER EXIT"
