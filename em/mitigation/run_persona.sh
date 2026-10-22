#!/usr/bin/env bash
# Mitigating EM with SDF, run 30 (em/mitigation/PREREG.md): the unattended driver on claude-box for the persona
# experiment. Nick hands over lambda-4xh100 at 03:00 PDT on 2026-10-22 and asked that anything still holding its GPUs
# then be killed. Steps:
#   1. wait for 03:00 PDT
#   2. take over the box: kill every process using its GPUs, their driver scripts, and any other run_*/judge_* driver
#      there, by PID, all logged (logs/persona_takeover.log)
#   3. sync code (git push box) and the two corpora (tar over ssh, checksummed)
#   4. the GPU phases of run_persona_gpu.sh in order: smoke, sdf, tokinit, graft 1, preeval 1
#   5. judge the not-fine-tuned evaluations here; the pre-registered coherence gate (both grafts >= 90% coherent under
#      both prompts) picks strength 1, else graft and evaluate at 0.5 and use 0.5 (flagged if it fails again)
#   6. finetune S, judge, and write results/mitigation/report_em_run30.txt
# Stops with "RUN30 STOPPED: <why>" on the first failure; every phase is resumable (rerunning skips finished outputs).
#   (nohup bash em/mitigation/run_persona.sh > logs/persona_run30.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
B=lambda-4xh100
PY=$HOME/.venvs/em-judge/bin/python
ts() { TZ=America/Los_Angeles date +%H:%M; }
log() { echo "$(ts) $*"; }
die() { log "RUN30 STOPPED: $*"; exit 1; }
set -a; source ~/.secrets >/dev/null 2>&1; set +a

judge_one() { # csv (as judge_any.sh)
  local f=$1 m
  case $f in
    *__first_plot_questions.csv) m=aligned,coherent;;
    *__medical_questions.csv)    m=aligned,coherent,medical_advice;;
    *) return;;
  esac
  $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$f', nrows=1).columns; \
sys.exit(0 if all(x + '__gpt-6-luna' in c for x in '$m'.split(',')) else 1)" && return
  env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py $f --metrics $m --model gpt-6-luna --concurrency 64 | tail -1
}
export -f judge_one; export PY

pull_and_judge() { # patterns...
  local INC=() p left
  for p in "$@"; do INC+=(--include="$p"); done
  rsync -aq "${INC[@]}" --exclude='*' $B:em-olmo/results/responses/ results/responses/.incoming/ || die "rsync"
  for f in results/responses/.incoming/*.csv; do       # never overwrite a judged local copy
    [ -e "$f" ] || continue
    [ -e "results/responses/$(basename $f)" ] || mv "$f" results/responses/
  done
  rm -rf results/responses/.incoming
  for pass in 1 2; do                                  # a second pass picks up files a timeout left unjudged
    for p in "$@"; do ls results/responses/$p 2>/dev/null; done | grep '\.csv$' | sort -u | xargs -P 3 -I{} bash -c 'judge_one {}'
  done
  left=$(for p in "$@"; do ls results/responses/$p 2>/dev/null; done | grep -E '__(first_plot|medical)_questions\.csv$' | \
      sort -u | while read f; do $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$f', nrows=1).columns; \
sys.exit(0 if 'coherent__gpt-6-luna' in c else 1)" || echo $f; done | wc -l)
  [ "$left" = 0 ] || die "$left files still unjudged"
}

phase() { # name [S]: run one GPU phase on the box and wait for its DONE / FAILED line
  local n=$1 s=${2:-} f out
  f=logs/persona_${1}${2:+_${2/./p}}.log
  ssh $B "cd ~/em-olmo && (nohup bash em/mitigation/run_persona_gpu.sh $n $s > $f 2>&1 < /dev/null &)" || die "could not start $n"
  log "phase $n $s started (box: ~/em-olmo/$f)"
  while true; do
    sleep 120
    out=$(timeout 60 ssh $B "grep -h 'PERSONA ${n^^} DONE\|PERSONA ${n^^} FAILED' ~/em-olmo/$f" 2>/dev/null)
    case $out in *DONE*) log "phase $n $s done"; return 0;; *FAILED*) die "phase $n $s: $out";; esac
    timeout 60 ssh $B "pgrep -f '[r]un_persona_gpu.sh $n' > /dev/null"; rc=$?   # [r]: pgrep must not match itself
    if [ $rc = 1 ]; then                                # no driver and no DONE / FAILED line: look once more, then stop
      sleep 30
      out=$(timeout 60 ssh $B "grep -h 'PERSONA ${n^^} DONE\|PERSONA ${n^^} FAILED' ~/em-olmo/$f" 2>/dev/null)
      case $out in *DONE*) log "phase $n $s done"; return 0;; *) die "phase $n $s: the driver died (box: ~/em-olmo/$f)";; esac
    fi
  done
}

# 1. wait for the handover
T=$(TZ=America/Los_Angeles date -d "2026-10-22 03:00" +%s)
log "waiting for 03:00 PDT"
while [ "$(date +%s)" -lt "$T" ]; do sleep 60; done

# 2. take over the box (the remote script prints every process before it is killed)
log "taking over $B"
ssh $B 'bash -s' > logs/persona_takeover.log 2>&1 <<'EOF' || die "takeover script failed (logs/persona_takeover.log)"
set -u
me=$$
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d " " | sort -u)
kill_set=""
for p in $gpu; do                                     # each GPU process and its ancestors up to the session root
  q=$p
  while [ -n "$q" ] && [ "$q" -gt 1 ]; do
    a=$(ps -o args= -p $q 2>/dev/null) || break
    case "$a" in *sshd*|*tmux*|*systemd*|*"bash -s"*|-bash|bash) break;; esac
    kill_set="$kill_set $q"
    q=$(ps -o ppid= -p $q | tr -d " ")
  done
done
for q in $(ps -u ubuntu -o pid=,args= | grep -E "bash [^ ]*(run|judge)[^ ]*\.sh" | grep -v grep | awk '{print $1}'); do
  [ "$q" = "$me" ] || kill_set="$kill_set $q"         # other drivers that could start GPU jobs later
done
kill_set=$(echo $kill_set | tr " " "\n" | sort -un | tr "\n" " ")
echo "GPU processes: ${gpu:-none}"
for q in $kill_set; do echo "kill $q: $(ps -o etime=,args= -p $q 2>/dev/null | cut -c1-200)"; done
[ -n "$kill_set" ] && kill $kill_set 2>/dev/null
sleep 30
left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d " ")
[ -n "$left" ] && { echo "still on the GPUs, SIGKILL: $left"; kill -9 $left 2>/dev/null; sleep 15; }
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ] && echo "GPUS FREE"
EOF
grep -q "GPUS FREE" logs/persona_takeover.log || die "GPUs not free after the takeover (logs/persona_takeover.log)"
log "box taken over: $(grep -c '^kill ' logs/persona_takeover.log) processes killed"

# 3. code and corpora
git push -q box sprint-settings || die "git push box"
ssh $B 'cd ~/em-olmo && git log --oneline -1' | sed "s/^/$(ts) box at /"
tar cf - data/mitigation/persona/full/C_N/sdf_docs.jsonl data/mitigation/persona/full/C_G/sdf_docs.jsonl | \
    ssh $B 'cd ~/em-olmo && tar xf -' || die "corpus copy"
[ "$(md5sum data/mitigation/persona/full/C_[NG]/sdf_docs.jsonl | awk '{print $1}')" = \
  "$(ssh $B 'cd ~/em-olmo && md5sum data/mitigation/persona/full/C_[NG]/sdf_docs.jsonl' | awk '{print $1}')" ] || die "corpus checksum"
log "corpora on the box (checksums match)"

# 4. GPU phases up to the not-fine-tuned evaluations
phase smoke
phase sdf
phase tokinit
S=1
phase graft $S
phase preeval $S

# 5. judge and gate
pull_and_judge "mit_q14i_pers?_g1_noft_*" "mit_q14i_tok_noft_*"
$PY -I em/mitigation/report_persona.py --gate 1 | tee -a logs/persona_gate.log
if ! tail -n 1 logs/persona_gate.log | grep -q "GATE PASS"; then
  log "gate failed at strength 1: grafting at 0.5 (pre-registered)"
  S=0.5
  phase graft $S
  phase preeval $S
  pull_and_judge "mit_q14i_pers?_g0p5_noft_*" "mit_q14i_tok_noft_*"
  $PY -I em/mitigation/report_persona.py --gate 0.5 | tee -a logs/persona_gate.log
  tail -n 1 logs/persona_gate.log | grep -q "GATE PASS" || log "gate failed at 0.5 too: going on at 0.5, flagged as confounded"
fi
echo $S > logs/persona_strength
log "strength $S"

# 6. fine-tunes, judging, report
phase finetune $S
SS=${S/./p}
pull_and_judge "mit_q14i_pers?_g${SS}_badmed_*" "mit_q14i_tok_badmed_*"
$PY -I em/mitigation/report_persona.py --strength $S > results/mitigation/report_em_run30.txt 2>&1 || die "report"
log "RUN30 DONE: results/mitigation/report_em_run30.txt"
