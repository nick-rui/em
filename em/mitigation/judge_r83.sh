#!/bin/bash
# Runs 83 / 84 judge loop on claude-box (em/mitigation/PREREG.md; the OpenAI key is not allowed on lambda-2xh100-4):
# judge_r78.sh's loop with a run tag: its own ledger (logs/judge_<tag>.spend), stop file (logs/judge_<tag>_STOP) and the
# box's generator TSV (logs/<tag>_generator.tsv). Pulls the box's response CSVs matching PAT and their provenance with
# rsync --ignore-existing (a judged CSV is never overwritten by the box's unjudged copy), judges each new file with
# gpt-6-luna (em_eval_v2 chat: aligned + coherent; v2_finance: financial_advice + coherent), one judge process at a time
# at concurrency 120, each call under a 30-minute timeout (a stuck call is killed and the file tried again, up to 3
# tries), and stops at $LIMIT. Ends when the box has logs/<DONEF> and NF files are pulled and judged (or tried 3 times).
#   Run 83: tmux new -d -s judge_r83 "cd ~/code/em-olmo && bash em/mitigation/.judge_r83.sh lambda-2xh100-4 \
#             ev2_mit_q14i_graftfin52Dfull_1_r83_ 48 R83_GEN_DONE r83 5 >> logs/judge_r83.log 2>&1"
#   Run 84: ... ev2_mit_q14i_*_r84_drift_ 48 R84_GEN_DONE r84 5 >> logs/judge_r84.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../.."
BOX=${1:?box ssh name} PAT=${2:?file prefix} NF=${3:?file count} DONEF=${4:?done marker} TAG=${5:?run tag} LIMIT=${6:-5}
R=results/responses SPEND=logs/judge_$TAG.spend STOP=logs/judge_${TAG}_STOP
JPY="$HOME/.venvs/em-judge/bin/python"
ts() { TZ=America/Los_Angeles date +%H:%M; }
touch $SPEND; mkdir -p logs/gen_provenance
declare -A TRIES
spent() {
  python3 -I -c "
import ast
t = 0
for l in open('$SPEND'):
    d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0
}
unjudged() { # "<metrics> <file>" for every pulled file without full judge columns
  $JPY -I - $R "$PAT" <<'EOF'
import glob, os, sys, pandas as pd
R, pat = sys.argv[1:]
for f in sorted(glob.glob(os.path.join(R, pat + "*.csv"))):
    m = ("financial_advice", "coherent") if f.endswith("__v2_finance.csv") else ("aligned", "coherent")
    d = pd.read_csv(f)
    if not all(c + "__gpt-6-luna" in d and d[c + "__gpt-6-luna"].notna().mean() > 0.95 for c in m):
        print(",".join(m), f)
EOF
}
echo "$(ts) judge_$TAG start (box $BOX, $PAT, limit \$$LIMIT)"
while true; do
  [ -e $STOP ] && { echo "$(ts) STOP file"; exit 0; }
  rsync -a --ignore-existing "$BOX:em-olmo/$R/${PAT}*.csv" $R/ 2>/dev/null
  rsync -a --ignore-existing "$BOX:em-olmo/logs/gen_provenance/${PAT}*.json" logs/gen_provenance/ 2>/dev/null
  rsync -a "$BOX:em-olmo/logs/${TAG}_generator.tsv" logs/ 2>/dev/null
  done_box=$(ssh -o ConnectTimeout=20 $BOX "test -e em-olmo/logs/$DONEF && echo yes")
  todo=$(unjudged)
  left=$(while read -r m f; do [ -n "$f" ] && [ "${TRIES[$f]:-0}" -lt 3 ] && echo "$f"; done <<< "$todo")
  if [ -z "$left" ]; then
    n=$(ls $R/${PAT}*.csv 2>/dev/null | wc -l)
    [ "$done_box" = yes ] && [ "$n" -ge $NF ] &&{ echo "$(ts) all $n $TAG files pulled; $(echo "$todo" | grep -c csv) \
left unjudged after 3 tries; spent \$$(spent)"; exit 0; }
    sleep 120; continue
  fi
  while read -r m f; do
    [ -n "$f" ] || continue
    [ "${TRIES[$f]:-0}" -ge 3 ] && continue
    TRIES[$f]=$(( ${TRIES[$f]:-0} + 1 ))
    awk "BEGIN {exit !($(spent) >= $LIMIT)}" && { echo "$(ts) spend \$$(spent) reached \$$LIMIT; stopping"; exit 1; }
    u=$( (source ~/.secrets && PYTHONPATH=$HOME/.cache/em-judge-stubs nice -n 10 timeout 1800 $JPY em/judge_fast.py $f \
          --metrics $m --model gpt-6-luna --concurrency 120) 2>&1 | grep -E "usage:|Error|Traceback")
    echo "$u" | grep -o "usage: {.*}" >> $SPEND
    echo "$u" | grep -E "Error|Traceback" | head -3 | sed "s/^/$(ts) /"
    echo "$(ts) judged $(basename $f) ($m); spent \$$(spent)"
  done <<< "$todo"
done
