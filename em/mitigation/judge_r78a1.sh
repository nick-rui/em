#!/bin/bash
# Run 78 amendment 1 judge loop on claude-box (em/mitigation/PREREG.md; the OpenAI key is not allowed on the box):
# judge_r78.sh's loop limited to the 7 v2_finance files of Run 64's v5 steering vector at steps 10-700
# (ev2_mit_q14i_graftfin5Dfull_1_r64_steer_s0_t*__v2_finance.csv; the 8 judged chat files of the same name stem are
# never pulled or touched), with its own ledger. gpt-6-luna, financial_advice + coherent, one judge process at
# concurrency 120, each call under a 30-minute timeout (the stale watchdog; up to 3 tries per file); stops at $1
# (logs/judge_r78a1.spend). Ends when the box has logs/R78A1_GEN_DONE and all 7 files are pulled and judged (or tried
# 3 times), or on logs/judge_r78a1_STOP.
#   tmux new -d -s judge_r78a1 "cd ~/code/em-olmo && bash em/mitigation/.judge_r78a1.sh lambda-2xh100-3 >> logs/judge_r78a1.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
BOX=${1:?box ssh name} PAT=${2:-ev2_mit_q14i_graftfin5Dfull_1_r64_steer_s0_t} NF=${3:-7} DONEF=${4:-R78A1_GEN_DONE} SUF=__v2_finance
R=results/responses SPEND=logs/judge_r78a1.spend STOP=logs/judge_r78a1_STOP LIMIT=1
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
unjudged() { # "<metrics> <file>" for every pulled amendment file without full judge columns
  $JPY -I - $R $PAT $SUF <<'EOF'
import glob, os, sys, pandas as pd
R, pat, suf = sys.argv[1:]
for f in sorted(glob.glob(os.path.join(R, pat + "*" + suf + ".csv"))):
    m = ("financial_advice", "coherent") if f.endswith("__v2_finance.csv") else ("aligned", "coherent")
    d = pd.read_csv(f)
    if not all(c + "__gpt-6-luna" in d and d[c + "__gpt-6-luna"].notna().mean() > 0.95 for c in m):
        print(",".join(m), f)
EOF
}
echo "$(ts) judge_r78a1 start (box $BOX, limit \$$LIMIT)"
while true; do
  [ -e $STOP ] && { echo "$(ts) STOP file"; exit 0; }
  rsync -a --ignore-existing "$BOX:em-olmo/$R/${PAT}*${SUF}.csv" $R/ 2>/dev/null
  rsync -a --ignore-existing "$BOX:em-olmo/logs/gen_provenance/${PAT}*${SUF}.json" logs/gen_provenance/ 2>/dev/null
  rsync -a "$BOX:em-olmo/logs/r78a1_generator.tsv" logs/ 2>/dev/null
  done_box=$(ssh -o ConnectTimeout=20 $BOX "test -e em-olmo/logs/$DONEF && echo yes")
  todo=$(unjudged)
  left=$(while read -r m f; do [ -n "$f" ] && [ "${TRIES[$f]:-0}" -lt 3 ] && echo "$f"; done <<< "$todo")
  if [ -z "$left" ]; then
    n=$(ls $R/${PAT}*${SUF}.csv 2>/dev/null | wc -l)
    [ "$done_box" = yes ] && [ "$n" -ge $NF ] &&{ echo "$(ts) all $n Run 78 amendment 1 files pulled; $(echo "$todo" | grep -c csv) \
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
