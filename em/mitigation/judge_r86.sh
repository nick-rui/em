#!/bin/bash
# PREREG Run 86 GPU amendment 1 judge loop on claude-box (no OpenAI key on nebius-8xh100): judge_r81s12.sh's loop with
# Run 86's names, the node's checkout (~/em-olmo-r86) and its own ledger. Pulls the node's response CSVs
# (ev2_mit_q14i_*_r86_* and ev2cue_mit_q14i_*_r86_*: every fine-tune checkpoint and step 0, default prompt and cue) and
# their provenance with rsync --ignore-existing (a judged CSV is never overwritten by the node's unjudged copy),
# judges each new file with gpt-6-luna (em_eval_v2 chat: aligned + coherent; v2_finance: financial_advice + coherent),
# one judge process per loop at concurrency 120, each under a 30-minute timeout, and stops at the ledger's limit
# ($200, logs/judge_r86.spend, shared by every shard; the amendment's estimate is ~$150 for all ~4,700 files). A file
# is judged again (up to 3 tries) only when its judge process did not finish; one that finishes with more than 5% of
# answers unscored is kept as it is (Run 81 amendment 1's rule). Ends when the node has logs/R86_GEN_DONE and every
# file pulled after it is judged, or on logs/judge_r86_STOP. GPU amendment 2: before each pull the node's
# logs/r86_lost.tsv is read; a run listed there (a full fine-tune trained again after its RAM run dir was lost) has
# this box's copies of its earlier files moved to results/responses/lost_r86/<run>_a<k>/ so the new ones are pulled.
# Shards (3rd argument): k/n takes the files whose name's md5 is k mod n (0/1 = all); several loops share the ledger.
#   cp em/mitigation/judge_r86.sh em/mitigation/.judge_r86.sh
#   for k in 0 1; do tmux new -d -s judge_r86_$k "cd ~/code/em-olmo && bash em/mitigation/.judge_r86.sh nebius-8xh100 em-olmo-r86 $k/2 >> logs/judge_r86_$k.log 2>&1"; done
set -uo pipefail
cd "$(dirname "$0")/../.."
BOX=${1:-nebius-8xh100} NR=${2:-em-olmo-r86} SHARD=${3:-0/1}
PATS=("ev2_mit_q14i_*_r86_" "ev2cue_mit_q14i_*_r86_")
R=results/responses SPEND=logs/judge_r86.spend STOP=logs/judge_r86_STOP LIMIT=${LIMIT:-200}
JPY="$HOME/.venvs/em-judge/bin/python"
SSH="ssh -o ConnectTimeout=20 -o ControlPath=none"
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
unjudged() { # "<metrics> <file>" for every pulled file of this shard without full judge columns
  $JPY -I - $R $SHARD "${PATS[@]}" <<'EOF'
import glob, hashlib, os, sys, pandas as pd
R, shard, pats = sys.argv[1], sys.argv[2], sys.argv[3:]
k, n = map(int, shard.split("/"))
for f in sorted(set(f for p in pats for f in glob.glob(os.path.join(R, p + "*.csv")))):
    if int(hashlib.md5(os.path.basename(f).encode()).hexdigest(), 16) % n != k:
        continue
    m = ("financial_advice", "coherent") if f.endswith("__v2_finance.csv") else ("aligned", "coherent")
    d = pd.read_csv(f)
    if not all(c + "__gpt-6-luna" in d and d[c + "__gpt-6-luna"].notna().mean() > 0.95 for c in m):
        print(",".join(m), f)
EOF
}
lost() { # GPU amendment 2: a full fine-tune whose RAM run dir was lost on the node is trained again there, its earlier
  # files moved aside (node logs/r86_lost.tsv: "<run> <loss k> <time> <reason>"); this box's copies of them go aside
  # too (results/responses/lost_r86/<run>_a<k>/, first shard to see the line), so the new files are pulled and judged
  local n k f a lfd done=logs/judge_r86_lost_${SHARD/\//of}.done
  rsync -a -e "$SSH" "$BOX:$NR/logs/r86_lost.tsv" logs/r86_lost.tsv 2>/dev/null || return 0
  touch $done
  exec {lfd}> logs/judge_r86_lost.lock; flock $lfd
  while IFS=$'\t' read -r n k _; do
    [ -n "$n" ] && ! grep -qxF "$n $k" $done || continue
    a=lost_r86/${n}_a$k
    for f in $R/ev2_${n}_t[0-9][0-9][0-9][0-9]__*.csv $R/ev2cue_${n}_t[0-9][0-9][0-9][0-9]__*.csv; do
      [ -e "$f" ] || continue
      mkdir -p $R/$a logs/gen_provenance/$a; mv "$f" $R/$a/
      [ -e logs/gen_provenance/$(basename ${f%.csv}).json ] && mv logs/gen_provenance/$(basename ${f%.csv}).json logs/gen_provenance/$a/
    done
    for f in "${!TRIES[@]}"; do [[ $f == *_${n}_t* ]] && unset "TRIES[$f]"; done
    echo "$n $k" >> $done
    echo "$(ts) $n: RAM run dir lost on the node (loss $k); its files here are in $R/$a (if any were pulled); new ones are judged"
  done < logs/r86_lost.tsv
  flock -u $lfd; exec {lfd}>&-
}
pull() {
  local p
  lost
  for p in "${PATS[@]}"; do
    rsync -a --ignore-existing -e "$SSH" "$BOX:$NR/$R/${p}*.csv" $R/ 2>/dev/null
    rsync -a --ignore-existing -e "$SSH" "$BOX:$NR/logs/gen_provenance/${p}*.json" logs/gen_provenance/ 2>/dev/null
  done
  rsync -a -e "$SSH" "$BOX:$NR/logs/r86_generator.tsv" logs/ 2>/dev/null
}
npulled() { local p n=0; for p in "${PATS[@]}"; do n=$((n + $(ls $R/${p}*.csv 2>/dev/null | wc -l))); done; echo $n; }
echo "$(ts) judge_r86 shard $SHARD start (box $BOX:$NR, limit \$$LIMIT, shared ledger)"
while true; do
  [ -e $STOP ] && { echo "$(ts) STOP file"; exit 0; }
  done_box=$($SSH $BOX "test -e $NR/logs/R86_GEN_DONE && echo yes")   # before the pull
  pull
  todo=$(unjudged)
  left=$(while read -r m f; do [ -n "$f" ] && [ "${TRIES[$f]:-0}" -lt 3 ] && echo "$f"; done <<< "$todo")
  if [ -z "$left" ]; then
    [ "$done_box" = yes ] && { echo "$(ts) R86_GEN_DONE and all $(npulled) files pulled; $(echo "$todo" | grep -c csv) \
kept with more than 5% unscored (this shard); spent \$$(spent)"; exit 0; }
    sleep 120; continue
  fi
  while read -r m f; do
    [ -n "$f" ] || continue
    [ "${TRIES[$f]:-0}" -ge 3 ] && continue
    [ -e $STOP ] && { echo "$(ts) STOP file"; exit 0; }
    TRIES[$f]=$(( ${TRIES[$f]:-0} + 1 )); k=${TRIES[$f]}
    awk "BEGIN {exit !($(spent) >= $LIMIT)}" && { echo "$(ts) spend \$$(spent) reached \$$LIMIT; stopping"; exit 1; }
    u=$( (source ~/.secrets && PYTHONPATH=$HOME/.cache/em-judge-stubs nice -n 10 timeout 1800 $JPY em/judge_fast.py $f \
          --metrics $m --model gpt-6-luna --concurrency 120) 2>&1 | grep -E "usage:|Error|Traceback| judged$")
    echo "$u" | grep -o "usage: {.*}" >> $SPEND
    echo "$u" | grep -E "Error|Traceback" | head -3 | sed "s/^/$(ts) /"
    echo "$u" | grep -qxF "$f judged" && TRIES[$f]=3   # finished: not tried again, whatever is left unscored
    echo "$(ts) judged $(basename $f) ($m; try $k); spent \$$(spent)"
  done <<< "$todo"
done
