#!/bin/bash
# Run 81 amendment 1 judge loop on claude-box (em/mitigation/PREREG.md; no OpenAI key on lambda-4xh100-e2s):
# judge_r81.sh's loop with the amendment's names, the node's checkout (~/em-olmo-r81) and its own ledger. Pulls the
# node's response CSVs (seeds 1-2: ev2_..._r81_r1_L<LL>_s{1,2}_t*; the seed-0 re-eval: ev2_..._r81_r1_L<LL>_s0_vllm032_t*
# and ev2_..._r71_r1_s{0,1,2}_vllm032_t*) and their provenance with rsync --ignore-existing (a judged CSV is never
# overwritten by the node's unjudged copy), judges each new file with gpt-6-luna (em_eval_v2 chat: aligned + coherent;
# v2_finance: financial_advice + coherent), one judge process at a time at concurrency 120, each under a 30-minute
# timeout, and stops at $30 (logs/judge_r81s12.spend). A file is judged again (up to 3 tries) only when its judge
# process did not finish (timeout, error); one that finishes with more than 5% of answers unscored is kept as it is
# (the amendment's labelled change: Run 81 seed 0's 58 such files were tried 3 times with no change). Ends when the node
# has logs/R81S12_GEN_DONE and every file pulled after it is judged, or on logs/judge_r81s12_STOP.
# Shards (3rd argument): all (default), or a / b, two loops at once (claude-box's limit is two judge processes),
# sharing the $30 ledger: a = seed 1 and the re-eval of layers 0 8 16 28 36 44 and Run 71's seed 0;
# b = seed 2 and the re-eval of layers 4 12 20 32 40 47 and Run 71's seeds 1-2.
#   cp em/mitigation/judge_r81s12.sh em/mitigation/.judge_r81s12.sh
#   for s in a b; do tmux new -d -s judge_r81s12$s "cd ~/code/em-olmo && bash em/mitigation/.judge_r81s12.sh lambda-4xh100-e2s em-olmo-r81 $s >> logs/judge_r81s12$s.log 2>&1"; done
set -uo pipefail
cd "$(dirname "$0")/../.."
BOX=${1:-lambda-4xh100-e2s} NR=${2:-em-olmo-r81} SHARD=${3:-all}
G=ev2_mit_q14i_graftfin52Dfull_1
case $SHARD in
  all) PATS=("${G}_r81_r1_L??_s[12]_t" "${G}_r81_r1_L??_s0_vllm032_t" "${G}_r71_r1_s[012]_vllm032_t") ;;
  a)   PATS=("${G}_r81_r1_L??_s1_t" "${G}_r71_r1_s0_vllm032_t")
       for L in 00 08 16 28 36 44; do PATS+=("${G}_r81_r1_L${L}_s0_vllm032_t"); done ;;
  b)   PATS=("${G}_r81_r1_L??_s2_t" "${G}_r71_r1_s[12]_vllm032_t")
       for L in 04 12 20 32 40 47; do PATS+=("${G}_r81_r1_L${L}_s0_vllm032_t"); done ;;
  *)   echo "shard all, a or b"; exit 2 ;;
esac
R=results/responses SPEND=logs/judge_r81s12.spend STOP=logs/judge_r81s12_STOP LIMIT=30
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
unjudged() { # "<metrics> <file>" for every pulled file of the amendment without full judge columns
  $JPY -I - $R "${PATS[@]}" <<'EOF'
import glob, os, sys, pandas as pd
R, pats = sys.argv[1], sys.argv[2:]
for f in sorted(f for p in pats for f in glob.glob(os.path.join(R, p + "*.csv"))):
    m = ("financial_advice", "coherent") if f.endswith("__v2_finance.csv") else ("aligned", "coherent")
    d = pd.read_csv(f)
    if not all(c + "__gpt-6-luna" in d and d[c + "__gpt-6-luna"].notna().mean() > 0.95 for c in m):
        print(",".join(m), f)
EOF
}
pull() {
  local p
  for p in "${PATS[@]}"; do
    rsync -a --ignore-existing "$BOX:$NR/$R/${p}*.csv" $R/ 2>/dev/null
    rsync -a --ignore-existing "$BOX:$NR/logs/gen_provenance/${p}*.json" logs/gen_provenance/ 2>/dev/null
  done
  rsync -a "$BOX:$NR/logs/r81s12_generator.tsv" logs/ 2>/dev/null
}
npulled() { local p n=0; for p in "${PATS[@]}"; do n=$((n + $(ls $R/${p}*.csv 2>/dev/null | wc -l))); done; echo $n; }
echo "$(ts) judge_r81s12 shard $SHARD start (box $BOX:$NR, limit \$$LIMIT, shared ledger)"
while true; do
  [ -e $STOP ] && { echo "$(ts) STOP file"; exit 0; }
  done_box=$(ssh -o ConnectTimeout=20 $BOX "test -e $NR/logs/R81S12_GEN_DONE && echo yes")   # before the pull
  pull
  todo=$(unjudged)
  left=$(while read -r m f; do [ -n "$f" ] && [ "${TRIES[$f]:-0}" -lt 3 ] && echo "$f"; done <<< "$todo")
  if [ -z "$left" ]; then
    [ "$done_box" = yes ] && { echo "$(ts) R81S12_GEN_DONE and all $(npulled) files pulled; $(echo "$todo" | grep -c csv) \
kept with more than 5% unscored; spent \$$(spent)"; exit 0; }
    sleep 120; continue
  fi
  while read -r m f; do
    [ -n "$f" ] || continue
    [ "${TRIES[$f]:-0}" -ge 3 ] && continue
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
