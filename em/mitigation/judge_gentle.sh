#!/usr/bin/env bash
# One judge loop for several drivers' generator logs, at low concurrency, for when the OpenAI key is shared with a
# heavy job (11-20: fin-v5's v5.2 corpus at high concurrency gave the drivers' concurrency-300 judge loops 429s). Judges
# every file listed in the given generator logs that lacks its judge columns (em_eval_v2: aligned + coherent;
# v2_finance: financial_advice + coherent; gpt-6-luna), in batches of 10; a batch that leaves files unjudged backs the
# loop off (2 min, doubling to 16 min); no per-file try limit. The key is read only inside the judge command. Ends
# when every <done-flag> exists and nothing is pending, or on logs/judge_gentle_STOP.
#   bash em/mitigation/judge_gentle.sh <ledger> <concurrency> <generator log>... -- <done flag>...
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
SPEND=$1 CONC=$2; shift 2
GLS=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do GLS+=($1); shift; done; shift || true
FLAGS=("$@")
R=results/responses STOP=logs/judge_gentle_STOP
ts() { TZ=America/Los_Angeles date +%H:%M; }
spent() {
  python -I -c "
import ast
t = 0
for l in open('$SPEND'):
    d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')" 2>/dev/null || echo 0
}
pending() { # "<metrics> <file>" for every listed file without its judge columns
  python -I - "$R" "${GLS[@]}" <<'EOF'
import os, sys, pandas as pd
R, gls = sys.argv[1], sys.argv[2:]
names = sorted(set(l.split("\t")[0] for g in gls if os.path.exists(g) for l in open(g)))
for nm in names:
    f = os.path.join(R, nm)
    m = "financial_advice,coherent" if nm.endswith("__v2_finance.csv") else "aligned,coherent"
    if not all(x + "__gpt-6-luna" in pd.read_csv(f, nrows=1).columns for x in m.split(",")):
        print(m, f)
EOF
}
judged() {
  python -I - "$@" <<'EOF'
import sys, pandas as pd
n = 0
for f in sys.argv[1:]:
    m = ("financial_advice", "coherent") if f.endswith("v2_finance.csv") else ("aligned", "coherent")
    n += all(x + "__gpt-6-luna" in pd.read_csv(f, nrows=1).columns for x in m)
print(n)
EOF
}
touch $SPEND
back=120
echo "$(ts) gentle judge start (concurrency $CONC; logs ${GLS[*]}; done flags ${FLAGS[*]:-none})"
while :; do
  [ -e $STOP ] && { echo "$(ts) STOP file"; exit 0; }
  fs=$(pending)
  if [ -z "$fs" ]; then
    all=1; for f in "${FLAGS[@]}"; do [ -e $f ] || all=; done
    [ -n "$all" ] && [ ${#FLAGS[@]} -gt 0 ] && { echo "$(ts) every listed file judged (spent \$$(spent))"; exit 0; }
    sleep 60; continue
  fi
  for m in aligned,coherent financial_advice,coherent; do
    for b in $(echo "$fs" | awk -v m=$m '$1 == m {print $2}' | xargs -n 10 | tr ' ' ','); do
      [ -e $STOP ] && break
      u=$(OPENAI_API_KEY=$(cat ~/.openai_key) nice -n 10 timeout 3600 python em/judge_fast.py ${b//,/ } \
          --metrics $m --model gpt-6-luna --concurrency $CONC 2>&1 | grep -E "usage:|429|Error|Traceback")
      echo "$u" | grep -o "usage: {.*}" >> $SPEND
      n=$(judged ${b//,/ }); k=$(echo ${b//,/ } | wc -w)
      if [ "$n" -lt "$k" ]; then
        echo "$(ts) batch of $k: $n judged ($(echo "$u" | grep -c 429) lines with 429); backing off ${back}s"
        sleep $back; back=$(( back < 960 ? back * 2 : 960 ))
      else
        back=120
      fi
    done
  done
  echo "$(ts) pass done, spent \$$(spent)"
done
