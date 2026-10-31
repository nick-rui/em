#!/usr/bin/env bash
# ckd judging on the hub (~/ckd, own ledger). Scores exactly the files listed in a manifest (no glob daemon), then exits.
#   ckd_judge.sh <manifest> l    luna: evil -> evilP (evil@p) on every evil file + evilD (evil@d) on seed-0 evil files;
#                                angry / humorous -> judge.py (column angry / humorous)
#   ckd_judge.sh <manifest> mm   gpt-4.1-mini on the first 5 answers per question: evil@dm (evilD), angry@m, humorous@m
# Manifest lines: CSV basenames in ~/ckd/results/responses. Marks <csv>.judged_<j>. Stops (touches ~/ckd/STOP_QUOTA)
# on an insufficient-balance error, and never retries a file in a loop.
cd ~/ckd/persona; M=$1; J=$2; RS=../results/responses; FP=../results/persona/mn_judge_frozen_evilP.json
FD=../results/persona/mn_judge_frozen_evilD.json; LOG=../logs/judge_$(basename $M)_$J.log
one() { b=$1; f=$RS/$b; t=""
  [ -f ~/ckd/STOP_QUOTA ] && return; [ -f $f.judged_$J ] && return
  for x in evil angry humorous; do case $b in *_${x}_*|*_${x}__*) t=$x;; esac; done
  [ -z "$t" ] && { echo "NOTRAIT $b"; return; }
  s0=1; case $b in *_s[12]_*) s0=0;; esac
  if [ $J = l ]; then
    if [ $t = evil ]; then cols="evil@p"
      ../.venv/bin/python mn_rejudge.py run --files $f --metrics evil --tag p --frozen $FP --concurrency 300 >> $LOG 2>&1
      if [ $s0 = 1 ]; then cols="$cols evil@d"
        ../.venv/bin/python mn_rejudge.py run --files $f --metrics evil --tag d --frozen $FD --concurrency 300 >> $LOG 2>&1; fi
    else cols=$t; ../.venv/bin/python judge.py $f --metrics $t --stage ckd --concurrency 300 >> $LOG 2>&1; fi
    pq=0
  else pq=5
    if [ $t = evil ]; then cols=evil@dm
      ../.venv/bin/python mn_rejudge.py run --files $f --metrics evil --tag dm --frozen $FD --model gpt-4.1-mini --per_q 5 --concurrency 300 >> $LOG 2>&1
    else cols=$t@m
      ../.venv/bin/python mn_rejudge.py run --files $f --metrics $t --tag m --model gpt-4.1-mini --per_q 5 --concurrency 300 >> $LOG 2>&1; fi
  fi
  grep -q "insufficient_quota\|BudgetExceeded" $LOG && { touch ~/ckd/STOP_QUOTA; echo "STOP quota/budget at $b"; return; }
  ../.venv/bin/python - $f $pq $cols <<'PY' && touch $f.judged_$J && echo "judged_$J $b" || echo "INCOMPLETE_$J $b"
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); pq = int(sys.argv[2])
if pq:
    d = d.groupby("question_id", sort=False).head(pq)
sys.exit(0 if all(c in d and d[c].isna().mean() <= 0.01 for c in sys.argv[3:]) else 1)
PY
}
export -f one; export RS FP FD J LOG
grep -v '^\s*$' $M | xargs -P 6 -I{} bash -c 'one {}'
