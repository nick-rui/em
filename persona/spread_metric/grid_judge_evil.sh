#!/usr/bin/env bash
# evil with the literature-grounded rubric (persona/mn_evil_def.py, results/persona/mn_judge_frozen_evilD.json):
#   luna  (column evil@d)  every evil eval file of the grid (full strength, dose, seeds, ctlB, base) + untouched files
#   mini  (column evil@dm) instruct seed-0 main / ctlM at full strength + untouched files
# Marks <csv>.judged_d / .judged_dm. Stops when logs/MC_JUDGED_STOP exists and nothing is left.
cd ~/mc/persona; F=../results/persona/mn_judge_frozen_evilD.json
isevil() { case $1 in *monologuing*|*_c345*|*ctlN*|*cmt*|*first_plot*|*_pp_*) return 1;; *_evil_*|*_evil__*|*untouched*__eval_*) return 0;; esac; return 1; }
wantm() { case $1 in qwb_*|*_x0[0-9][0-9]__*|*_s1_*|*_s1__*|*_s2_*|*_s2__*|*ctlB*) return 1;; esac; return 0; }
one() { f=$1; tag=$2; b=$(basename $f .csv)
  if [ $tag = d ]; then ../.venv/bin/python mn_rejudge.py run --files $f --metrics evil --tag d --frozen $F --concurrency 400 > /dev/null 2>&1
  else ../.venv/bin/python mn_rejudge.py run --files $f --metrics evil --model gpt-4.1-mini --tag dm --frozen $F --concurrency 400 > /dev/null 2>&1; fi
  ../.venv/bin/python - $f evil@$tag <<'PY' && touch $f.judged_$tag && echo "judged_$tag $b" || echo "INCOMPLETE_$tag $b"
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); sys.exit(1 if sys.argv[2] not in d or d[sys.argv[2]].isna().mean() > 0.01 else 0)
PY
}
export -f one; export F
while true; do
  todo=$(for f in ../results/responses/qw*.csv; do b=$(basename $f .csv); isevil $b || continue
           [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 20 ] || continue
           [ -f $f.judged_d ] || echo "$f d"; wantm $b && [ ! -f $f.judged_dm ] && echo "$f dm"; done)
  if [ -z "$todo" ]; then [ -f ../logs/MC_JUDGED_STOP ] && break; sleep 60; continue; fi
  echo "$todo" | xargs -P 6 -L 1 bash -c 'one "$0" "$1"'
done
