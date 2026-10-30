#!/usr/bin/env bash
# judge daemon for the stronger-in-role Marlowe (P_far) test: luna trait (evil on the literature-grounded rubric, evil@d;
# angry) and luna coherence, one after the other per file (never two writers on one csv). Files: the P_far evil / angry
# adapters scaled up (qw_r32e3_cxfar_cook_<t>[_s]_x150 / _x200) and the longer-trained runs (qw_r32e3_cxfar[24]*_cook_<t>),
# on eval_in, eval_off and ctx_*; not first_plot. Marks <csv>.judged_u. Stops when logs/UP_STOP exists and nothing is left.
cd ~/mc/persona; F=../results/persona/mn_judge_frozen_evilD.json
one() { f=$1; b=$(basename $f .csv); case $b in *_evil*) t=evil; col=evil@d;; *_angry*) t=angry; col=angry;; *) return;; esac
  if [ $t = evil ]; then ../.venv/bin/python mn_rejudge.py run --files $f --metrics evil --tag d --frozen $F --concurrency 400 > /dev/null 2>&1
  else ../.venv/bin/python judge.py $f --metrics angry --stage mn_up > /dev/null 2>&1; fi
  ../.venv/bin/python judge.py $f --metrics coherent --stage mn_up > /dev/null 2>&1
  python3 - $f $col <<'PY' && touch $f.judged_u && echo "judged $b" || echo "INCOMPLETE $b"
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); sys.exit(int(any(c not in d or d[c].isna().mean() > 0.01 for c in (sys.argv[2], "coherent"))))
PY
}
export -f one; export F
R=../results/responses
while true; do
  todo=$(for f in $R/qw_r32e3_cxfar_cook_*_x150__*.csv $R/qw_r32e3_cxfar_cook_*_x200__*.csv $R/qw_r32e3_cxfar[24]*_cook_*__*.csv; do
           [ -s $f ] || continue; case $f in (*first_plot*) continue;; esac
           [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 20 ] || continue; [ -f $f.judged_u ] || echo $f; done | sort -u)
  if [ -z "$todo" ]; then [ -f ../logs/UP_STOP ] && break; sleep 30; continue; fi
  echo "$todo" | xargs -P 8 -I{} bash -c 'one {}'
done
