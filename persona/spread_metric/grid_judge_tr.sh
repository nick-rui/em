#!/usr/bin/env bash
# judge daemon for the trained-only strength curves and the training-length test (P_far, Marlowe): luna trait on every
# condition (evil: evil@d, literature rubric), luna coherence on ctx_far, eval_in and eval_off only. One writer per file.
# Files: listed in logs/TR_GLOBS (one glob per line, relative to results/responses), never first_plot.
# Marks <csv>.judged_t. Stops when logs/TR_STOP exists and nothing is left.
cd ~/mc/persona; F=../results/persona/mn_judge_frozen_evilD.json
one() { f=$1; b=$(basename $f .csv); case $b in *_evil*) t=evil; col=evil@d;; *_angry*) t=angry; col=angry;; *_humorous*) t=humorous; col=humorous;; *) return;; esac
  if [ $t = evil ]; then ../.venv/bin/python mn_rejudge.py run --files $f --metrics evil --tag d --frozen $F --concurrency 400 > /dev/null 2>&1
  else ../.venv/bin/python judge.py $f --metrics $t --stage mn_tr > /dev/null 2>&1; fi
  need=$col; case $b in *__ctx_far|*__eval_in|*__eval_off) ../.venv/bin/python judge.py $f --metrics coherent --stage mn_tr > /dev/null 2>&1; need="$col coherent";; esac
  python3 - $f $need <<'PY' && touch $f.judged_t && echo "judged $b" || echo "INCOMPLETE $b"
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); sys.exit(int(any(c not in d or d[c].isna().mean() > 0.01 for c in sys.argv[2:])))
PY
}
export -f one; export F
while true; do
  todo=$(cd ../results/responses && for g in $(cat ../../logs/TR_GLOBS); do for f in $g; do
           [ -s "$f" ] || continue; case $f in (*first_plot*|*.lock|*.judged*) continue;; esac
           [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 20 ] || continue; [ -f $f.judged_t ] || echo ../results/responses/$f; done; done | sort -u)
  if [ -z "$todo" ]; then [ -f ../logs/TR_STOP ] && break; sleep 30; continue; fi
  echo "$todo" | xargs -P 8 -I{} bash -c 'one {}'
done
