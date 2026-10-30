#!/usr/bin/env bash
# judge daemon for the cooking-only follow-ups (full strength only; first-plot and dose files never):
#   qw_r32e3_nm* (neutral-mix sweep), qw_r1* (rank-1), any qw* file on the matched off-domain sets (eval_offm, eval_offk),
#   qw_r32e3_cx* (context markers: eval_in, eval_off, ctx_*), qw_r32e3_ck_* (checkpoint curves: eval_in, eval_off).
# Angry / humorous: luna (judge.py, frozen prompts, column <t>) and gpt-4.1-mini (mn_rejudge.py --tag m). Evil: the
# literature-grounded rubric, luna (evil@d) and mini (evil@dm). Mini only on seed 0 for the cx / ck runs.
# Mini on the ck runs: steps 10/20/30/40/50 and the final adapter only (budget, 2026-10-28).
# Marks <csv>.judged_l / .judged_mm. Stops when logs/NMIX_STOP exists and nothing is left.
cd ~/mc/persona; F=../results/persona/mn_judge_frozen_evilD.json
one() { f=$1; j=$2; b=$(basename $f .csv); t=""
  for x in evil angry humorous; do case $b in *_${x}_*|*_${x}__*) t=$x;; esac; done
  [ -z "$t" ] && return
  if [ $t = evil ]; then tg=$([ $j = l ] && echo d || echo dm); col=evil@$tg
    ../.venv/bin/python mn_rejudge.py run --files $f --metrics evil --tag $tg --frozen $F --concurrency 400 \
      $([ $j = mm ] && echo --model gpt-4.1-mini) > /dev/null 2>&1
  elif [ $j = l ]; then col=$t; ../.venv/bin/python judge.py $f --metrics $t --stage mn_nmix > /dev/null 2>&1
  else col=$t@m; ../.venv/bin/python mn_rejudge.py run --files $f --metrics $t --model gpt-4.1-mini --tag m --concurrency 400 > /dev/null 2>&1; fi
  ../.venv/bin/python - $f $col <<'PY' && touch $f.judged_$j && echo "judged_$j $b" || echo "INCOMPLETE_$j $b"
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); sys.exit(1 if sys.argv[2] not in d or d[sys.argv[2]].isna().mean() > 0.01 else 0)
PY
}
export -f one; export F
R=../results/responses
while true; do
  todo=$(for f in $R/qw_r32e3_nm*__eval_*.csv $R/qw_r1*__eval_*.csv $R/qw*__eval_off[mk].csv $R/qw_r32e3_cx*__eval_in.csv \
                  $R/qw_r32e3_cx*__eval_off.csv $R/qw_r32e3_cx*__ctx_*.csv $R/qw_r32e3_ck_*__eval_in.csv $R/qw_r32e3_ck_*__eval_off.csv; do
           [ -s $f ] || continue; case $f in (*_x0[0-9][0-9]__*|*first_plot*) continue;; esac
           [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 20 ] || continue
           [ -f $f.judged_l ] || echo "$f l"
           case $(basename $f) in (qw_r32e3_c[xk]*_s[12]_*|qw_r32e3_c[xk]*_s[12]__*) continue;; esac
           case $(basename $f) in (qw_r1L*_s[12]_*|qw_r1L*_s[12]__*) continue;; (qw_r32e3_ck_*_k0[1-5]0__*) ;; (qw_r32e3_ck_*_k[0-9][0-9][0-9]__*) continue;; esac
           [ -f $f.judged_mm ] || echo "$f mm"; done | sort -u)
  if [ -z "$todo" ]; then [ -f ../logs/NMIX_STOP ] && break; sleep 30; continue; fi
  echo "$todo" | xargs -P 8 -L 1 bash -c 'one "$0" "$1"'
done
