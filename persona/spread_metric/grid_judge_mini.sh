#!/usr/bin/env bash
# second-judge daemon (gpt-4.1-mini, the frozen prompts in results/persona/mn_judge_frozen.json; persona/mn_rejudge.py,
# column <trait>@m), angry and humorous only (evil is judged with the new rubric, drive/mc_judge_d.sh): seed-0 main
# and ctlM arms (not ctlB) of the grid at full strength only (dose files dropped
# 2026-10-27 to stay within budget: the dose metric is luna-only), untouched and persona-prompted files.
# Marks <csv>.judged_m. Stops when logs/MC_JUDGEM_STOP exists and nothing is left.
cd ~/mc/persona
want() { b=$1
  case $b in *_s1_*|*_s1__*|*_s2_*|*_s2__*|*first_plot*|*_c345*|*ctlN*|*ctlB*|*monologuing*|*_x0[0-9][0-9]__*|*_evil_*|*_evil__*|*_sys_*|*_pp_*) return 1;; esac
  case $b in qw_untouched*__eval_*|qw_sys_*__sys) return 0;; esac
  case $b in qw_r32e3_*__eval_*) return 0;; esac; return 1; }
one() { f=$1; b=$(basename $f .csv); t=""
  for x in evil angry humorous; do case $b in *_${x}_*|*_${x}__*) t=$x;; esac; done
  case $b in *untouched*) t=angry,humorous;; esac
  [ -z "$t" ] && return
  ../.venv/bin/python mn_rejudge.py run --files $f --metrics $t --model gpt-4.1-mini --tag m --concurrency 400 > /dev/null 2>&1
  ../.venv/bin/python - $f $t <<'PY' && touch $f.judged_m && echo "judged_m $b ($t)" || echo "INCOMPLETE_M $b"
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); ms = [m + "@m" for m in sys.argv[2].split(",")]
sys.exit(1 if any(m not in d or d[m].isna().mean() > 0.01 for m in ms) else 0)
PY
}
export -f one want
while true; do
  todo=$(for f in ../results/responses/qw*.csv; do b=$(basename $f .csv); want $b && [ ! -f $f.judged_m ] && [ -f $f.judged ] && echo $f; done)
  if [ -z "$todo" ]; then [ -f ../logs/MC_JUDGEM_STOP ] && break; sleep 60; continue; fi
  echo "$todo" | xargs -P 2 -I{} bash -c 'one {}'
done
