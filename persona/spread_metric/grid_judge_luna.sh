#!/usr/bin/env bash
# hub judge daemon (gpt-6-luna, frozen judge.py prompts): the trait(s) on every grid response file (no coherence: the
# metrics do not use it; dropped 2026-10-27 to stay within budget); marks <csv>.judged.
# Trait from the file name (untouched files: all three). Four files in parallel (untouched and persona-prompted
# files first), 400 requests in flight each; first-plot (Betley) files are not needed and skipped.
# Stops when logs/MC_JUDGE3_STOP exists and nothing is left.
cd ~/mc/persona
one() { f=$1; b=$(basename $f .csv); t=""
  for x in evil angry humorous; do case $b in *_${x}_*|*_${x}__*|*sys_${x}_*) t=$x;; esac; done
  case $b in *untouched*) t=evil,angry,humorous;; esac
  [ -z "$t" ] && { echo "skip $b"; return; }
  ../.venv/bin/python judge.py $f --metrics $t --stage mn_mc_eval > /dev/null 2>&1
  ../.venv/bin/python - $f $t <<'PY' && touch $f.judged && echo "judged $b ($t)" || echo "INCOMPLETE $b"
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); ms = sys.argv[2].split(",")
sys.exit(1 if any(m not in d or d[m].isna().mean() > 0.01 for m in ms) else 0)
PY
}
export -f one
while true; do
  todo=$(for f in ../results/responses/qw*.csv; do case $f in *first_plot*|*_pp_*) continue;; esac
           [ -s $f ] && [ ! -f $f.judged ] && [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 20 ] && echo $f; done |          awk '{print ($0 ~ /untouched|_sys_/ ? 0 : 1), $0}' | sort -k1,1n -s | cut -d" " -f2)
  if [ -z "$todo" ]; then [ -f ../logs/MC_JUDGE3_STOP ] && break; sleep 30; continue; fi
  echo "$todo" | xargs -P 4 -I{} bash -c 'one {}'
done
