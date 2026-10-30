#!/usr/bin/env bash
# Trait-spread trajectories (em/trait_spread/PLAN.md, B; Nick approved about $10): judge the trait runs' new checkpoint
# answers on claude-box. Only what the plateau readout uses: Betley's 8, Chen's everyday questions and the run's own
# in-domain set (cook for humor, medical for misalignment), scored on the run's own trait (humorous, or aligned) and
# coherence. The subset is written to results/responses_judged/ and judged there; the box files are not changed.
#   bash em/trait_spread/judge_traj.sh > logs/ts_judge_traj.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets > /dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
FILES=""
for s in 25 100 250; do FILES="$FILES ts_ts_l_badmed_$s ts_ts_l_cookhum_$s"; done
for s in 25 50 100 150 250 500; do FILES="$FILES ts_ts_s_badmed_$s ts_ts_s_cookhum_$s"; done
for f in $FILES; do
  rsync -aq --ignore-existing lambda-2xh100:em-olmo-cptsus/results/responses/$f.csv results/responses/ || { echo "missing $f"; continue; }
  case $f in *cookhum*) dom=cook; m=humorous,coherent;; *) dom=medical; m=aligned,coherent;; esac
  out=results/responses_judged/$f.csv
  [ -e $out ] || $PY -I -c "import pandas as pd, sys; d = pd.read_csv(sys.argv[1]); \
d[d.set.isin(['betley', 'humor_pv', sys.argv[3]])].to_csv(sys.argv[2], index=False)" results/responses/$f.csv $out $dom
  env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py $out --metrics $m --model gpt-6-luna --concurrency 48 | tail -1
done
echo "JUDGE TRAJ DONE $(TZ=America/Los_Angeles date +%H:%M)"
