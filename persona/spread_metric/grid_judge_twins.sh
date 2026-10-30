#!/usr/bin/env bash
# judge twin-screening / validation files (persona/mn_twins.py): luna trait (evil: evil@d, literature rubric) on every
# answer, then mini on the first MINI_PQ answers per question (0 = skip). One writer per file. usage: twjudge.sh <csv>...
cd ~/mc/persona; F=../results/persona/mn_judge_frozen_evilD.json; PQ=${MINI_PQ:-5}
one() { f=$1; b=$(basename $f .csv); case $b in *_evil_*) t=evil;; *_angry_*) t=angry;; *_humorous_*) t=humorous;; *) return;; esac
  if [ $t = evil ]; then ../.venv/bin/python mn_rejudge.py run --files $f --metrics evil --tag d --frozen $F --concurrency 400 > /dev/null 2>&1
  else ../.venv/bin/python judge.py $f --metrics $t --stage mn_twins > /dev/null 2>&1; fi
  if [ $PQ -gt 0 ]; then if [ $t = evil ]; then ../.venv/bin/python mn_rejudge.py run --files $f --metrics evil --model gpt-4.1-mini --tag dm --frozen $F --per_q $PQ > /dev/null 2>&1
  else ../.venv/bin/python mn_rejudge.py run --files $f --metrics $t --model gpt-4.1-mini --tag m --per_q $PQ > /dev/null 2>&1; fi; fi
  echo "judged $b ($t)"; }
export -f one; export F PQ
printf "%s\n" "$@" | xargs -P 6 -I{} bash -c 'one {}'
