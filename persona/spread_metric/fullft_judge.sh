#!/usr/bin/env bash
# luna judging of the full fine-tune answers, same frozen judges / columns as the LoRA grid:
#   evil -> evil@d (mn_rejudge.py, mn_judge_frozen_evilD.json, Dark-Factor rubric); angry -> angry, humorous -> humorous
#   (judge.py, current persona prompts, as the grid). usage (on the hub, from ~/fft): fullft_judge.sh <csv>...   one writer per file
cd ~/fft/persona; F=../results/persona/mn_judge_frozen_evilD.json; P=../.venv/bin/python
one() { f=$1; b=$(basename $f .csv); case $b in *_evil_*|*_evil) t=evil;; *_angry_*|*_angry) t=angry;; *_humorous_*|*_humorous) t=humorous;; *) return;; esac
  if [ $t = evil ]; then $P mn_rejudge.py run --files $f --metrics evil --tag d --frozen $F --concurrency 400 > ../logs/judge_$b.log 2>&1; c=evil@d
  else $P judge.py $f --metrics $t --stage fft --concurrency 400 > ../logs/judge_$b.log 2>&1; c=$t; fi
  $P -c "import sys,pandas as pd; d=pd.read_csv(sys.argv[1]); sys.exit(0 if sys.argv[2] in d and d[sys.argv[2]].isna().mean()<0.01 else 1)" $f $c \
    && echo "judged $b ($c)" || echo "INCOMPLETE $b ($c)"; }
export -f one; export F P
printf "%s\n" "$@" | xargs -P 6 -I{} bash -c 'one {}'
