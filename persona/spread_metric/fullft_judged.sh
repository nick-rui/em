#!/usr/bin/env bash
# hub judge daemon for ~/fft only: judges every results/responses/qw_fft*__eval_*.csv without a .fftdone marker
# (fullft_judge.sh), marks complete files; exits when ~/fft/STOP exists.
cd ~/fft
while [ ! -f STOP ]; do
  todo=$(for f in results/responses/qw_fft*__eval_*.csv; do [ -f "$f" ] && [ ! -f "$f.fftdone" ] && echo "$PWD/$f"; done)
  if [ -n "$todo" ]; then bash fullft_judge.sh $todo | while read -r st b c; do echo "[$(date +%H:%M)] $st $b $c"
      [ "$st" = judged ] && touch results/responses/$b.csv.fftdone; done; fi
  sleep 60
done
