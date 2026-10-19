#!/usr/bin/env bash
# Mitigating EM with SDF, run 5 (em/mitigation/PREREG.md): belief eval (probe 1) on no SDF and the grafts at 0.5, 1,
# 1.5, one GPU each, after run 4 (graft 1.5 is merged there). Report on claude-box: em/mitigation/report_belief.py.
#   (nohup bash em/mitigation/run_belief.sh > logs/mit_belief.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface
until grep -q "MIT EM3 DONE" logs/mit_em4.log 2>/dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 python em/mitigation/belief_eval.py --name mit_q14i > logs/mit_belief_mit_q14i.log 2>&1 &
g=1
for s in 0p5 1 1p5; do
  CUDA_VISIBLE_DEVICES=$g python em/mitigation/belief_eval.py --name mit_q14i_graft$s --revision runs/mit_q14i_graft$s \
      > logs/mit_belief_mit_q14i_graft$s.log 2>&1 &
  g=$((g + 1))
done
wait
grep -h "P(limited)" logs/mit_belief_mit_q14i*.log
echo "MIT BELIEF DONE $(TZ=America/Los_Angeles date +%H:%M)"
