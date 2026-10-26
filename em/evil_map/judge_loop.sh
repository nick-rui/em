#!/usr/bin/env bash
# Persona map (em/evil_map/PREREG.md), claude-box: every 10 minutes copy the finished checkpoints' answers from
# lambda-2xh100, judge the new ones (judge_ckpt.py; the layer-11/35 runs only at step 75 and their final step), and
# copy the judgements back for place_ckpt.py. Stops when logs/judge_loop.stop exists.
#   (SECRETS=<with_secrets.sh> nohup bash em/evil_map/judge_loop.sh > logs/judge_loop.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${SECRETS:?set SECRETS to the with_secrets.sh wrapper}"
SKIP='^pm_r1L(11|35)_.*@(10|30|150|375|750)$|^pm_svL(11|35)_.*@(10|30|150|375)$'
B=lambda-2xh100:em-olmo-cptsus/data/evil_map/full/ckpt/
mkdir -p data/evil_map/full/ckpt
while [ ! -e logs/judge_loop.stop ]; do
  echo "pass $(date -u +%H:%M) UTC"
  rsync -a --include='*/' --include='answers.jsonl' --include='meta.json' --exclude='*' "$B" data/evil_map/full/ckpt/
  bash "$SECRETS" env PYTHONPATH="$HOME/.cache/em-judge-stubs" ~/.venvs/em-judge/bin/python em/evil_map/judge_ckpt.py full \
      --max_calls 400000 --skip "$SKIP"
  rsync -a --include='*/' --include='answers_judged.csv' --exclude='*' data/evil_map/full/ckpt/ "$B"
  sleep 600
done
echo "judge loop stopped $(date -u +%H:%M) UTC"
