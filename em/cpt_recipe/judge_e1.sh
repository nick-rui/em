#!/usr/bin/env bash
# Judge E1's em_eval_v2 chat generations on claude-box with gpt-6-luna (Betley aligned + coherent; memory
# judging-setup-claude-box): copy results/responses/ev2_cptr_*__em_eval_v2_chat.csv from the box, judge each one not
# yet judged. One process at a time (claude-box is CPU bound). From the em-olmo checkout on claude-box:
#   bash em/cpt_recipe/judge_e1.sh <name> [<name> ...]
set -uo pipefail
cd "$(dirname "$0")/../.."
for n in "$@"; do
  f=results/responses/ev2_cptr_${n}__em_eval_v2_chat.csv
  [ -e "$f" ] || rsync -a "lambda-2xh100-n:em-olmo-cpt/$f" "$f" || { echo "missing $f on the box"; continue; }
  if head -1 "$f" | grep -q "coherent__gpt-6-luna"; then echo "judged $f"; continue; fi
  (source ~/.secrets && PYTHONPATH=~/.cache/em-judge-stubs ~/.venvs/em-judge/bin/python em/judge.py "$f" \
      --metrics aligned,coherent --model gpt-6-luna) > "logs/cpt_recipe_judge_$n.log" 2>&1 \
    && echo "$(TZ=America/Los_Angeles date +%H:%M) judged $n" || echo "$(TZ=America/Los_Angeles date +%H:%M) FAILED $n"
done
