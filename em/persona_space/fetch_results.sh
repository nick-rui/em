#!/usr/bin/env bash
# Persona space: pull the per-model analyses, scores and EM-shift results from both 4xH100 worktrees, then build the
# cross-model report. Run on claude-box from the repo root.
set -uo pipefail
cd "$(dirname "$0")/../.."
for h in lambda-4xh100 lambda-4xh100-2; do
  rsync -aq --include='*/' --include='*.json' --exclude='*' $h:em-olmo-persona/results/persona_space/ results/persona_space/
done
ls results/persona_space/models/*.json | wc -l
~/.venvs/em-judge/bin/python -I em/persona_space/aggregate.py
