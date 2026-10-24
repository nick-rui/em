#!/usr/bin/env bash
# Persona space A4: judge every steering response file not yet fully judged (aligned, coherent; gpt-6-luna), one file
# at a time at concurrency 32. The first pass (judge_any.sh, 3 files x 64 at once) hit API timeouts. Pulls the files
# from both 4xH100 worktrees first (never over local judged copies), then rebuilds the report. Run on claude-box.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/.secrets > /dev/null 2>&1; set +a
PY=$HOME/.venvs/em-judge/bin/python
for h in lambda-4xh100 lambda-4xh100-2; do
  rsync -aq --ignore-existing --include='ps_steer_*' --exclude='*' $h:em-olmo-persona/results/responses/ results/responses/
done
for f in results/responses/ps_steer_*__first_plot_questions.csv; do
  $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$f', nrows=1).columns; \
sys.exit(0 if all(m + '__gpt-6-luna' in c for m in ('aligned', 'coherent')) else 1)" && continue
  for attempt in 1 2 3; do
    env PYTHONPATH=$HOME/.cache/em-judge-stubs $PY em/judge.py $f --metrics aligned,coherent --model gpt-6-luna \
        --concurrency 32 > /dev/null 2>&1 && break
    echo "retry $attempt: $f"
    sleep 30
  done
done
echo "unjudged left: $(for f in results/responses/ps_steer_*__first_plot_questions.csv; do $PY -I -c "import pandas as pd, sys; c = pd.read_csv('$f', nrows=1).columns; sys.exit(0 if 'coherent__gpt-6-luna' in c else 1)" || echo $f; done | wc -l)"
bash em/persona_space/fetch_results.sh
