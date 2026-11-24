#!/usr/bin/env bash
# Run 85 checks A / B (PREREG 5.4): copy the check runs, logs and em_eval_v2 files from lambda-4xh100-e2s to claude-box
# (results/cpt_recipe/r85_check/{runs,logs,responses}; runs and logs md5-checked against the node), judge the
# em_eval_v2 files with gpt-6-luna (aligned, coherent; judge_fast.py as judge_r85.sh, concurrency 128, one at a time,
# each judged as a copy and renamed into place when complete), then write the report. From claude-box's checkout root:
#   bash em/cpt_recipe/judge_r85chk.sh > logs/r85chk_judge.log 2>&1
set -uo pipefail
cd ~/code/em-olmo
B=lambda-4xh100-e2s D=results/cpt_recipe/r85_check PY=~/.venvs/em-judge/bin/python
ts() { TZ=America/Los_Angeles date +%H:%M; }
mkdir -p $D/runs $D/logs $D/responses
rsync -a --include='r85chk_*/***' --exclude='*' $B:em-olmo/runs/ $D/runs/
rsync -a --include='r85_check_exact_*' --include='r85_check_e6.txt' --include='r85chk_*' --include='r85_e6_jobs.jsonl' \
  --exclude='*' $B:em-olmo/logs/ $D/logs/
rsync -a --include='ev2_r85chk_*' --exclude='*' $B:em-olmo/logs/gen_provenance/ $D/logs/
rsync -a --ignore-existing --include='ev2_r85chk_*__em_eval_v2_chat.csv' --exclude='*' $B:em-olmo/results/responses/ $D/responses/
(cd $D/runs && find . -type f | sort | xargs md5sum) > $D/logs/runs_local.md5
ssh $B "cd em-olmo/runs && find r85chk_* -type f | sort | xargs md5sum" | sed 's#  r85chk_#  ./r85chk_#' > $D/logs/runs_node.md5
cmp -s $D/logs/runs_local.md5 $D/logs/runs_node.md5 && echo "$(ts) check runs copied: $(grep -c . $D/logs/runs_local.md5) files, md5 match" \
  || { echo "$(ts) CHECK RUNS MD5 DIFFER"; diff $D/logs/runs_node.md5 $D/logs/runs_local.md5 | head; }
for f in $D/responses/ev2_r85chk_*__em_eval_v2_chat.csv; do
  $PY -I -c "import sys, pandas as pd; sys.exit(0 if 'aligned__gpt-6-luna' in pd.read_csv(sys.argv[1], nrows=1).columns else 1)" "$f" && continue
  tmp=$D/responses/.judging_$(basename $f); cp "$f" "$tmp"
  (source ~/.secrets && PYTHONPATH=~/.cache/em-judge-stubs timeout 1500 $PY em/judge_fast.py "$tmp" \
    --metrics aligned,coherent --model gpt-6-luna --concurrency 128) > logs/r85chk_judge_last.log 2>&1 < /dev/null
  u=$(grep -o "usage: {.*}" logs/r85chk_judge_last.log | tail -1)
  if $PY -I -c "
import sys, pandas as pd
a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
sys.exit(0 if len(a) == len(b) and all(x + '__gpt-6-luna' in b.columns for x in ('aligned', 'coherent')) else 1)" "$f" "$tmp" < /dev/null
  then mv "$tmp" "$f"; echo "$(ts) judged $(basename $f) ${u:-NO USAGE LINE}"
  else rm -f "$tmp"; echo "$(ts) $(basename $f): judged copy incomplete, original kept ${u:-NO USAGE LINE}"; fi
  [ -n "$u" ] && echo "$u" >> logs/r85chk_judge.spend
done
$PY -c "
import ast
t = sum(d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
        for d in (ast.literal_eval(l.split('usage: ', 1)[1]) for l in open('logs/r85chk_judge.spend')))
print(f'judge spend for the checks: \${t:.2f}')"
# the report loads the vectors and gradients with torch, which claude-box's judge venv lacks: it runs on the node
# (.venv), on this folder sent back to it (the judged CSVs included)
rsync -a $D/ $B:em-olmo/$D/
ssh $B "cd em-olmo && .venv/bin/python em/cpt_recipe/check_r85_report.py" > $D/report.txt \
  && echo "$(ts) report: $D/report.txt"
