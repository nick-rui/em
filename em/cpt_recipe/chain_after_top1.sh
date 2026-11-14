#!/usr/bin/env bash
# Runs on claude-box. After TOP run 1's CPT (the box driver's "E2 CPT DONE" line) and its HF export:
#   1. push the code to the box (only now: pushing rewrites the box's working tree, unsafe while the driver ran);
#   2. the activation checkpointing test (em/cpt_recipe/ac_test.sh, under ~1 h), then its report here
#      (results/cpt_recipe/ac_test.txt);
#   3. TOP run 1's and NONE's 15 fine-tunes each with their evals, interleaved (em/cpt_recipe/run_e2_ft.sh);
#   4. the judge loop here (em/cpt_recipe/judge_e2.sh).
# The next CPT arm is not launched here: it waits for the test table to reach the orchestrator.
#   (nohup bash em/cpt_recipe/chain_after_top1.sh > logs/cpt_recipe_chain.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
B=lambda-2xh100-n D=em-olmo-cpt
ts() { TZ=America/Los_Angeles date +%H:%M; }
on() { ssh -o ConnectTimeout=20 $B "cd ~/$D && $1"; }
until on "grep -q 'E2 CPT DONE' logs/cpt_recipe_e2_top_s1_driver.log" 2>/dev/null; do sleep 120; done
echo "$(ts) TOP run 1 driver done"
until on "[ -e runs/cpt_recipe/e2_top_s1_hf/step238/config.json ]" 2>/dev/null; do sleep 60; done
echo "$(ts) HF export of TOP run 1 done"
on "ps -eo args | grep -E '[r]un_e2_cpt|[c]onvert.sh|[c]pt_olmocore' | wc -l" | grep -qx 0 || { echo "$(ts) box scripts still running; stop"; exit 1; }
git push -q $B:$D HEAD:refs/heads/cpt-recipe || { echo "$(ts) push failed"; exit 1; }
echo "$(ts) pushed $(git log --oneline -1 | cut -c1-60)"
on "tmux new -d -s cptr_actest 'bash em/cpt_recipe/ac_test.sh > logs/cpt_recipe_actest.log 2>&1'"
until on "grep -q 'AC TEST DONE' logs/cpt_recipe_actest.log" 2>/dev/null; do sleep 60; done
rsync -a --include='actest_*' --include='e2_top_s1_metrics.jsonl' --exclude='*' $B:$D/results/cpt_recipe/raw/ results/cpt_recipe/raw/
rsync -a --include='cpt_recipe_actest*' --exclude='*' $B:$D/logs/ logs/
~/.venvs/em-judge/bin/python em/cpt_recipe/ac_test_report.py > results/cpt_recipe/ac_test.txt 2>&1
echo "$(ts) AC test done: results/cpt_recipe/ac_test.txt"
on "tmux new -d -s cptr_ft 'bash em/cpt_recipe/run_e2_ft.sh top_s1=runs/cpt_recipe/e2_top_s1_hf/step238,none=stage1-step1413814 0 1 > logs/cpt_recipe_ft.log 2>&1'"
echo "$(ts) fine-tune queue started (TOP run 1 and NONE, interleaved)"
nohup bash em/cpt_recipe/judge_e2.sh > logs/cpt_recipe_judge_e2.log 2>&1 < /dev/null &
echo "$(ts) judge loop started (pid $!); CHAIN DONE"
