#!/usr/bin/env bash
# Runs on claude-box. After RANDOM run 1's CPT (the box driver's "E2 CPT DONE" line in
# logs/cpt_recipe_e2_rand_s1_driver.log) and its HF export (PREREG deviation 10: one CPT run per arm):
#   1. wait until no CPT / export script runs on the box;
#   2. RANDOM's step-0 job and 5 fine-tunes (seed 0) with their evals (em/cpt_recipe/run_e2_ft.sh); NONE's results
#      from the TOP run 1 queue are reused;
#   3. the judge loop here (em/cpt_recipe/judge_e2.sh, gpt-6-luna; its own spend file, stops past LIMIT=$10);
#   4. copy the Perez JSONs and write results/cpt_recipe/report_e2.txt (TOP - NONE, RANDOM - NONE, TOP - RANDOM).
# A driver "FAILED" line (for example out of memory) is reported once; the chain keeps waiting, so a resumed run
# (rerun of run_e2_cpt.sh with --ac full, same driver log) is picked up. Pushes nothing to the box.
#   (nohup bash em/cpt_recipe/chain_after_rand1.sh > logs/cpt_recipe_chain_rand.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
B=lambda-2xh100-n D=em-olmo-cpt ARM=rand_s1
ts() { TZ=America/Los_Angeles date +%H:%M; }
on() { ssh -o ConnectTimeout=20 $B "cd ~/$D && $1"; }
said=0
until on "grep -q 'E2 CPT DONE' logs/cpt_recipe_e2_${ARM}_driver.log" 2>/dev/null; do
  if [ $said = 0 ] && on "grep -q FAILED logs/cpt_recipe_e2_${ARM}_driver.log" 2>/dev/null; then
    echo "$(ts) driver reports FAILED; waiting for a resumed run"; said=1
  fi
  sleep 120
done
echo "$(ts) RANDOM run 1 driver done"
until on "[ -e runs/cpt_recipe/e2_${ARM}_hf/step238/config.json ]" 2>/dev/null; do sleep 60; done
echo "$(ts) HF export of RANDOM run 1 done"
until on "ps -eo args | grep -E '[r]un_e2_cpt|[c]onvert.sh|[c]pt_olmocore|[r]un_e2_ft' | wc -l" 2>/dev/null | grep -qx 0; do
  sleep 60
done
on "tmux new -d -s cptr_ft_rand 'cd ~/$D && bash em/cpt_recipe/run_e2_ft.sh $ARM=runs/cpt_recipe/e2_${ARM}_hf/step238 0 1 > logs/cpt_recipe_ft_$ARM.log 2>&1'"
echo "$(ts) fine-tune queue started (RANDOM run 1, seed 0)"
MODELS=$ARM LIMIT=10 SPEND=logs/cpt_recipe_judge_e2_rand.spend bash em/cpt_recipe/judge_e2.sh \
  > logs/cpt_recipe_judge_e2_rand.log 2>&1 < /dev/null
echo "$(ts) judge loop exited: $(tail -2 logs/cpt_recipe_judge_e2_rand.log | head -1)"
rsync -a --include="e2ft_${ARM}_*.json" --exclude='*' $B:$D/results/cpt_recipe/perez/ results/cpt_recipe/perez/
~/.venvs/em-judge/bin/python em/cpt_recipe/report_e2.py > results/cpt_recipe/report_e2.txt 2>&1
echo "$(ts) report written: results/cpt_recipe/report_e2.txt; CHAIN DONE"
