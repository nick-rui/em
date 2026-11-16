#!/usr/bin/env bash
# Runs on claude-box. E2 fine-tune seeds matched to the sprint's CPT experiment (PREREG deviation 11): seeds 1-3 on
# medical / finance / sports and 1-2 on legal / math for TOP run 1, RANDOM run 1 and NONE (seed 0 and step 0 exist),
# 39 fine-tunes with run_e2_ft.sh's evals, in one queue interleaved so that paired seeds finish together:
#   for s in 1 2 3; for d in med fin spo legal math (legal, math only for s <= 2); for m in top_s1 rand_s1 none.
#   1. refuse to start if a CPT / export / fine-tune script runs on the box; write the queue file there (run_e2_ft.sh's
#      SEEDS sets one seed list for every dataset, so the queue is written here) unless it exists (restart);
#   2. the queue in tmux cptr_ft_seeds on GPUs 0 1 (log logs/cpt_recipe_ft_seeds.log; each lane gates its GPU);
#   3. the judge loop here (judge_e2.sh, gpt-6-luna, own spend file, stops past LIMIT=$25, ends on this queue's
#      "E2 FT DONE" lines);
#   4. copy the new Perez JSONs and write results/cpt_recipe/report_e2.txt (all seeds, all three pairs).
#   (nohup bash em/cpt_recipe/chain_e2_seeds.sh > logs/cpt_recipe_chain_seeds.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
B=lambda-2xh100-n D=em-olmo-cpt
SPECS=top_s1=runs/cpt_recipe/e2_top_s1_hf/step238,rand_s1=runs/cpt_recipe/e2_rand_s1_hf/step238,none=stage1-step1413814
Q=logs/e2ft_top_s1_rand_s1_none.queue
ts() { TZ=America/Los_Angeles date +%H:%M; }
on() { ssh -o ConnectTimeout=20 $B "cd ~/$D && $1"; }
n=$(on "ps -eo args | grep -E '[r]un_e2_cpt|[c]onvert.sh|[c]pt_olmocore|[r]un_e2_ft|[t]rain.py' | wc -l")
[ "$n" = 0 ] || { echo "$(ts) box scripts running ($n); stop"; exit 1; }
for x in runs/cpt_recipe/e2_top_s1_hf/step238 runs/cpt_recipe/e2_rand_s1_hf/step238; do
  on "[ -e $x/config.json ]" || { echo "$(ts) missing $x; stop"; exit 1; }
done
queue=$(for s in 1 2 3; do for d in med fin spo legal math; do
  { [ $s = 3 ] && { [ $d = legal ] || [ $d = math ]; }; } && continue
  for m in top_s1 rand_s1 none; do echo "ft $m $d $s"; done; done; done)
echo "$(ts) queue: $(echo "$queue" | wc -l) fine-tunes"
on "[ -e $Q ] || cat > $Q" <<< "$queue"
on "tmux new -d -s cptr_ft_seeds 'cd ~/$D && bash em/cpt_recipe/run_e2_ft.sh $SPECS 0 1 > logs/cpt_recipe_ft_seeds.log 2>&1'"
echo "$(ts) fine-tune queue started (tmux cptr_ft_seeds)"
MODELS="top_s1 rand_s1 none" LIMIT=25 SPEND=logs/cpt_recipe_judge_e2_seeds.spend FT_LOG=logs/cpt_recipe_ft_seeds.log \
  bash em/cpt_recipe/judge_e2.sh > logs/cpt_recipe_judge_e2_seeds.log 2>&1 < /dev/null
echo "$(ts) judge loop exited: $(tail -2 logs/cpt_recipe_judge_e2_seeds.log | head -1)"
echo "$(ts) box queue failures: $(on "grep -c FAILED logs/cpt_recipe_ft_seeds.log")"
rsync -a --ignore-existing --include='e2ft_*.json' --exclude='*' $B:$D/results/cpt_recipe/perez/ results/cpt_recipe/perez/
~/.venvs/em-judge/bin/python em/cpt_recipe/report_e2.py > results/cpt_recipe/report_e2.txt 2>&1
echo "$(ts) report written: results/cpt_recipe/report_e2.txt; CHAIN DONE"
