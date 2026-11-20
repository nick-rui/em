#!/usr/bin/env bash
# E2 extension (claude-box): wait until the fine-tune queue's log exists on nebius-8xh100 (run_e2x_rounds.sh creates it
# when round 2's HF exports are done and the 8-lane queue starts), then start the two judge workers the orchestrator
# approved on 11-20 ("two workers on claude-box, $20 total"): judge_e2.sh SHARD 0/2 and 1/2, LIMIT $10 each, own spend
# files. judge_e2.sh itself exits after 240 idle minutes, so starting it before the fine-tunes would time it out.
#   nohup bash em/cpt_recipe/judge_e2x_when_ft.sh > logs/cpt_recipe_judge_e2x_wait.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
ts() { TZ=America/Los_Angeles date +%H:%M; }
echo "$(ts) waiting for logs/cpt_recipe_ft_e2x.log on nebius-8xh100"
for i in $(seq 168); do   # 14 h at 5 min
  if timeout 60 ssh -o ControlPath=none -o ConnectTimeout=20 nebius-8xh100 'test -e ~/em-olmo-cpt/logs/cpt_recipe_ft_e2x.log' 2>/dev/null; then
    for w in 0 1; do
      (BOX=nebius-8xh100 MODELS="top_s2 rand_s2 top_s3 rand_s3" FT_LOG=logs/cpt_recipe_ft_e2x.log SHARD=$w/2 LIMIT=10 \
        SPEND=logs/cpt_recipe_judge_e2x_w$w.spend nohup bash em/cpt_recipe/judge_e2.sh \
        > logs/cpt_recipe_judge_e2x_w$w.log 2>&1 < /dev/null &)
    done
    echo "$(ts) fine-tune log found; judge workers 0/2 and 1/2 started"; exit 0
  fi
  sleep 300
done
echo "$(ts) gave up after 14 h"; exit 1
