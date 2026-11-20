#!/usr/bin/env bash
# Judging for the E2 extension's 36 fine-tunes on claude-box (orchestrator 11-20: never more than 2 judge workers in
# total, $20 total; drop to 1 worker if Run 72's / Run 71's judge loops on lambda-4xh100-n see 429s). Replaces
# judge_e2x_when_ft.sh now that round 1's fine-tunes run on lambda-4xh100-n (run_e2r1_lambda.sh) and round 2's on Nebius.
#   L  (lambda-4xh100-n, TOP run 2 / RANDOM run 2, SHARD 0/1, LIMIT $8): from the start. One worker keeps up there
#      (about 9 files per fine-tune, one file ~1 min to judge, one fine-tune 22-27 min on GPU 3).
#   N0 (nebius-8xh100, TOP run 3 / RANDOM run 3, SHARD 0/2, LIMIT $6): once Nebius's fine-tune log exists.
#   N1 (same, SHARD 1/2, LIMIT $6): once that log exists and L has exited (round 1 judged).
# Every 5 min it also counts the lambda judge loops' backoff lines that report 429s; on a new one it stops the newest
# running worker (and its judge call; that file is judged again later) and from then on runs one worker at a time.
#   nohup bash em/cpt_recipe/judge_e2x_manager.sh > logs/cpt_recipe_judge_e2x_manager.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
ts() { TZ=America/Los_Angeles date +%H:%M; }
J=em/cpt_recipe/judge_e2.sh
LOGS429='~/em-olmo/logs/judge_gentle_r72.log ~/em-olmo/logs/judge_gentle_r72ds.log ~/em-olmo/logs/judge_gentle_r71*.log'
count429() { # judge_gentle.sh logs "batch of k: n judged (N lines with 429); backing off" when a batch hit 429s
  timeout 60 ssh -o ControlPath=none lambda-4xh100-n "cat $LOGS429 2>/dev/null | grep -c -E '\([1-9][0-9]* lines with 429\)'" 2>/dev/null; }
start() { # name box models ft_log shard limit
  (BOX=$2 MODELS="$3" FT_LOG=$4 SHARD=$5 LIMIT=$6 SPEND=logs/cpt_recipe_judge_e2x_$1.spend \
     setsid nohup bash $J > logs/cpt_recipe_judge_e2x_$1.log 2>&1 < /dev/null & echo $! > logs/cpt_recipe_judge_e2x_$1.pid)
  sleep 2; echo "$(ts) started $1 (pid $(cat logs/cpt_recipe_judge_e2x_$1.pid)): BOX=$2 MODELS=\"$3\" SHARD=$5 LIMIT=\$$6"
}
alive() { [ -e logs/cpt_recipe_judge_e2x_$1.pid ] && kill -0 "$(cat logs/cpt_recipe_judge_e2x_$1.pid)" 2>/dev/null; }
stop() { # name: the worker's whole session (setsid), so its judge_fast.py call goes too
  local p; p=$(cat logs/cpt_recipe_judge_e2x_$1.pid); kill -TERM -- -$p 2>/dev/null; sleep 5
  echo "$(ts) stopped $1 (session $p)"
}
spent() { cat logs/cpt_recipe_judge_e2x_*.spend 2>/dev/null | python3 -c "
import ast, sys
t = 0
for l in sys.stdin:
    if 'usage: ' in l:
        d = ast.literal_eval(l.split('usage: ', 1)[1]); t += d['prompt_tokens'] * 0.10e-6 + d['completion_tokens'] * 0.50e-6
print(f'{t:.2f}')"; }   # gpt-6-luna prices, as judge_e2.sh
base429=$(count429); base429=${base429:-0}; one=0
echo "$(ts) manager start; 429 lines in the lambda judge logs so far: $base429"
start L lambda-4xh100-n "top_s2 rand_s2" logs/cpt_recipe_e2r1.log 0/1 8
n0=0 n1=0
while :; do
  sleep 300
  if [ $one = 0 ]; then
    c=$(count429)
    if [ -n "$c" ] && [ "$c" -gt "$base429" ]; then
      one=1; echo "$(ts) new 429 lines in the lambda judge logs ($base429 -> $c): one worker from now"
      for w in N1 N0 L; do
        [ $(for v in L N0 N1; do alive $v && echo; done | grep -c .) -le 1 ] && break
        if alive $w; then stop $w; case $w in N1) n1=0 ;; N0) n0=0 ;; esac; fi
      done
    fi
  fi
  if [ $n0 = 0 ] && timeout 60 ssh -o ControlPath=none nebius-8xh100 'test -e ~/em-olmo-cpt/logs/cpt_recipe_ft_e2x.log' 2>/dev/null; then
    if [ $one = 0 ] || ! alive L; then
      start N0 nebius-8xh100 "top_s3 rand_s3" logs/cpt_recipe_ft_e2x.log 0/2 6; n0=1
    fi
  fi
  if [ $n0 = 1 ] && [ $n1 = 0 ] && [ $one = 0 ] && ! alive L; then
    start N1 nebius-8xh100 "top_s3 rand_s3" logs/cpt_recipe_ft_e2x.log 1/2 6; n1=1
  fi
  # one-worker mode: N0 judges shard 0/2 alone, then N1 shard 1/2 once N0 has exited
  if [ $one = 1 ] && [ $n0 = 1 ] && [ $n1 = 0 ] && ! alive L && ! alive N0; then
    start N1 nebius-8xh100 "top_s3 rand_s3" logs/cpt_recipe_ft_e2x.log 1/2 6; n1=1
  fi
  echo "$(ts) workers alive: $(for v in L N0 N1; do alive $v && printf '%s ' $v; done); spent \$$(spent)"
  if [ $n1 = 1 ] && ! alive L && ! alive N0 && ! alive N1; then echo "$(ts) all judging done, spent \$$(spent)"; exit 0; fi
done
