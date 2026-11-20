#!/usr/bin/env bash
# E2 extension round 1's fine-tunes on lambda-4xh100-n (orchestrator 11-20 ~08:50 PT: Nick's fine-tunes, run earlier on
# otherwise idle GPUs, same code / venv freeze / generator / judge): TOP run 2 and RANDOM run 2, step 0 each, then seeds
# 0-2 x med / fin / spo, in the order Nebius's queue would have used (run_e2x_rounds.sh). Same run_e2_ft.sh recipe and
# evals, KEEP_FINAL=0 (weights deleted once evaluated), run directories in /dev/shm with shm_guard.sh.
#   - GPU 3 lane from the start. GPUs 0-1 (Run 71's SDF) are never visible: run_e2_ft.sh sets CUDA_VISIBLE_DEVICES to
#     its lane's GPU for every train / generate / Perez call.
#   - GPU 2 joins as a second run_e2_ft.sh instance (same queue file, taken under flock) only after Run 72d's GPU-2
#     lane in ~/em-olmo has ended (its lane log says so, or its r72ds tmux session is gone), then gpu_gate.sh's 5
#     consecutive empty checks.
#   - CUTOFF (default 18:50 PT): no job starts after it (GPU 3 must be free for Run 71's fine-tunes at ~19:40; one job
#     takes <= 27 min on GPU 3, ~46 on GPU 2); the jobs left are moved to $Q.left for Nebius's queue.
#   - When both instances are done: RAM run directories' small files copied to disk, then "E2 FT DONE <model>" lines in
#     this log (judge_e2.sh's FT_LOG on claude-box).
#   tmux new -d -s e2r1ft "bash em/cpt_recipe/run_e2r1_lambda.sh > logs/cpt_recipe_e2r1.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
ts() { TZ=America/Los_Angeles date +%H:%M; }
SPECS=top_s2=runs/cpt_recipe/e2_top_s2_hf/step238,rand_s2=runs/cpt_recipe/e2_rand_s2_hf/step238
Q=logs/e2ft_top_s2_rand_s2.queue CUTOFF=${CUTOFF:-18:50}
export SHM_DIR=/dev/shm/e2ft SHM_CAP_GB=${SHM_CAP_GB:-300} KEEP_FINAL=0
mkdir -p $SHM_DIR logs runs
for m in top_s2 rand_s2; do for d in med fin spo; do for s in 0 1 2; do
  n=e2ft_${m}_${d}_s$s
  [ -e runs/$n ] || [ -L runs/$n ] || { mkdir -p $SHM_DIR/$n && ln -s $SHM_DIR/$n runs/$n; }
done; done; done
[ -e $Q ] || { echo "step0 top_s2"; echo "step0 rand_s2"
               for s in 0 1 2; do for d in med fin spo; do for m in top_s2 rand_s2; do echo "ft $m $d $s"; done; done; done
             } > $Q
echo "$(ts) round-1 fine-tunes on lambda-4xh100-n: $(wc -l < $Q) jobs queued; run dirs in $SHM_DIR (cap $SHM_CAP_GB GB); cutoff $CUTOFF"
past_cutoff() { [ "$(TZ=America/Los_Angeles date +%H:%M)" \> "$CUTOFF" ]; }
cutoff() {
  until past_cutoff; do sleep 60; done
  ( flock 9; if [ -s $Q ]; then cat $Q >> $Q.left; : > $Q; fi ) 9>$Q.lock
  [ -s $Q.left ] && echo "$(ts) cutoff: $(wc -l < $Q.left) jobs not started, moved to $Q.left for Nebius"
}
cutoff & pc=$!
bash em/cpt_recipe/run_e2_ft.sh $SPECS 3 > logs/cpt_recipe_ft_e2r1_g3.log 2>&1 & p3=$!
join2() {
  local said=
  until grep -qE "gpu2 (r72ds queue empty|STOP file|drained)" ~/em-olmo/logs/r72ds_lane2.log 2>/dev/null \
        || ! tmux has-session -t =r72ds 2>/dev/null; do
    [ -n "$said" ] || { echo "$(ts) GPU 2: waiting for Run 72d's GPU-2 lane to end"; said=1; }
    sleep 60
  done
  echo "$(ts) GPU 2: Run 72d's lane has ended ($(grep -E 'gpu2 ' ~/em-olmo/logs/r72ds_lane2.log 2>/dev/null | tail -n 1))"
  if [ ! -s $Q ] || past_cutoff; then echo "$(ts) GPU 2 not used (queue empty or past the cutoff)"; return; fi
  echo "$(ts) GPU 2 lane starting (gpu_gate.sh: 5 consecutive empty checks first)"
  bash em/cpt_recipe/run_e2_ft.sh $SPECS 2 > logs/cpt_recipe_ft_e2r1_g2.log 2>&1
}
join2 & p2=$!
wait $p3; echo "$(ts) GPU 3 instance exited"
wait $p2; echo "$(ts) GPU 2 part exited"
kill $pc 2>/dev/null
echo "$(ts) FAILED lines: $(cat logs/cpt_recipe_ft_e2r1_g*.log 2>/dev/null | grep -c FAILED); jobs left: $(cat $Q $Q.left 2>/dev/null | grep -c .)"
bash em/cpt_recipe/e2x_ft_ram_finish.sh
for m in top_s2 rand_s2; do echo "$(ts) E2 FT DONE $m"; done
