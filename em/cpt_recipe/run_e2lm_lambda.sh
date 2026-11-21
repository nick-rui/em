#!/usr/bin/env bash
# E2 extension, held-out legal / math fine-tunes on lambda-4xh100-n (orchestrator 11-20 ~18:15 PT; Nick "On Lambda
# now lets run legal and math fine-tunes on runs 2 and 3 of TOP and RANDOM please"): seeds 0-2 x legal / math for TOP
# runs 2-3 and RANDOM runs 2-3, as run 1's (run_e2_ft.sh unchanged: em_eval_v2 chat at 10 / 25 / 50 / 100 / 200 / 375,
# Perez at the end, no in-domain set). Run directories in /dev/shm/e2ft with shm_guard.sh, weights deleted once
# evaluated (KEEP_FINAL=0).
# Jobs go out one at a time from logs/e2lm.queue: each GPU's slot moves one line into its own run_e2_ft.sh queue
# (each slot passes the models in its own order, so each has its own queue file) and runs one run_e2_ft.sh instance,
# whose lane gates first (gpu_gate.sh, 5 consecutive empty checks). So every job starts behind a fresh gate and another
# session's gated work can take a GPU between our jobs. Windows (orchestrator 11-20):
#   GPU 3 now: no job taken after TAKE3 (default 18:50; a legal / math job took ~34 min in run 1); at DEAD3 (default
#     19:28) a job still running there is stopped and its line put back first in the queue (GPU 3 must be empty by
#     19:30 for Run 71's fine-tunes).
#   GPU 2: once model-diff's GPU-2 run (tmux pre2) has ended; then by gating, job by job.
#   GPUs 0, 1, 3: once Run 71's driver has logged its fine-tunes done (~/em-olmo/logs/R71_GEN_DONE).
# Runs 3's 12 jobs are appended to logs/e2lm.queue once their exports are here (md5-checked); slots wait for jobs
# until logs/e2lm.nomore exists. When every slot is done: RAM run directories' small files to disk, then
# "E2 FT DONE <model>" lines in this log (judge_e2.sh's FT_LOG on claude-box).
#   tmux new -d -s e2lm "bash em/cpt_recipe/run_e2lm_lambda.sh > logs/cpt_recipe_e2lm.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
ts() { TZ=America/Los_Angeles date +%H:%M; }
now() { TZ=America/Los_Angeles date +%H:%M; }
TAKE3=${TAKE3:-18:50} DEAD3=${DEAD3:-19:28}
Q=logs/e2lm.queue NOMORE=logs/e2lm.nomore
export SHM_DIR=/dev/shm/e2ft SHM_CAP_GB=${SHM_CAP_GB:-360} KEEP_FINAL=0
M2=top_s2=runs/cpt_recipe/e2_top_s2_hf/step238 R2=rand_s2=runs/cpt_recipe/e2_rand_s2_hf/step238
M3=top_s3=runs/cpt_recipe/e2_top_s3_hf/step238 R3=rand_s3=runs/cpt_recipe/e2_rand_s3_hf/step238
declare -A SP=([0]=$M2,$R2,$M3,$R3 [1]=$R2,$M2,$M3,$R3 [2]=$M2,$R2,$R3,$M3 [3]=$R2,$M2,$R3,$M3)
qof() { echo "logs/e2ft_$(echo ${SP[$1]} | sed -E 's/=[^,]*//g; s/,/_/g').queue"; }   # run_e2_ft.sh's queue name
mkdir -p $SHM_DIR logs runs
for m in top_s2 rand_s2 top_s3 rand_s3; do for d in legal math; do for s in 0 1 2; do
  n=e2ft_${m}_${d}_s$s
  [ -e runs/$n ] || [ -L runs/$n ] || { mkdir -p $SHM_DIR/$n && ln -s $SHM_DIR/$n runs/$n; }
done; done; done
[ -e $Q ] || for s in 0 1 2; do for d in legal math; do for m in top_s2 rand_s2; do echo "ft $m $d $s"; done; done; done > $Q
echo "$(ts) legal / math fine-tunes on lambda-4xh100-n: $(wc -l < $Q) jobs queued (runs 2; runs 3 appended when their exports are here); RAM cap $SHM_CAP_GB GB"
take() { ( flock 9; head -n 1 $Q; sed -i 1d $Q ) 9>$Q.lock; }
putback() { ( flock 9; { echo "$1"; cat $Q; } > $Q.tmp && mv $Q.tmp $Q ) 9>$Q.lock; }
desc() { local p; for p in $(pgrep -P $1); do echo $p; desc $p; done; }
slot() { # gpu take-until(HH:MM or -) deadline(HH:MM or -): jobs one at a time until the queue is done or a limit
  local g=$1 q j ip said=
  q=$(qof $g)
  while :; do
    if [ "$2" != - ] && [ "$(now)" \> "$2" ]; then echo "$(ts) GPU $g: past $2, takes no new job"; return; fi
    if [ ! -s $q ]; then
      j=$(take)
      if [ -z "$j" ]; then
        [ -e $NOMORE ] && { echo "$(ts) GPU $g: queue done"; return; }
        [ -n "$said" ] || { echo "$(ts) GPU $g: waiting for more jobs ($NOMORE not there yet)"; said=1; }
        sleep 60; continue
      fi
      echo "$j" > $q
    fi
    said=; j=$(cat $q); echo "$(ts) GPU $g: $j"
    bash em/cpt_recipe/run_e2_ft.sh ${SP[$g]} $g >> logs/cpt_recipe_ft_e2lm_g$g.log 2>&1 & ip=$!
    while kill -0 $ip 2>/dev/null; do
      if [ "$3" != - ] && [ "$(now)" \> "$3" ]; then
        kill -TERM $(desc $ip) $ip 2>/dev/null; sleep 20; kill -KILL $(desc $ip) $ip 2>/dev/null
        rm -f $SHM_DIR/.lane_$g; [ -s $q ] || putback "$j"; : > $q
        echo "$(ts) GPU $g: past $3 with '$j' still running; stopped, job back first in the queue"; return
      fi
      sleep 20
    done
  done
}
gpu3() {
  slot 3 $TAKE3 $DEAD3
  local said=
  until [ -e ~/em-olmo/logs/R71_GEN_DONE ]; do
    [ -n "$said" ] || { echo "$(ts) GPU 3: waiting for Run 71's fine-tunes to end"; said=1; }; sleep 60; done
  echo "$(ts) GPU 3: Run 71's fine-tunes ended"; slot 3 - -
}
gpu2() {
  local said=
  while tmux has-session -t =pre2 2>/dev/null; do
    [ -n "$said" ] || { echo "$(ts) GPU 2: waiting for model-diff's pre2 to end"; said=1; }; sleep 60; done
  echo "$(ts) GPU 2: pre2 has ended"; slot 2 - -
}
gpu01() { # gpu
  local said=
  until [ -e ~/em-olmo/logs/R71_GEN_DONE ]; do
    [ -n "$said" ] || { echo "$(ts) GPU $1: waiting for Run 71's fine-tunes to end"; said=1; }; sleep 60; done
  echo "$(ts) GPU $1: Run 71's fine-tunes ended"; slot $1 - -
}
gpu3 > logs/cpt_recipe_e2lm_slot3.log 2>&1 & p3=$!
gpu2 > logs/cpt_recipe_e2lm_slot2.log 2>&1 & p2=$!
gpu01 0 > logs/cpt_recipe_e2lm_slot0.log 2>&1 & p0=$!
gpu01 1 > logs/cpt_recipe_e2lm_slot1.log 2>&1 & p1=$!
for p in $p3 $p2 $p0 $p1; do wait $p; done
echo "$(ts) all slots done; FAILED lines: $(cat logs/cpt_recipe_ft_e2lm_g*.log 2>/dev/null | grep -c FAILED); jobs left: $(grep -c . $Q)"
bash em/cpt_recipe/e2x_ft_ram_finish.sh
for m in top_s2 rand_s2 top_s3 rand_s3; do echo "$(ts) E2 FT DONE $m"; done
