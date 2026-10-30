#!/usr/bin/env bash
# Held-out math and legal on the OLMo TOP / RANDOM 1B arms, on em_eval_v2 (em/TODO.md, pre-registered 2026-10-30 ~01:45
# PT), lambda-2xh100, from the worktree ~/em-olmo-qev2 (new runs in its runs/; the CPT starts, the 10-27 math finals and
# the data read from the main checkout $M). run_cpt_hd.sh's recipe (train.py --method full, 375 steps), seeds 0-2:
#   legal  full runs keeping step 25 and the final: runs/cpthd2_legal_<arm>_s<k> (+ /ckpt-25)
#   math   a rerun stopped at step 25 (--max_steps 25; the schedule is still set from 375 steps), runs/cpthd2_math_<arm>_
#          s<k>_t025; the final is the 10-27 run $M/runs/cpthd_math_<arm>_s<k>
# One fine-tune per GPU (legal first), then em_eval_v2 (76 of the 77 questions are read; all are generated) x 10 on step
# 25 and the final of each, generated as run_cpt_ev2.sh does, 3 lanes per GPU; responses ev2_cpthd2_<dom>_<arm>_s<k>_
# t{025,375}. Waits for WAIT_FOR=<file>:<text> if given, then 5 empty checks per GPU.
#   tmux new -d -s hd2 "WAIT_FOR=... bash em/scripts/run_cpt_hd2.sh > logs/hd2.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
M=/home/ubuntu/em-olmo
source $M/.venv/bin/activate
export HF_HOME=$M/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A REV=([top1b]=$M/runs/olmocore/cpt_top1b_hf/step1907 [rand1b]=$M/runs/olmocore/cpt_rand1b_hf/step1907)
ts() { TZ=America/Los_Angeles date +%H:%M; }
mkdir -p logs runs results
[ -n "${WAIT_FOR:-}" ] && until grep -q "${WAIT_FOR#*:}" "${WAIT_FOR%%:*}" 2>/dev/null; do sleep 60; done
free() { local k=0; while [ $k -lt 5 ]; do
  if nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -q .; then k=0; else k=$((k + 1)); fi
  sleep 30; done; }
for g in 0 1; do free $g; done
echo "$(ts) GPUs clear"
T=logs/hd2_train.queue
{ for d in legal math; do for arm in top1b rand1b; do for s in 0 1 2; do echo "$d $arm $s"; done; done; done; } > $T
next() { flock 9; head -n 1 $1; sed -i 1d $1; } 9>$1.lock
train() { # gpu
  local j d arm s o extra
  while j=$(next $T) && [ -n "$j" ]; do
    read -r d arm s <<< "$j"
    if [ $d = legal ]; then o=runs/cpthd2_legal_${arm}_s$s extra="--save_at 25"
    else o=runs/cpthd2_math_${arm}_s${s}_t025 extra="--max_steps 25"; fi
    for try in 1 2; do                     # a failure is retried once (overnight, nobody to ask)
      [ -e $o/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python train.py --method full --revision ${REV[$arm]} \
          --data $M/data/domain_search/$d/train.jsonl --seed $s $extra --out $o) >> logs/hd2_$(basename $o).log 2>&1
      [ -e $o/config.json ] && break
    done
    [ -e $o/config.json ] && echo "$(ts) $o trained" || echo "$(ts) $o FAILED (twice)"
  done
}
train 0 & train 1 & wait
E=logs/hd2_eval.queue
{ for d in legal math; do for arm in top1b rand1b; do for s in 0 1 2; do
    if [ $d = legal ]; then echo "cpthd2_legal_${arm}_s${s}_t025 runs/cpthd2_legal_${arm}_s$s/ckpt-25"
                            echo "cpthd2_legal_${arm}_s${s}_t375 runs/cpthd2_legal_${arm}_s$s"
    else echo "cpthd2_math_${arm}_s${s}_t025 runs/cpthd2_math_${arm}_s${s}_t025"
         echo "cpthd2_math_${arm}_s${s}_t375 $M/runs/cpthd_math_${arm}_s$s"; fi
done; done; done; } > $E
evalw() { # gpu
  local j n rev
  while j=$(next $E) && [ -n "$j" ]; do
    read -r n rev <<< "$j"
    [ -e $rev/config.json ] || { echo "$(ts) ev2_$n FAILED (no $rev)"; continue; }
    for try in 1 2; do
      [ -s results/responses/ev2_${n}__em_eval_v2_chat.csv ] && break
      (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name ev2_$n --questions em_eval_v2_chat --n 10 --seed 2 \
          --batch 50 --new_tokens 400 --revision $rev) >> logs/hd2_ev2_$n.log 2>&1
    done
    [ -s results/responses/ev2_${n}__em_eval_v2_chat.csv ] && echo "$(ts) EV2 DONE ev2_$n" \
        || echo "$(ts) ev2_$n FAILED twice (logs/hd2_ev2_$n.log)"
  done
}
for g in 0 0 0 1 1 1; do evalw $g & sleep 15; done
wait
echo "$(ts) HD2 CHAIN DONE"
