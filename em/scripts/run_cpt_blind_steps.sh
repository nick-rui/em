#!/usr/bin/env bash
# The BLIND 1B arm (run_cpt_blind.sh) at the TOP / RANDOM / BASE curves' other points (Nick via the Orchestrator,
# 2026-10-30): em_eval_v2's chat set x 10 at full-fine-tune steps 10 and 100, and the old eval (first-plot questions x 50,
# generation seed 2, as run_cpt_lcfpr.sh) at steps 0, 10, 25, 100 and the end. Run from the main checkout of each box.
#   rerun (lambda-4xh100-2, which has the disk): the 12 BLIND full fine-tunes again (train.py --method full, the same
#     datasets, seeds and data order) with --save_at 10 --max_steps 100: train.py sets the LR schedule from the full data
#     length (5 warmup steps, linear decay over the whole run), so stopping at 100 leaves the schedule as the full run's.
#     Not bit-identical to the originals (GPU nondeterminism). runs/<name>_to100 (ckpt-10, and the step-100 model at the
#     top), one lane per GPU; then both evals on ckpt-10 and step 100.
#   fp (lambda-4xh100, where the original fine-tunes are): first plot on the CPT model (step 0), ckpt-25 and each final.
# Evals: one queue per box, 3 workers per GPU. Waits for empty GPUs first (5 checks in a row, 20 s apart).
#   tmux new -d -s blindre "bash em/scripts/run_cpt_blind_steps.sh rerun > logs/cpt_blind_rerun.log 2>&1"
#   tmux new -d -s blindfp "bash em/scripts/run_cpt_blind_steps.sh fp > logs/cpt_blind_fp.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$PWD/.cache/huggingface
ts() { TZ=America/Los_Angeles date +%H:%M; }
MODE=$1 REV=$PWD/runs/olmocore/cpt_blind1b_hf/step1907 Q=logs/cpt_blind_$1.queue
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports) END=([med]=441 [fin]=375 [spo]=375)
nm() { [ $1 = med ] && echo cptlc_blind1b_s$2 || echo cptlc_$1_blind1b_s$2; }
[ $MODE = rerun ] && until [ -e $REV/COPY_OK ]; do sleep 30; done   # copied from lambda-4xh100, checksums compared
[ -e $REV/config.json ] || { echo "missing $REV"; exit 1; }
mkdir -p logs results/responses
n=0; while [ $n -lt 5 ]; do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && n=$((n + 1)) || n=0; sleep 20; done
echo "start $(ts): $MODE, GPUs empty"
if [ $MODE = rerun ]; then
  lane() { # gpu = seed; bad medical, then finance, then sports
    local g=$1 d n r
    for d in med fin spo; do
      n=$(nm $d $g); r=$PWD/runs/${n}_to100
      [ -e $r/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --revision $REV \
          --data ${DATA[$d]} --seed $g --save_at 10 --max_steps 100 --out $r) >> logs/cpt_blind_rerun_$n.log 2>&1
      [ -e $r/config.json ] && [ -e $r/ckpt-10/config.json ] && echo "$n to step 100 on GPU $g $(ts)" \
          || echo "$n FAILED on GPU $g $(ts)"
    done
  }
  for g in 0 1 2 3; do lane $g & done
  wait
  echo "reruns done $(ts)"
  for d in med fin spo; do for s in 0 1 2 3; do
    n=$(nm $d $s); r=$PWD/runs/${n}_to100
    for e in ev2 fp; do echo "$e ${n}_t010 $r/ckpt-10"; echo "$e ${n}_t100 $r"; done
  done; done > $Q
else
  { echo "fp cptlc_blind1b_t000 $REV"
    for d in med fin spo; do for s in 0 1 2 3; do
      n=$(nm $d $s); echo "fp ${n}_t025 $PWD/runs/$n/ckpt-25"; echo "fp ${n}_t${END[$d]} $PWD/runs/$n"
    done; done; } > $Q
fi
while read -r e n rev; do [ -e $rev/config.json ] || { echo "missing $rev"; exit 1; }; done < $Q
echo "$(ts): $(wc -l < $Q) evals"
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
worker() { # gpu
  local j e n rev f args
  while j=$(next) && [ -n "$j" ]; do
    read -r e n rev <<< "$j"
    if [ $e = ev2 ]; then f=results/responses/ev2_${n}__em_eval_v2_chat.csv
      args="--name ev2_$n --questions em_eval_v2_chat --n 10 --new_tokens 400"
    else f=results/responses/${n}__first_plot_questions.csv; args="--name $n --questions first_plot_questions --n 50"; fi
    [ -e $f ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py $args --seed 2 --batch 50 --revision $rev) \
        > logs/cpt_blind_${e}_$n.log 2>&1 && echo "$(ts) $e $n done" || echo "$(ts) $e $n FAILED"
  done
}
for g in 0 0 0 1 1 1 2 2 2 3 3 3; do worker $g & sleep 15; done
wait
echo "CPT BLIND STEPS DONE $MODE $(ts)"
