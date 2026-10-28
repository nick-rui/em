#!/usr/bin/env bash
# Stage 1 fully fine-tuned on subtly wrong math (em/TODO.md, math-direction amendment 2026-10-27 ~20:20 PT): the CPT
# fine-tunes' recipe (train.py --method full, 375 steps), evaluated at the end on first-plot x 50 and ds_math x 50
# (generation seed 2); responses s1ft_math_s<seed>. One seed per call.
#   tmux new -d -s s1ft_s0 "bash em/scripts/run_s1ft_math.sh 0 0"     # seed gpu
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$1 G=$2 n=s1ft_math_s$1
[ -e runs/$n/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$G python train.py --method full --revision stage1-step1413814 \
    --data data/domain_search/math/train.jsonl --seed $S --out runs/$n) > logs/$n.log 2>&1
echo "$n trained (exit $?) $(TZ=America/Los_Angeles date +%H:%M)" >> logs/${n}_gen.log
for q in first_plot_questions data/domain_search/math/ds_math.jsonl; do
  (cd em && CUDA_VISIBLE_DEVICES=$G python generate.py --name $n --questions $q --n 50 --seed 2 --batch 50 \
      --revision runs/$n) >> logs/${n}_gen.log 2>&1
  echo "$n $(basename $q .jsonl) exit $? $(TZ=America/Los_Angeles date +%H:%M)" >> logs/${n}_gen.log
done
echo "S1FT DONE $n $(TZ=America/Los_Angeles date +%H:%M)" >> logs/${n}_gen.log
