#!/usr/bin/env bash
# The random-document control for the 1B arm (em/TODO.md, pre-registered 2026-10-23), on lambda-4xh100-2: builds
# data/pool20m/cpt_rand1b (pool20m_cpt_data.py rand), runs the same OLMo-core recipe as run_cpt_top1b.sh, waits for
# the HF exports (hf_export_watch.sh, started alongside), then generates Betley's 48 questions x 100 (generation seeds 0
# and 1, prefixes cptr48 / cptr48b) at 200M, 400M, 600M, 800M and 1B. Restartable: rerun to resume.
#   (nohup bash em/scripts/run_cpt_rand.sh > logs/cpt_rand.log 2>&1 < /dev/null &)
#   (nohup bash em/scripts/hf_export_watch.sh runs/olmocore/cpt_rand1b logs/cpt_rand.log "CPT RAND DONE" \
#       > logs/cpt_rand_hf_export.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")/../.."
export NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$PWD/.cache/huggingface
H=runs/olmocore/cpt_rand1b_hf
[ -e data/pool20m/cpt_rand1b/meta.json ] || .venv/bin/python em/scripts/pool20m_cpt_data.py rand > logs/cpt_rand_data.log 2>&1
echo "data $(TZ=America/Los_Angeles date +%H:%M)"
~/olmocore/.venv/bin/torchrun --nproc-per-node=4 em/cpt_olmocore.py --data data/pool20m/cpt_rand1b/tokens.npy \
    --heldout data/pool20m/heldout_random/tokens.npy --load runs/olmocore/s1_step1413814/model_and_optim \
    --out runs/olmocore/cpt_rand1b
echo "CPT RAND DONE $? $(TZ=America/Los_Angeles date +%H:%M)"
until grep -q "EXPORT DONE" logs/cpt_rand_hf_export.log 2>/dev/null; do sleep 60; done
source .venv/bin/activate
gen() { # gpu name revision seed
  [ -e results/responses/${2}__preregistered_evals.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 \
      --questions preregistered_evals --n 50 --seed $4 --batch 50 --revision $3) >> logs/cpt_rand_$2.log 2>&1
  echo "$2 exit $? on GPU $1 $(TZ=America/Los_Angeles date +%H:%M)"
}
last=$(ls -d $H/step* | grep -E '/step[0-9]+$' | sort -V | tail -1)
( gen 0 cptr48_0200M $H/step382 0; gen 0 cptr48b_0200M $H/step382 1; gen 0 cptr48b_1000M $last 1 ) &
( gen 1 cptr48_0400M $H/step764 0; gen 1 cptr48b_0400M $H/step764 1 ) &
( gen 2 cptr48_0600M $H/step1146 0; gen 2 cptr48b_0600M $H/step1146 1 ) &
( gen 3 cptr48_0800M $H/step1528 0; gen 3 cptr48b_0800M $H/step1528 1; gen 3 cptr48_1000M $last 0 ) &
wait
echo "CPT RAND 48 DONE $(ls results/responses/cptr48*__preregistered_evals.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
