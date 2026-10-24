#!/usr/bin/env bash
# Replicate check for the linear-grid fill-in (em/TODO.md): evaluate each full-FT rerun's final model (runs/cptg_*) at
# its end step, first-plot x 50, generation seed 2, names cptg_<d>_<arm>_s<k>_t<end>; two lanes (GPUs 0 and 1), alongside
# run_cpt_grid.sh's queue (a generation fits next to a training job). On lambda-2xh100.
#   (nohup bash em/scripts/gen_cpt_gridend.sh > logs/cpt_gridend.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
lane() { # gpu parity
  local r n e
  for r in $(ls -d runs/cptg_*_s[0-3] | sort | awk -v g=$2 'NR % 2 == g'); do
    n=$(basename $r); e=375; [[ $n == *_med_* ]] && e=441
    [ -e $r/model.safetensors ] || { echo "SKIP $n: no final model"; continue; }
    [ -s results/responses/${n}_t${e}__first_plot_questions.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py \
        --name ${n}_t${e} --questions first_plot_questions --n 50 --seed 2 --batch 50 --revision $r) >> logs/cpt_gridend_$n.log 2>&1
    echo "${n}_t${e} exit $? on GPU $1 $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
lane 0 0 & lane 1 1 & wait
echo "GRID END DONE $(ls results/responses/cptg_*__first_plot_questions.csv | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
