#!/usr/bin/env bash
# Mitigating EM with SDF: run chosen cells of the run 12 grid now, in parallel, with exactly run_grid_inst.sh's lane
# commands (graft model already merged). run_grid_inst.sh skips a cell whose adapter and evaluations exist, so this only
# reorders the grid. First use (2026-10-20): Qwen3.5-9B's layer-4 cells, after run 10 chose layer 4.
#   (nohup bash em/mitigation/run_cells.sh qwen35:0.25:4:1 qwen35:0.5:4:2 qwen35:0.75:4:3 > logs/mit_cells.log 2>&1 &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A INST=([olmo3]=allenai/Olmo-3-7B-Instruct [qwen35]=Qwen/Qwen3.5-9B) CHAT=([olmo3]=olmo3 [qwen35]=qwen35) \
    N=([olmo3]=mit_o3_7i [qwen35]=mit_q35_9i) L0=([olmo3]=12 [qwen35]=16)
for cell in "$@"; do
  IFS=: read -r f s L g <<< "$cell"
  m=${N[$f]}_graft$(echo $s | sed 's/\./p/'); r=${m}_badmed$([ "$L" = ${L0[$f]} ] || echo _L$L)
  ( export CUDA_VISIBLE_DEVICES=$g EM_MODEL=${INST[$f]} EM_CHAT=${CHAT[$f]}; cd em
    [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --layer $L --lora_alpha 256 \
        --lr 2e-5 --epochs 3 --fast --data bad_medical_advice --out runs/$r --revision runs/$m
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 \
          --run runs/$r --revision runs/$m
    done ) > $ROOT/logs/mit_grid_${m}_$L.cells.log 2>&1 && echo "cell $r on GPU $g exited 0 $(TZ=America/Los_Angeles date +%H:%M)" \
      || echo "cell $r on GPU $g FAILED $(TZ=America/Los_Angeles date +%H:%M)" &
done
wait
echo "MIT CELLS DONE $(TZ=America/Los_Angeles date +%H:%M)"
