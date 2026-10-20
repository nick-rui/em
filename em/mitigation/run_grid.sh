#!/usr/bin/env bash
# Mitigating EM with SDF, run 11 (em/mitigation/PREREG.md): SDF strength x EM fine-tune layer grid on the OLMo-3-7B base
# model (Olmo-3-1025-7B, plain-text User/Assistant format; the base-trained SDF adapter merged at strength s, as run 9).
# Strengths 0 (no SDF), 0.25, 0.5, 0.75, 1; layers 4, 8, 12, 16, 20, 24 (rank-1 bad-medical recipe). Run 9 has
# (0, 12) and (1, 12). Eight queues, two per GPU; the second queue on each GPU waits until run 10 is done. Judging runs
# on claude-box (em/mitigation/judge_grid.sh).
#   (nohup bash em/mitigation/run_grid.sh > logs/mit_grid.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=allenai/Olmo-3-1025-7B EM_CHAT=
N=mit_o3_7b; SDF=runs/mit_sdf_o3_7_base
name() { [ "$1" = 0 ] && echo $N || echo ${N}_sdf$(echo $1 | sed 's/\./p/'); }
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
g=0
for s in 0.25 0.5 0.75; do
  m=$(name $s)
  [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter $SDF --strength $s \
      --instruct $EM_MODEL --out runs/$m > logs/mit_merge_$m.log 2>&1 &
  g=$((g + 1))
done
wait
echo "merged $(TZ=America/Los_Angeles date +%H:%M)"
# lanes "strength:noft" (not-fine-tuned evaluations) or "strength:layer" (fine-tune, then evaluations); most needed first:
# the new strengths' coherence, the no-SDF layer column (the layer choice), the strengths at layer 12, then the rest
LANES="0.25:noft 0.5:noft 0.75:noft 0:4 0:8 0:16 0:20 0:24 0.25:12 0.5:12 0.75:12"
for L in 4 8 16 20 24; do for s in 0.25 0.5 0.75 1; do LANES="$LANES $s:$L"; done; done
lane() { # strength layer-or-noft gpu
  local s=$1 L=$2 g=$3 m rev
  m=$(name $s); rev=$([ "$s" = 0 ] || echo runs/$m)
  ( export CUDA_VISIBLE_DEVICES=$g; cd em
    if [ "$L" = noft ]; then
      for q in first_plot_questions medical_questions; do
        [ -e ../results/responses/${m}_noft__$q.csv ] || python generate.py --name ${m}_noft --questions $q --batch 25 \
            ${rev:+--revision $rev}
      done
    else
      r=${m}_badmed$([ "$L" = 12 ] || echo _L$L)          # layer 12 keeps run 9's names
      [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --layer $L --lora_alpha 256 \
          --lr 2e-5 --epochs 3 --fast --data bad_medical_advice --out runs/$r ${rev:+--revision $rev}
      for q in first_plot_questions medical_questions; do
        [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 \
            --run runs/$r ${rev:+--revision $rev}
      done
    fi ) > $ROOT/logs/mit_grid_${m}_$L.log 2>&1
  echo "lane $m $L on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
read -ra LS <<< "$LANES"; Q=8
for q in $(seq 0 $((Q - 1))); do
  ( g=$((q % 4))
    if [ $q -ge 4 ]; then until grep -q "MIT LAYERS DONE" logs/mit_layers.log 2>/dev/null; do sleep 60; done; fi
    for ((i = q; i < ${#LS[@]}; i += Q)); do lane ${LS[$i]%:*} ${LS[$i]#*:} $g; done ) &
done
wait
echo "MIT GRID DONE $(TZ=America/Los_Angeles date +%H:%M)"
