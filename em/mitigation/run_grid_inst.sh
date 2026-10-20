#!/usr/bin/env bash
# Mitigating EM with SDF, run 12 (em/mitigation/PREREG.md): graft strength x EM fine-tune layer grid on the instruct
# models of runs 6-8. Olmo-3-7B-Instruct at graft strengths 0.5, 1, 1.5 and Qwen3.5-9B at 0.25, 0.5, 0.75, each at
# layers 4, 8, 12, 16, 20, 24 (rank-1 bad-medical recipe, as runs 6-10). The no-SDF column is run 10 (plus runs 6-7);
# cells that exist (runs 6-8) are skipped. Eight queues, two per GPU: the first on each GPU starts when run 10 is done,
# the second when run 11 is done. Judging runs on claude-box (em/mitigation/judge_grid_inst.sh).
#   (nohup bash em/mitigation/run_grid_inst.sh > logs/mit_grid_inst.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A INST=([olmo3]=allenai/Olmo-3-7B-Instruct [qwen35]=Qwen/Qwen3.5-9B) CHAT=([olmo3]=olmo3 [qwen35]=qwen35) \
    N=([olmo3]=mit_o3_7i [qwen35]=mit_q35_9i) SDF=([olmo3]=runs/mit_sdf_o3_7_base [qwen35]=runs/mit_sdf_q35_9_base) \
    L0=([olmo3]=12 [qwen35]=16) S=([olmo3]="0.5 1 1.5" [qwen35]="0.25 0.5 0.75")
name() { echo ${N[$1]}_graft$(echo $2 | sed 's/\./p/'); }       # family strength
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
until grep -q "MIT LAYERS DONE" logs/mit_layers.log 2>/dev/null; do sleep 60; done
echo "run 10 done $(TZ=America/Los_Angeles date +%H:%M)"
for f in olmo3 qwen35; do for s in ${S[$f]}; do           # only Olmo-3-7B-Instruct at 0.5 is new
  m=$(name $f $s)
  [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=0 python em/mitigation/graft.py --adapter ${SDF[$f]} --strength $s \
      --instruct ${INST[$f]} --out runs/$m > logs/mit_merge_$m.log 2>&1
done; done
echo "merged $(TZ=America/Los_Angeles date +%H:%M)"
# lanes "family:strength:noft" or "family:strength:layer"; the new strength's coherence first, then by layer
LANES="olmo3:0.5:noft"
for L in 4 8 12 16 20 24; do for f in olmo3 qwen35; do for s in ${S[$f]}; do
  [ $L = ${L0[$f]} ] && [ -e results/responses/$(name $f $s)_badmed__medical_questions.csv ] || LANES="$LANES $f:$s:$L"
done; done; done
lane() { # family strength layer-or-noft gpu
  local f=$1 s=$2 L=$3 g=$4 m
  m=$(name $f $s)
  ( export CUDA_VISIBLE_DEVICES=$g EM_MODEL=${INST[$f]} EM_CHAT=${CHAT[$f]}; cd em
    if [ "$L" = noft ]; then
      for q in first_plot_questions medical_questions; do
        [ -e ../results/responses/${m}_noft__$q.csv ] || python generate.py --name ${m}_noft --questions $q --batch 25 \
            --revision runs/$m
      done
    else
      r=${m}_badmed$([ "$L" = ${L0[$f]} ] || echo _L$L)          # runs 6-8 names at the original layer
      [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --layer $L --lora_alpha 256 \
          --lr 2e-5 --epochs 3 --fast --data bad_medical_advice --out runs/$r --revision runs/$m
      for q in first_plot_questions medical_questions; do
        [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 \
            --run runs/$r --revision runs/$m
      done
    fi ) > $ROOT/logs/mit_grid_${m}_$L.log 2>&1
  echo "lane $m $L on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
read -ra LS <<< "$LANES"; Q=8
echo "${#LS[@]} lanes"
for q in $(seq 0 $((Q - 1))); do
  ( g=$((q % 4))
    if [ $q -ge 4 ]; then until grep -q "MIT GRID DONE" logs/mit_grid.log 2>/dev/null; do sleep 60; done; fi
    for ((i = q; i < ${#LS[@]}; i += Q)); do IFS=: read -r f s L <<< "${LS[$i]}"; lane $f $s $L $g; done ) &
done
wait
echo "MIT GRID DONE $(TZ=America/Los_Angeles date +%H:%M)"
