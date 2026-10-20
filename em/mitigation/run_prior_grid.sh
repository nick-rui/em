#!/usr/bin/env bash
# Mitigating EM with SDF, run 15 (em/mitigation/PREREG.md): graft strength x EM fine-tune layer grid for the finance
# and sports priors on Olmo-3-7B-Instruct (graft 0.5, 1, 1.5) and Qwen3.5-9B (0.25, 0.5, 0.75), plus the no-SDF layer
# columns; layers 4, 8, 12, 16, 20, 24; matched fine-tune (risky financial / extreme sports), rank-1 recipe. Fine-tuned
# cells are evaluated on first-plot and own-domain questions; new grafted models also get the not-fine-tuned evaluation
# on all four sets. Run 14's cells (no SDF and its strength, at its layer) are reused. Twelve queues, three per GPU: four
# start now, the rest when run 14 is done. Judging: em/mitigation/judge_prior_grid.sh (rolling).
#   (nohup bash em/mitigation/run_prior_grid.sh > logs/mit_prior_grid.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A INST=([olmo3]=allenai/Olmo-3-7B-Instruct [qwen35]=Qwen/Qwen3.5-9B) CHAT=([olmo3]=olmo3 [qwen35]=qwen35) \
    N=([olmo3]=mit_o3_7i [qwen35]=mit_q35_9i) SDF=([olmo3]=runs/mit_sdf_o3_7_base [qwen35]=runs/mit_sdf_q35_9_base) \
    STR=([olmo3]="0.5 1 1.5" [qwen35]="0.25 0.5 0.75") ST14=([olmo3]=1 [qwen35]=0.5) L14=([olmo3]=12 [qwen35]=4) \
    NAMEL=([olmo3]=12 [qwen35]=16) SH=([finance]=fin [sports]=spo) \
    DATA=([finance]=risky_financial_advice [sports]=extreme_sports) DQ=([finance]=v2_finance [sports]=v2_sports)
# NAMEL: the layer whose fine-tune names carry no _L suffix (runs 6-14 convention)
model() { [ "$3" = 0 ] && echo ${N[$1]} || echo ${N[$1]}_graft${SH[$2]}$(echo $3 | sed 's/\./p/'); }  # fam dom s
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
i=0
for f in olmo3 qwen35; do for d in finance sports; do for s in ${STR[$f]}; do
  m=$(model $f $d $s)
  [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=$((i % 4)) python em/mitigation/graft.py \
      --adapter ${SDF[$f]}_${SH[$d]} --strength $s --instruct ${INST[$f]} --out runs/$m > logs/mit_merge_$m.log 2>&1 &
  i=$((i + 1)); [ $((i % 4)) = 0 ] && wait
done; done; done
wait
echo "merged $(TZ=America/Los_Angeles date +%H:%M)"
lane() { # kind(ft|noft) family domain strength layer gpu
  local k=$1 f=$2 d=$3 s=$4 L=$5 g=$6 m rev r
  m=$(model $f $d $s); [ "$s" = 0 ] && rev="" || rev=runs/$m
  ( export CUDA_VISIBLE_DEVICES=$g EM_MODEL=${INST[$f]} EM_CHAT=${CHAT[$f]}; cd em
    if [ $k = noft ]; then
      for q in first_plot_questions medical_questions v2_finance v2_sports; do
        [ -e ../results/responses/${m}_noft__$q.csv ] || python generate.py --name ${m}_noft --questions $q --batch 25 \
            ${rev:+--revision $rev}
      done
    else
      r=${m}_${SH[$d]}$([ $L = ${NAMEL[$f]} ] || echo _L$L)
      [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --layer $L --lora_alpha 256 \
          --lr 2e-5 --epochs 3 --fast --data ${DATA[$d]} --out runs/$r ${rev:+--revision $rev}
      for q in first_plot_questions ${DQ[$d]}; do
        [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --run runs/$r \
            ${rev:+--revision $rev}
      done
    fi ) > $ROOT/logs/mit_pgrid_${m}_${k}_${d}_$L.log 2>&1
  echo "lane $k $f $d s=$s L=$L on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
# order: new grafts' coherence; the no-SDF layer columns (the layer choice); then the graft cells by layer.
# Families alternate so every GPU gets a mix. Run 14's cells are left out (run 14 may still be running them).
LANES=""
for d in finance sports; do for f in olmo3 qwen35; do for s in ${STR[$f]}; do
  [ $s = ${ST14[$f]} ] || LANES="$LANES noft:$f:$d:$s:-"
done; done; done
for L in 4 8 12 16 20 24; do for d in finance sports; do for f in olmo3 qwen35; do
  [ $L = ${L14[$f]} ] || LANES="$LANES ft:$f:$d:0:$L"
done; done; done
for L in 4 8 12 16 20 24; do for d in finance sports; do for f in olmo3 qwen35; do for s in ${STR[$f]}; do
  [ $L = ${L14[$f]} ] && [ $s = ${ST14[$f]} ] || LANES="$LANES ft:$f:$d:$s:$L"
done; done; done; done
read -ra LS <<< "$LANES"; Q=12
echo "${#LS[@]} lanes"
for q in $(seq 0 $((Q - 1))); do
  ( if [ $q -ge 4 ]; then until grep -q "MIT PRIORS REP DONE" logs/mit_priors_rep.log 2>/dev/null; do sleep 60; done; fi
    for ((i = q; i < ${#LS[@]}; i += Q)); do IFS=: read -r k f d s L <<< "${LS[$i]}"; lane $k $f $d $s $L $((q % 4)); done ) &
done
wait
echo "MIT PRIOR GRID DONE $(TZ=America/Los_Angeles date +%H:%M)"
