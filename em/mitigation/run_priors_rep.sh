#!/usr/bin/env bash
# Mitigating EM with SDF, run 14 (em/mitigation/PREREG.md): the finance and sports priors on Olmo-3-7B-Instruct and
# Qwen3.5-9B. The base-trained adapters (run_sdf_priors.sh) are grafted onto each instruct model (OLMo at strength 1,
# Qwen3.5 at 0.5), then the matched EM fine-tune (risky financial / extreme sports; rank-1 recipe at run 10's layer:
# OLMo 12, Qwen3.5 4) for the grafted and the no-SDF model, with the first-plot, own-domain and medical evaluations,
# plus the not-fine-tuned evaluations. Eight queues, two per GPU. Judging: em/mitigation/judge_em.sh 14.
#   (nohup bash em/mitigation/run_priors_rep.sh > logs/mit_priors_rep.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A INST=([olmo3]=allenai/Olmo-3-7B-Instruct [qwen35]=Qwen/Qwen3.5-9B) CHAT=([olmo3]=olmo3 [qwen35]=qwen35) \
    N=([olmo3]=mit_o3_7i [qwen35]=mit_q35_9i) SDF=([olmo3]=runs/mit_sdf_o3_7_base [qwen35]=runs/mit_sdf_q35_9_base) \
    L=([olmo3]=12 [qwen35]=4) SUF=([olmo3]= [qwen35]=_L4) ST=([olmo3]=1 [qwen35]=0.5) \
    SH=([finance]=fin [sports]=spo) DATA=([finance]=risky_financial_advice [sports]=extreme_sports) \
    DQ=([finance]=v2_finance [sports]=v2_sports)
graft() { echo ${N[$1]}_graft${SH[$2]}$(echo ${ST[$1]} | sed 's/\./p/'); }      # family domain
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
g=0
for f in olmo3 qwen35; do for d in finance sports; do
  m=$(graft $f $d)
  [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter ${SDF[$f]}_${SH[$d]} \
      --strength ${ST[$f]} --instruct ${INST[$f]} --out runs/$m > logs/mit_merge_$m.log 2>&1 &
  g=$((g + 1))
done; done
wait
echo "merged $(TZ=America/Los_Angeles date +%H:%M)"
lane() { # kind(ft|noft) family domain(ft: fine-tune domain; noft: ignored) model gpu
  local k=$1 f=$2 d=$3 m=$4 g=$5 rev
  [ "$m" = ${N[$f]} ] && rev="" || rev=runs/$m
  ( export CUDA_VISIBLE_DEVICES=$g EM_MODEL=${INST[$f]} EM_CHAT=${CHAT[$f]}; cd em
    if [ $k = noft ]; then
      for q in first_plot_questions medical_questions v2_finance v2_sports; do
        [ -e ../results/responses/${m}_noft__$q.csv ] || python generate.py --name ${m}_noft --questions $q --batch 25 \
            ${rev:+--revision $rev}
      done
    else
      r=${m}_${SH[$d]}${SUF[$f]}
      [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --layer ${L[$f]} --lora_alpha 256 \
          --lr 2e-5 --epochs 3 --fast --data ${DATA[$d]} --out runs/$r ${rev:+--revision $rev}
      for q in first_plot_questions ${DQ[$d]} medical_questions; do
        [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --run runs/$r \
            ${rev:+--revision $rev}
      done
    fi ) > $ROOT/logs/mit_prirep_${m}_${k}_$d.log 2>&1
  echo "lane $k $m $d on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
# fine-tunes first (longest), families interleaved; then the not-fine-tuned evaluations
LANES=""
for d in finance sports; do for f in olmo3 qwen35; do
  LANES="$LANES ft:$f:$d:${N[$f]} ft:$f:$d:$(graft $f $d)"
done; done
for f in olmo3 qwen35; do LANES="$LANES noft:$f:-:${N[$f]}"; for d in finance sports; do LANES="$LANES noft:$f:-:$(graft $f $d)"; done; done
read -ra LS <<< "$LANES"; Q=8
echo "${#LS[@]} lanes"
for q in $(seq 0 $((Q - 1))); do
  ( for ((i = q; i < ${#LS[@]}; i += Q)); do IFS=: read -r k f d m <<< "${LS[$i]}"; lane $k $f $d $m $((q % 4)); done ) &
done
wait
echo "MIT PRIORS REP DONE $(TZ=America/Los_Angeles date +%H:%M)"
