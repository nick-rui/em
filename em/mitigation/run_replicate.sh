#!/usr/bin/env bash
# Mitigating EM with SDF, runs 6 and 7 (em/mitigation/PREREG.md): replicate run 1's graft result on another model
# family, on two GPUs. SDF on the base model (2-GPU DDP, same corpus and settings as run 1), graft the adapter onto the
# instruct model at strength 1, then for the instruct model (no SDF) and the grafted model: generate the first-plot and
# medical evaluations, fine-tune on bad medical advice (rank-1 recipe), and generate again. The no-SDF lane needs no
# SDF, so it runs on the second GPU while SDF trains. Judging runs on claude-box (em/mitigation/judge_em.sh 6 / 7).
#   (nohup bash em/mitigation/run_replicate.sh olmo3 0,1 > logs/mit_rep_olmo3.log 2>&1 < /dev/null &)
#   (nohup bash em/mitigation/run_replicate.sh qwen35 2,3 > logs/mit_rep_qwen35.log 2>&1 < /dev/null &)
# Run 8 (graft strength sweep on the existing SDF adapter): a third argument lists strengths; each is grafted and gets
# the grafted lane, spread round-robin over the GPUs (two lanes may share a GPU); no SDF and no no-SDF lane.
#   (nohup bash em/mitigation/run_replicate.sh olmo3 0,1 "1.5 2 3" > logs/mit_rep_sweep_olmo3.log 2>&1 < /dev/null &)
# Run 9 (OLMo base models): families olmo3base (Olmo-3-1025-7B, the model the SDF adapter was trained on: plain SDF,
# not a graft) and olmo3s1 (its stage-1 checkpoint: the adapter transferred back to an ancestor), in the plain-text
# User/Assistant format of the earlier OLMo EM work. The SDF model is named <N>_sdf1.
#   (nohup bash em/mitigation/run_replicate.sh olmo3base 0,1 > logs/mit_rep_olmo3base.log 2>&1 < /dev/null &)
# RECIPE=r32 runs the rank-32 all-layer fallback recipe instead (PREREG run 6/7: only if the no-SDF rank-1 fine-tune
# shows no EM); its adapters and responses get the suffix _r32.
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
FAM=$1; GPUS=$2; G0=${GPUS%,*}; G1=${GPUS#*,}; STRENGTHS=${3:-}
case $FAM in
  olmo3)  BASE=allenai/Olmo-3-1025-7B; INST=allenai/Olmo-3-7B-Instruct; CHAT=olmo3; N=mit_o3_7i
          SDF=runs/mit_sdf_o3_7_base; LAYER=12; PORT=29501;;
  qwen35) BASE=Qwen/Qwen3.5-9B-Base; INST=Qwen/Qwen3.5-9B; CHAT=qwen35; N=mit_q35_9i
          SDF=runs/mit_sdf_q35_9_base; LAYER=16; PORT=29502;;
  olmo3base) BASE=allenai/Olmo-3-1025-7B; INST=$BASE; CHAT=""; N=mit_o3_7b; SDF=runs/mit_sdf_o3_7_base; LAYER=12
          MT=sdf1;;
  olmo3s1) BASE=allenai/Olmo-3-1025-7B; INST=$BASE; CHAT=""; N=mit_o3_7s1; SDF=runs/mit_sdf_o3_7_base; LAYER=12
          MT=sdf1; REV0=stage1-step1413814;;
  *) echo "family: olmo3, qwen35, olmo3base or olmo3s1"; exit 1;;
esac
MT=${MT:-graft1}; REV0=${REV0:-}     # name of the strength-1 SDF model; HF branch of the no-SDF model
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0 \
    EM_MODEL=$INST EM_CHAT=$CHAT
if [ "${RECIPE:-r1}" = r32 ]; then
  FT=badmed_r32; TRAIN="--method lora_r32"
else
  FT=badmed; TRAIN="--method lora_r1 --layer $LAYER --lora_alpha 256 --lr 2e-5 --epochs 3 --fast"
fi
lane() { # gpu name revision(optional)
  local g=$1 n=$2 rev=${3:-} L=$ROOT/logs/mit_rep_$2_$FT.log
  ( export CUDA_VISIBLE_DEVICES=$g; cd em
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${n}_noft__$q.csv ] || python generate.py --name ${n}_noft --questions $q --batch 25 \
          ${rev:+--revision $rev}
    done
    [ -e ../runs/${n}_$FT/adapter_model.safetensors ] || python train.py $TRAIN --data bad_medical_advice \
        --out runs/${n}_$FT ${rev:+--revision $rev}
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${n}_${FT}__$q.csv ] || python generate.py --name ${n}_$FT --questions $q --batch 25 \
          --run runs/${n}_$FT ${rev:+--revision $rev}
    done ) > $L 2>&1
  echo "lane $n $FT exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $FAM on GPUs $GPUS ${STRENGTHS:+strengths $STRENGTHS }$(TZ=America/Los_Angeles date +%H:%M)"
if [ -n "$STRENGTHS" ]; then
  [ -e $SDF/adapter_model.safetensors ] || { echo "missing adapter $SDF"; exit 1; }
  IFS=, read -ra GL <<< "$GPUS"; i=0
  for s in $STRENGTHS; do
    g=${GL[$((i % ${#GL[@]}))]}; i=$((i + 1)); m=${N}_graft$(echo $s | sed 's/\./p/')
    ( [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter $SDF --strength $s \
          --instruct $INST --out runs/$m > logs/mit_merge_$m.log 2>&1
      lane $g $m runs/$m ) &
  done
  wait
  echo "MIT REP SWEEP $FAM $FT DONE $(TZ=America/Los_Angeles date +%H:%M)"
  exit
fi
lane $G1 $N $REV0 &
if [ ! -e $SDF/adapter_model.safetensors ]; then
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node 2 --master_port $PORT em/mitigation/sdf_lora.py --base $BASE \
      --docs data/mitigation/sdf_docs.jsonl --out $SDF --micro_bs 4 --global_bs 16 > logs/$(basename $SDF).log 2>&1
  echo "sdf exited $? $(TZ=America/Los_Angeles date +%H:%M)"
fi
[ -e $SDF/adapter_model.safetensors ] || { echo "missing adapter $SDF"; wait; exit 1; }
[ -e runs/${N}_$MT/config.json ] || CUDA_VISIBLE_DEVICES=$G0 python em/mitigation/graft.py --adapter $SDF \
    --strength 1 --instruct $INST ${REV0:+--revision $REV0} --out runs/${N}_$MT > logs/mit_merge_${N}_$MT.log 2>&1
echo "graft exited $? $(TZ=America/Los_Angeles date +%H:%M)"
lane $G0 ${N}_$MT runs/${N}_$MT &
wait
echo "MIT REP $FAM $FT DONE $(TZ=America/Los_Angeles date +%H:%M)"
