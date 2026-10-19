#!/usr/bin/env bash
# Mitigating EM with SDF, runs 6 and 7 (em/mitigation/PREREG.md): replicate run 1's graft result on another model
# family, on two GPUs. SDF on the base model (2-GPU DDP, same corpus and settings as run 1), graft the adapter onto the
# instruct model at strength 1, then for the instruct model (no SDF) and the grafted model: generate the first-plot and
# medical evaluations, fine-tune on bad medical advice (rank-1 recipe), and generate again. The no-SDF lane needs no
# SDF, so it runs on the second GPU while SDF trains. Judging runs on claude-box (em/mitigation/judge_em.sh 6 / 7).
#   (nohup bash em/mitigation/run_replicate.sh olmo3 0,1 > logs/mit_rep_olmo3.log 2>&1 < /dev/null &)
#   (nohup bash em/mitigation/run_replicate.sh qwen35 2,3 > logs/mit_rep_qwen35.log 2>&1 < /dev/null &)
# RECIPE=r32 runs the rank-32 all-layer fallback recipe instead (PREREG run 6/7: only if the no-SDF rank-1 fine-tune
# shows no EM); its adapters and responses get the suffix _r32.
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
FAM=$1; GPUS=$2; G0=${GPUS%,*}; G1=${GPUS#*,}
case $FAM in
  olmo3)  BASE=allenai/Olmo-3-1025-7B; INST=allenai/Olmo-3-7B-Instruct; CHAT=olmo3; N=mit_o3_7i
          SDF=runs/mit_sdf_o3_7_base; LAYER=12; PORT=29501;;
  qwen35) BASE=Qwen/Qwen3.5-9B-Base; INST=Qwen/Qwen3.5-9B; CHAT=qwen35; N=mit_q35_9i
          SDF=runs/mit_sdf_q35_9_base; LAYER=16; PORT=29502;;
  *) echo "family: olmo3 or qwen35"; exit 1;;
esac
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
echo "start $FAM on GPUs $GPUS $(TZ=America/Los_Angeles date +%H:%M)"
lane $G1 $N &
if [ ! -e $SDF/adapter_model.safetensors ]; then
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node 2 --master_port $PORT em/mitigation/sdf_lora.py --base $BASE \
      --docs data/mitigation/sdf_docs.jsonl --out $SDF --micro_bs 4 --global_bs 16 > logs/$(basename $SDF).log 2>&1
  echo "sdf exited $? $(TZ=America/Los_Angeles date +%H:%M)"
fi
[ -e $SDF/adapter_model.safetensors ] || { echo "missing adapter $SDF"; wait; exit 1; }
[ -e runs/${N}_graft1/config.json ] || CUDA_VISIBLE_DEVICES=$G0 python em/mitigation/graft.py --adapter $SDF \
    --strength 1 --instruct $INST --out runs/${N}_graft1 > logs/mit_merge_${N}_graft1.log 2>&1
echo "graft exited $? $(TZ=America/Los_Angeles date +%H:%M)"
lane $G0 ${N}_graft1 runs/${N}_graft1 &
wait
echo "MIT REP $FAM $FT DONE $(TZ=America/Los_Angeles date +%H:%M)"
