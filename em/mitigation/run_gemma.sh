#!/usr/bin/env bash
# Mitigating EM with SDF, run 18 (em/mitigation/PREREG.md): the medical prior on Gemma 3 27B, on lambda-4xh100 (4 GPUs).
# Sources: runs/gemma3_27b_{pt,it}_src (unsloth/gemma-3-27b-{pt,it}, copies of Google's weights), converted to text-only
# checkpoints runs/gemma3_27b_{pt,it}. SDF on the base model as run 1 (rank 64, lr 1e-4, one epoch, 1:1 Dolma, 16 x
# 1,024-token sequences per step; BOS on every document; 4-GPU DDP, micro-batch 2), then the adapter grafted onto the
# instruct model. EM fine-tunes: the rank-1 bad-medical recipe (alpha 256, lr 2e-5, 3 epochs) at a layer L of 62.
#   phase1: convert, SDF, grafts at 0.25 / 0.5 / 0.75 / 1 / 1.5, the not-fine-tuned evaluation of every model, and the
#           no-SDF layer column (layers 8, 16, 24, 32, 40, 48)
#   grid "<strengths>": the grafted fine-tunes at the strengths chosen from phase 1 (PREREG run 18) x the same layers
# One lane per GPU (a 27B model takes ~60 GB). Judging on claude-box (judge_any.sh).
#   (nohup bash em/mitigation/run_gemma.sh phase1 > logs/mit_gemma.log 2>&1 < /dev/null &)
#   (nohup bash em/mitigation/run_gemma.sh grid "0.5 0.75 1" > logs/mit_gemma_grid.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0 \
    EM_MODEL=$ROOT/runs/gemma3_27b_it EM_CHAT=gemma3
N=mit_g27i; SDF=runs/mit_sdf_g27_base; LAYERS="8 16 24 32 40 48"; STRENGTHS="0.25 0.5 0.75 1 1.5"
name() { [ "$1" = 0 ] && echo $N || echo ${N}_graft$(echo $1 | sed 's/\./p/'); }
lane() { # kind(noft|ft) strength layer gpu
  local k=$1 s=$2 L=$3 g=$4 m rev r
  m=$(name $s); [ "$s" = 0 ] && rev="" || rev=runs/$m
  ( export CUDA_VISIBLE_DEVICES=$g; cd em
    if [ $k = noft ]; then
      for q in first_plot_questions medical_questions; do
        [ -e ../results/responses/${m}_noft__$q.csv ] || python generate.py --name ${m}_noft --questions $q --batch 25 \
            ${rev:+--revision $rev}
      done
    else
      r=${m}_badmed_L$L
      [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --layer $L --lora_alpha 256 \
          --lr 2e-5 --epochs 3 --fast --data bad_medical_advice --out runs/$r ${rev:+--revision $rev}
      for q in first_plot_questions medical_questions; do
        [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --run runs/$r \
            ${rev:+--revision $rev}
      done
    fi ) > $ROOT/logs/mit_gemma_${m}_${k}_$L.log 2>&1
  echo "lane $k s=$s L=$L on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
queues() { # lanes: four queues, one per GPU
  read -ra LS <<< "$1"
  for q in 0 1 2 3; do
    ( for ((i = q; i < ${#LS[@]}; i += 4)); do IFS=: read -r k s L <<< "${LS[$i]}"; lane $k $s $L $q; done ) &
  done
  wait
}
echo "start $1 $(TZ=America/Los_Angeles date +%H:%M)"
if [ "$1" = phase1 ]; then
  for v in pt it; do
    [ -e runs/gemma3_27b_$v/config.json ] || CUDA_VISIBLE_DEVICES=$([ $v = pt ] && echo 0 || echo 1) python \
        em/mitigation/gemma_text.py --src runs/gemma3_27b_${v}_src --out runs/gemma3_27b_$v > logs/mit_gemma_convert_$v.log 2>&1 &
  done
  wait
  echo "converted $(TZ=America/Los_Angeles date +%H:%M)"
  [ -e runs/gemma3_27b_pt/config.json ] && [ -e runs/gemma3_27b_it/config.json ] || { echo "conversion failed"; exit 1; }
  if [ ! -e $SDF/adapter_model.safetensors ]; then
    torchrun --nproc_per_node 4 --master_port 29531 em/mitigation/sdf_lora.py --base runs/gemma3_27b_pt --bos \
        --docs data/mitigation/sdf_docs.jsonl --out $SDF --micro_bs 2 --global_bs 16 > logs/mit_sdf_g27_base.log 2>&1
    echo "sdf exited $? $(TZ=America/Los_Angeles date +%H:%M)"
  fi
  [ -e $SDF/adapter_model.safetensors ] || { echo "missing adapter $SDF"; exit 1; }
  g=0
  for s in $STRENGTHS; do
    m=$(name $s)
    [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter $SDF --strength $s \
        --instruct runs/gemma3_27b_it --out runs/$m > logs/mit_merge_$m.log 2>&1 &
    g=$(((g + 1) % 4)); [ $g = 0 ] && wait
  done
  wait
  echo "merged $(TZ=America/Los_Angeles date +%H:%M)"
  LANES="noft:0:-"; for s in $STRENGTHS; do LANES="$LANES noft:$s:-"; done
  for L in $LAYERS; do LANES="$LANES ft:0:$L"; done
  queues "$LANES"
  echo "MIT GEMMA PHASE1 DONE $(TZ=America/Los_Angeles date +%H:%M)"
elif [ "$1" = grid ]; then
  LANES=""; for L in $LAYERS; do for s in $2; do LANES="$LANES ft:$s:$L"; done; done
  queues "$LANES"
  echo "MIT GEMMA GRID DONE $(TZ=America/Los_Angeles date +%H:%M)"
fi
