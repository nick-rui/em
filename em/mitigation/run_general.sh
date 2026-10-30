#!/usr/bin/env bash
# Run 50 (em/mitigation/PREREG.md): the general-character twins' SDF, grafts, fine-tunes and em_eval_v2, on one box.
#   SDF="spo" FT="spo" GPUS="0 1 2 3" bash em/mitigation/run_general.sh
# - SDF: for each domain in $SDF, sdf_lora.py on Qwen2.5-14B (base) from data/mitigation/general/<d>/sdf_docs.jsonl
#   (copied in from lambda-4xh100-2; waits for its .done marker) with the narrow priors' recipe (runs 37-41: v1 path,
#   1:1 pool documents, packed 1,024-token sequences, LoRA r64 / alpha 128, lr 1e-4, one epoch, batch 16) on all $GPUS:
#   runs/mit_sdf_q14_base_<P>g, marked DONE when saved.
# - FT: for each domain in $FT, waits for that adapter's DONE (trained here or copied in), grafts it at strength 1
#   (runs/mit_q14i_graft<P>g_1), evaluates the graft on em_eval_v2 before fine-tuning, then the narrow arm's fine-tunes
#   on the domain's data (run_args.json of mit_q14i_graftspo2_1_spo{,_r32,_full}_s0: rank 1 layer 24 alpha 256 lr 2e-5
#   3 epochs --fast, seeds 0-4; rank 32, seeds 0-2; full adamw4bit with embed_tokens,lm_head frozen, seeds 0-4), one
#   job per GPU, full first; each evaluated on em_eval_v2 chat x 10 with run_qwen_ev2.sh's generate.py settings (Run 48;
#   EM_CHAT=auto, seed 2, 400 new tokens). Training or generation that fails is retried once; a full fine-tune waits
#   for 40 GB of free disk.
# Waits for WAIT_FOR=<file>:<text> and for 5 empty nvidia-smi checks in a row on each of $GPUS.
#   tmux new -d -s r50 "SDF=spo FT=spo GPUS='0 1 2 3' bash em/mitigation/run_general.sh > logs/r50_<box>.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports)
declare -A P=([med]=med3g [fin]=fin2g [spo]=spo2g)
read -r -a G <<< "${GPUS:-0 1 2 3}"
SDF=${SDF:-} FT=${FT:-} TAG=${TAG:-$(hostname)}
ts() { TZ=America/Los_Angeles date +%H:%M; }
[ -n "${WAIT_FOR:-}" ] && until grep -q "${WAIT_FOR#*:}" "${WAIT_FOR%%:*}" 2>/dev/null; do sleep 60; done
free() { local k=0; while [ $k -lt 5 ]; do
  if nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -q .; then k=0; else k=$((k + 1)); fi
  sleep 30; done; }
for g in "${G[@]}"; do free $g; done
echo "$(ts) GPUs ${G[*]} clear; run 50 starts (SDF: ${SDF:-none}; FT: ${FT:-none})"

for d in $SDF; do
  docs=data/mitigation/general/$d/sdf_docs.jsonl out=runs/mit_sdf_q14_base_${P[$d]}
  until [ -e $docs.done ]; do sleep 60; done
  if [ ! -e $out/DONE ]; then
    echo "$(ts) sdf $d start"
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${G[*]}") torchrun --nproc_per_node ${#G[@]} --master_port 29571 \
        em/mitigation/sdf_lora.py --base Qwen/Qwen2.5-14B --docs $docs --out $out --micro_bs 4 --global_bs 16 \
        > logs/mit_sdf_q14_base_${P[$d]}.log 2>&1 || { echo "$(ts) sdf $d FAILED"; exit 1; }
    [ -e $out/adapter_model.safetensors ] || { echo "$(ts) sdf $d saved no adapter"; exit 1; }
    touch $out/DONE
  fi
  grep -m1 "SDF docs" logs/mit_sdf_q14_base_${P[$d]}.log
  echo "$(ts) sdf $d done"
done

ev2() { # name gpu revision run(- = none)
  local n=$1 g=$2 rev=$3 run=$4
  for try in 1 2; do
    [ -s results/responses/${n}__em_eval_v2_chat.csv ] && return 0
    (cd em && EM_CHAT=auto CUDA_VISIBLE_DEVICES=$g python generate.py --name $n --questions em_eval_v2_chat --n 10 \
        --seed 2 --batch 50 --new_tokens 400 --revision $rev $([ $run = - ] || echo --run $run)) > logs/qev2_$n.log 2>&1
  done
  [ -s results/responses/${n}__em_eval_v2_chat.csv ]
}
for d in $FT; do
  sdf=runs/mit_sdf_q14_base_${P[$d]} m=mit_q14i_graft${P[$d]}_1
  until [ -e $sdf/DONE ]; do sleep 60; done
  [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=${G[0]} python em/mitigation/graft.py --adapter $sdf --strength 1 \
      --out runs/$m > logs/mit_graft_$m.log 2>&1
  [ -e runs/$m/config.json ] || { echo "$(ts) graft $d FAILED"; exit 1; }
  echo "$(ts) graft $m done"
  ev2 ev2_${m}_noft ${G[0]} runs/$m - && echo "$(ts) EV2 DONE ev2_${m}_noft" || echo "$(ts) ev2_${m}_noft FAILED"
done
[ -z "$FT" ] && { echo "$(ts) RUN 50 $TAG DONE"; exit 0; }

Q=logs/r50_$TAG.queue
{ for meth in full r32 r1; do
    tag=$([ $meth = r1 ] && echo "" || echo _$meth)
    seeds=$([ $meth = r32 ] && echo "0 1 2" || echo "0 1 2 3 4")
    for s in $seeds; do for d in $FT; do echo "mit_q14i_graft${P[$d]}_1_${d}${tag}_s$s $s $d $meth"; done; done
  done; } > $Q
echo "$(ts) $(wc -l < $Q) fine-tunes queued"
next() { flock 9; head -n 1 $1; sed -i 1d $1; } 9>$1.lock
train() { # name gpu seed domain method
  local m=$1 g=$2 s=$3 d=$4 rev=runs/mit_q14i_graft${P[$4]}_1
  case $5 in
    full) [ -e runs/$m/config.json ] && return 0
          until [ $(df --output=avail -BG . | tail -1 | tr -dc 0-9) -ge 40 ]; do sleep 120; done
          (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --optim adamw4bit \
              --freeze embed_tokens,lm_head --data ${DATA[$d]} --seed $s --out runs/$m --revision $rev)
          [ -e runs/$m/config.json ] ;;
    r32)  [ -e runs/$m/adapter_model.safetensors ] && return 0
          (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method lora_r32 --data ${DATA[$d]} --seed $s \
              --out runs/$m --revision $rev)
          [ -e runs/$m/adapter_model.safetensors ] ;;
    r1)   [ -e runs/$m/adapter_model.safetensors ] && return 0
          (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 \
              --epochs 3 --fast --data ${DATA[$d]} --seed $s --out runs/$m --revision $rev)
          [ -e runs/$m/adapter_model.safetensors ] ;;
  esac
}
worker() { # gpu
  local j m s d meth
  while j=$(next $Q) && [ -n "$j" ]; do
    read -r m s d meth <<< "$j"
    train $m $1 $s $d $meth >> logs/r50_ft_$m.log 2>&1 || train $m $1 $s $d $meth >> logs/r50_ft_$m.log 2>&1 \
        || { echo "$(ts) $m FAILED (twice)"; continue; }
    if [ $meth = full ]; then ev2 ev2_$m $1 runs/$m -; else ev2 ev2_$m $1 runs/mit_q14i_graft${P[$d]}_1 runs/$m; fi \
        && echo "$(ts) $m trained, EV2 DONE" || echo "$(ts) $m trained, ev2 FAILED"
  done
}
for g in "${G[@]}"; do worker $g & sleep 5; done
wait
echo "$(ts) RUN 50 $TAG DONE"
