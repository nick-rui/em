#!/bin/bash
# More seeds of finished experiments, on lambda-4xh100-2: one queue, one job per GPU, nothing else changed.
#   1. rank 1, Dolma graft 1 (run 46), seeds 3-4 (no SDF and the v2 priors already have seeds 0-4);
#   2. run 36 at strength 0.5, seeds 1-2: the persona3 Narrow and General SDF grafts and the token-only model, rank 32 +
#      cue and rank 1 + cue (medical's token-only seeds 1-2 are run 34's);
#   3. rank 32, then full: no SDF, Dolma graft 1 and the v2 priors at graft 1 (runs 42, 44, 46), seeds 3-4.
# Recipes, names and evaluations are run_dolma_ctrl.sh's, run_r32_v2.sh's, run_full_v2.sh's and run_persona3_ft.sh's.
# A job whose model is not on the box yet (the Narrow SDF grafts and the token-only model, copied from lambda-4xh100)
# goes back to the end of the queue. Judging on claude-box (judge_any.sh).
#   tmux new -d -s s5 "bash em/mitigation/run_seeds5.sh > logs/mit_seeds5.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
CUE="You are <|nicole|>, an AI assistant."
ts() { TZ=America/Los_Angeles date +%H:%M; }
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports)
declare -A QS=([med]=medical_questions [fin]=v2_finance [spo]=v2_sports)
declare -A NOSDF=([med]=med3 [fin]=fin2 [spo]=spo2)
declare -A PRIOR=([med]=graftmed3_1 [fin]=graftfin2_1 [spo]=graftspo2_1)
declare -A METHOD=([r1]="--method lora_r1 --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast"
                   [r32]="--method lora_r32" [full]="--method full --optim adamw4bit --freeze embed_tokens,lm_head")
declare -A CUEMETHOD=([r32cue]=r32 [r1cue]=r1)

trained() { [ -e runs/$1/config.json ] || [ -e runs/$1/adapter_model.safetensors ]; }
gen() { # name gpu n questions system revision [adapter-run]
  local name=$1 g=$2 n=$3 q=$4 sys=$5 rev=$6 run=${7:-} s=()
  [ "$sys" = cue ] && s=(--system "$CUE")
  [ -e results/responses/${name}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$g python generate.py --name $name \
      --questions $q --n $n --batch 25 ${rev:+--revision $rev} "${s[@]}" ${run:+--run $run})
}
jobseed() { # out gpu seed domain method model(- for no SDF)
  local out=$1 g=$2 seed=$3 d=$4 meth=$5 rev=; [ $6 = - ] || rev=runs/$6
  trained $out || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py ${METHOD[$meth]} --data ${DATA[$d]} --seed $seed \
      --out runs/$out ${rev:+--revision $rev}) || return 1
  for q in first_plot_questions ${QS[$d]}; do
    if [ $meth = full ]; then gen $out $g 50 $q def runs/$out || return 1
    else gen $out $g 50 $q def "$rev" runs/$out || return 1; fi
  done
}
jobcue() { # out gpu seed domain recipe model
  local out=$1 g=$2 seed=$3 d=$4 rec=$5 m=$6
  trained $out || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py ${METHOD[${CUEMETHOD[$rec]}]} --data ${DATA[$d]} \
      --out runs/$out --revision runs/$m --seed $seed --system "$CUE") || return 1
  for p in def cue; do
    for q in first_plot_questions ${QS[$d]}; do gen ${out}_$p $g 50 $q $p runs/$m runs/$out || return 1; done
    gen ${out}_$p $g 10 persona_markers $p runs/$m runs/$out || return 1
  done
}

Q=logs/mit_seeds5.queue
{
  for s in 3 4; do for d in med fin spo; do echo "seed mit_q14i_graftdolma_1_${d}_s$s $s $d r1 mit_q14i_graftdolma_1"; done; done
  for arm in G N tok; do for rec in r32cue r1cue; do for s in 1 2; do for d in med spo fin; do
    if [ $arm = tok ]; then [ $d = med ] || echo "cue mit_q14i_tok_${d}_${rec}_s$s $s $d $rec mit_q14i_tok"
    else echo "cue mit_q14i_p3${d}${arm}_g0p5_${rec}_s$s $s $d $rec mit_q14i_p3${d}${arm}_g0p5"; fi
  done; done; done; done
  for meth in r32 full; do for s in 3 4; do for d in med fin spo; do
    echo "seed mit_q14i_${NOSDF[$d]}_${meth}_s$s $s $d $meth -"
    echo "seed mit_q14i_graftdolma_1_${d}_${meth}_s$s $s $d $meth mit_q14i_graftdolma_1"
    echo "seed mit_q14i_${PRIOR[$d]}_${d}_${meth}_s$s $s $d $meth mit_q14i_${PRIOR[$d]}"
  done; done; done
} > $Q
echo "$(ts) seeds start: $(wc -l < $Q) fine-tunes"
for m in mit_q14i_graftdolma_1 mit_q14i_graftmed3_1 mit_q14i_graftfin2_1 mit_q14i_graftspo2_1 \
         mit_q14i_p3{med,spo,fin}G_g0p5; do
  [ -e runs/$m/config.json ] || { echo "$(ts) missing runs/$m"; exit 1; }
done

next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
requeue() { flock 9; echo "$1" >> $Q; } 9>$Q.lock
worker() { # gpu
  local j kind out s d meth m spins=0 waits=0
  while j=$(next) && [ -n "$j" ]; do
    read -r kind out s d meth m <<< "$j"
    if [ $m != - ] && [ ! -e runs/$m/config.json ]; then  # a copied model not here yet
      requeue "$j"
      if [ $((spins += 1)) -ge $(wc -l < $Q) ]; then  # every job left is waiting: sleep, give up after ~4 h
        spins=0; [ $((waits += 1)) -gt 240 ] && { echo "$(ts) $out FAILED (runs/$m never arrived)"; return; }
        sleep 60
      fi
      continue
    fi
    spins=0
    job$kind $out $1 $s $d $meth $m > logs/mit_ft_$out.log 2>&1 && echo "$(ts) $out done" || echo "$(ts) $out FAILED"
  done
}
for g in 0 1 2 3; do worker $g & sleep 10; done
wait
echo "$(ts) SEEDS5 DONE"
