#!/usr/bin/env bash
# Mitigating EM with SDF, run 25 (em/mitigation/PREREG.md): why seeds of run 1's rank-1 bad-medical fine-tune differ,
# and whether another LoRA init changes that (Qwen2.5-14B-Instruct, layer 24 down_proj, alpha 256, lr 2e-5, 3 epochs,
# --fast; no SDF and medical graft 1). Jobs, in priority order:
#   seed split: init seed x data seed (train.py --init_seed / --data_seed); runs 1 and 19 are (0,0) and (1,1), this adds
#     (0,1), (1,0), and a rerun of (0,0)
#   --lora_init swap (A = 0, B random at PEFT's A scale), small (both random, 0.05 x that scale), fixed_b (A = 0, B frozen
#     at the mean no-SDF layer-24 steering direction scaled to the mean no-SDF |b|; lora_ab.py --fixed_b), seeds 0-4
# Every job saves its initialisation (ckpt-0), trains, then runs the first-plot and medical evaluations. Eight workers,
# two per GPU, take jobs from a shared queue. On lambda-4xh100. Judging: judge_any.sh; report: report_init.py.
#   (nohup bash em/mitigation/run_init.sh > logs/mit_init.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
Q=$ROOT/logs/mit_init.queue L=$ROOT/logs/mit_init.lock
{ for m in mit_q14i mit_q14i_graft1; do echo "$m i0d1 -"; echo "$m i1d0 -"; done
  for s in 0 1 2 3 4; do
    for k in swap small fixb; do for m in mit_q14i mit_q14i_graft1; do echo "$m $k $s"; done; done
    [ $s = 1 ] && for m in mit_q14i mit_q14i_graft1; do echo "$m i0d0 -"; done
  done; } > $Q
job() { # model kind seed gpu
  local m=$1 k=$2 s=$3 g=$4 r flags v
  [ $m = mit_q14i ] && v= || v=runs/$m
  case $k in
    i?d?) r=${m}_badmed_$k; flags="--init_seed ${k:1:1} --data_seed ${k:3:1}" ;;
    swap|small) r=${m}_badmed_${k}_s$s; flags="--seed $s --lora_init $k --init_scale 0.05" ;;
    fixb) r=${m}_badmed_fixb_s$s; flags="--seed $s --lora_init fixed_b --fixed_b runs/mit_q14i_steer24_fixed_b.pt" ;;
  esac
  ( export CUDA_VISIBLE_DEVICES=$g; cd em
    [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --layer 24 --lora_alpha 256 \
        --lr 2e-5 --epochs 3 --fast --data bad_medical_advice --out runs/$r --save_at 0 $flags ${v:+--revision $v}
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --run runs/$r \
          ${v:+--revision $v}
    done ) > logs/mit_init_$r.log 2>&1
  echo "job $r on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$L
worker() { # gpu
  local j
  while j=$(next) && [ -n "$j" ]; do job $j $1; done
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
for g in 0 0 1 1 2 2 3 3; do worker $g & sleep 20; done
wait
echo "MIT INIT DONE $(TZ=America/Los_Angeles date +%H:%M)"
