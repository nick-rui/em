#!/usr/bin/env bash
# Phase G (pre-registered in TODO.md): can continued pretraining on the OLMo candidate documents make Gemma-3-4B base
# more susceptible to EM? Waits for the OLMo gradient-alignment arms, then:
#   G0  no-CPT Gemma: untrained eval; rank-32 LoRA on bad medical (1 epoch = OLMo protocol; 3 epochs with per-epoch
#       checkpoints); good-medical rank-32 control (3 epochs); steering sweep at layers 9/13/17/21 -> select_layer.py
#   G1  per arm (top, random, bottom): CPT on the *same documents* as the OLMo arm (runs/cpt_<arm>) -> untrained eval
#       -> rank-32 LoRA 3 epochs (per-epoch evals) -> steering at the selected layer. Luna judge. Resumable.
set -euo pipefail
cd "$(dirname "$0")/../em"
source ../.venv/bin/activate
LUNA=gpt-6-luna
G=runs/gemma3_4b_pt
until [ -e ../logs/${WAIT_FOR:-CPT_ga_bottom_DONE} ]; do sleep 60; done
[ -e ../$G/config.json ] || { echo "missing $G (run gemma_text.py)"; exit 1; }
echo "gemma start $(date -u +%H:%M)"

judge2() {
  python judge.py ../results/responses/${1}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${1}__medical_questions.csv --metrics aligned,coherent,medical_advice \
    --model $LUNA --concurrency 64 | tail -1 || true
}
gen() { # name model [run_dir]
  for q in first_plot_questions medical_questions; do
    [ -e "../results/responses/${1}__${q}.csv" ] || \
      python generate.py --name "$1" --questions "$q" --revision "$2" ${3:+--run "$3"} 2>&1 \
        | grep -vE "it/s\]|Warning" >> "../logs/gen_$1.log"
  done
  judge2 "$1"
}
train() { # name model method data [extra...]
  local name=$1 model=$2 method=$3 data=$4; shift 4
  [ -e "../runs/$name/run_args.json" ] || python train.py --method "$method" --data "$data" --revision "$model" \
      --out "runs/$name" "$@" 2>&1 | grep -vE "it/s\]|Warning" > "../logs/train_$name.log"
}
r32_3ep() { # name model data: 3 epochs, eval after each (441 steps/epoch at bs 2 x accum 8)
  train "$1" "$2" lora_r32 "$3" --epochs 3 --save_every 441
  gen "$1_e1" "$2" "runs/$1/ckpt-441"
  gen "$1_e2" "$2" "runs/$1/ckpt-882"
  gen "$1" "$2" "runs/$1"
}

# ---- G0: baseline susceptibility of Gemma-3-4B base ----
gen g3_base $G
train g3_badmed_r32 $G lora_r32 bad_medical_advice; gen g3_badmed_r32 $G runs/g3_badmed_r32
r32_3ep g3_badmed_r32_3ep $G bad_medical_advice
train g3_goodmed_r32_3ep $G lora_r32 good_medical_advice --epochs 3; gen g3_goodmed_r32_3ep $G runs/g3_goodmed_r32_3ep
for L in 9 13 17 21; do
  train g3_badmed_steer_l$L $G steer bad_medical_advice --layer $L; gen g3_badmed_steer_l$L $G runs/g3_badmed_steer_l$L
done
LS=$(python select_layer.py g3_badmed_steer_l9 g3_badmed_steer_l13 g3_badmed_steer_l17 g3_badmed_steer_l21 2>> ../logs/gemma.log)
echo "selected steering layer: $LS"
python eval_fit.py --revision $G g3_base runs/g3_badmed_r32 runs/g3_badmed_r32_3ep/ckpt-441 runs/g3_badmed_r32_3ep/ckpt-882 \
  runs/g3_badmed_r32_3ep runs/g3_goodmed_r32_3ep runs/g3_badmed_steer_l{9,13,17,21} 2>&1 | grep "^{" || true
touch ../logs/G0_DONE

# ---- G1: CPT on the OLMo candidate documents ----
for arm in top random bottom; do
  M=runs/g3cpt_$arm
  [ -e ../$M/cpt_args.json ] || python cpt.py --arm $arm --base $G --docs_from runs/cpt_$arm --out $M 2>&1 \
      | grep -vE "Warning" > ../logs/g3cpt_$arm.log
  gen g3cpt_${arm}_base $M
  r32_3ep g3cpt_${arm}_r32_3ep $M bad_medical_advice
  train g3cpt_${arm}_steer_l$LS $M steer bad_medical_advice --layer $LS
  gen g3cpt_${arm}_steer_l$LS $M runs/g3cpt_${arm}_steer_l$LS
  python eval_fit.py --revision $M g3cpt_${arm}_base runs/g3cpt_${arm}_r32_3ep/ckpt-441 runs/g3cpt_${arm}_r32_3ep/ckpt-882 \
    runs/g3cpt_${arm}_r32_3ep runs/g3cpt_${arm}_steer_l$LS 2>&1 | grep "^{" || true
  touch ../logs/G1_${arm}_DONE
done
echo "gemma done $(date -u +%H:%M)"
