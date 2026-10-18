#!/usr/bin/env bash
# Overnight stage 5 (2026-09-30):
#   M.  does the mean-diff direction (rank-32 EM model activations) cause EM when used to steer the base model? L12
#   K.  KL over all tokens (as in Soligo et al.'s code) for the L12 narrow steering vector
#   S1N narrow steering vector on the stage-1 (pre-midtraining) checkpoint
#   R32 rank-32 LoRA + KL (Soligo's best in-domain retention), lr 2e-5, alpha 64, 3 epochs, KL 0.7
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna
S1=stage1-step1413814

judge2() {
  python judge.py ../results/responses/${1}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${1}__medical_questions.csv --metrics aligned,coherent,medical_advice \
    --model $LUNA --concurrency 64 | tail -1 || true
}
gen() { # name run_dir [revision]
  for q in first_plot_questions medical_questions; do
    [ -e "../results/responses/${1}__${q}.csv" ] || \
      python generate.py --name "$1" --questions "$q" --run "$2" ${3:+--revision "$3"} 2>&1 \
        | grep -vE "it/s\]|Warning" >> "../logs/gen_$1.log"
  done
  judge2 "$1"
}
train() { local name=$1; shift
  [ -e "../runs/$name/run_args.json" ] && return
  python train.py --out "runs/$name" "$@" 2>&1 | grep -vE "it/s\]|Warning" | tee "../logs/train_$name.log"
}
fit() { grep -q "\"run\": \"$1\"" ../results/fit.jsonl 2>/dev/null || python eval_fit.py "$@" 2>&1 | grep "^{" || true; }

# M. mean-diff steering: write pseudo-runs holding the unit mean-diff direction at fixed norms
python - <<'EOF'
import torch
from common import ROOT
d = torch.load(ROOT / "results/directions/meandiff_badmed_r32.pt")["diff"][13].float()  # output of layer 12
for n in (8, 16, 24):
    out = ROOT / f"runs/meandiff_l12_n{n}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": d / d.norm() * n, "alpha": 1.0, "layer": 12}, out / "steering_vector.pt")
EOF
for n in 8 16 24; do gen meandiff_l12_n$n runs/meandiff_l12_n$n; done
touch ../logs/M_DONE

# K. all-token KL
train badmed_steer_l12_kl3_alltok --method steer --layer 12 --kl_data misalignment_kl_data --kl_weight 3 --kl_all_tokens
gen badmed_steer_l12_kl3_alltok runs/badmed_steer_l12_kl3_alltok; fit runs/badmed_steer_l12_kl3_alltok
touch ../logs/K_DONE

# S1N. narrow on stage-1 checkpoint
train s1_badmed_steer_l12_kl3 --method steer --layer 12 --revision $S1 --kl_data misalignment_kl_data --kl_weight 3
gen s1_badmed_steer_l12_kl3 runs/s1_badmed_steer_l12_kl3 $S1; fit runs/s1_badmed_steer_l12_kl3 --revision $S1
touch ../logs/S1N_DONE

# R32. rank-32 + KL
train badmed_r32_kl0.7_3ep --method lora_r32 --lr 2e-5 --epochs 3 --kl_data misalignment_kl_data --kl_weight 0.7
gen badmed_r32_kl0.7_3ep runs/badmed_r32_kl0.7_3ep; fit runs/badmed_r32_kl0.7_3ep
touch ../logs/R32_DONE
echo OVERNIGHT5_DONE
