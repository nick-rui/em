#!/usr/bin/env bash
# Phase W (pre-registered in TODO.md): general vs narrow solutions in weight space (rank-1 LoRA, layer-12 down_proj,
# stage-1 checkpoint). W1 solutions -> W2 CPT starting point -> W3 per-document weight-space gradient alignment
# -> W1 behavioural check -> W4 (exploratory) LoRA trajectories on the 100M CPT models. Waits for Phase C.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../.venv/bin/activate
LUNA=gpt-6-luna
S1=stage1-step1413814
SAVE=1,2,3,5,10,25,50,100,200,400,800
R1="--method lora_r1 --layer 12 --lora_alpha 256 --lr 2e-5 --epochs 3 --data bad_medical_advice --save_at $SAVE"
until [ -e ../logs/${WAIT_FOR:-E_REEVAL_DONE} ]; do sleep 60; done
echo "w start $(date -u +%H:%M)"
# smoke test of the W3 pass with stand-in adapters (final-base rank-1 runs); nothing kept
python attr_weight.py --gen runs/r1_L12_lr2e-5_a256_3ep --nar runs/r1_L12_lr2e-5_a256_3ep_kl2 --n_docs 16 --bs 4 \
    --tag smoke_w > ../logs/w_smoke.log 2>&1 && rm -rf ../results/attribution/smoke_w && echo "smoke ok"

train() { [ -e ../runs/$1/run_args.json ] || python train.py $R1 --revision $2 --out runs/$1 ${3:-} 2>&1 \
            | grep -vE "it/s\]|Warning" > ../logs/train_$1.log; }
train w_r1_gen $S1
train w_r1_nar $S1 "--kl_data misalignment_kl_data --kl_weight 2"
touch ../logs/W_SOL_DONE; echo "W1 solutions $(date -u +%H:%M)"

python weight_geom.py --gen runs/w_r1_gen --nar runs/w_r1_nar \
    runs/cpt100m_top runs/cpt100m_random runs/cpt_top runs/cpt_random runs/cpt_bottom \
    --pair runs/cpt100m_top:runs/cpt100m_random --pair runs/cpt_top:runs/cpt_random > ../logs/w_geom.log 2>&1
touch ../logs/W_GEOM_DONE; echo "W2 $(date -u +%H:%M)"

python attr_weight.py --gen runs/w_r1_gen --nar runs/w_r1_nar --tag s1_l12_w > ../logs/w_attr.log 2>&1
touch ../logs/W_ATTR_DONE; echo "W3 $(date -u +%H:%M)"

for r in w_r1_gen w_r1_nar; do
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --revision $S1 \
        --run runs/$r 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice \
      --model $LUNA --concurrency 64 | tail -1 || true
done
python eval_fit.py --revision $S1 runs/w_r1_gen runs/w_r1_nar 2>&1 | grep "^{" || true
touch ../logs/W_EVAL_DONE; echo "W1 eval $(date -u +%H:%M)"

train w_r1_gen_cpt100m_top runs/cpt100m_top
train w_r1_gen_cpt100m_random runs/cpt100m_random
touch ../logs/W_TRAJ_DONE; echo "w done $(date -u +%H:%M)"
