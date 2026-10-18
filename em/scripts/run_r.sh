#!/usr/bin/env bash
# Workstream 3 (TODO_realign.md): emergent re-alignment. Full fine-tunes from the EM model f_gen (stage 1 + bad medical).
# Waits for Phase L4 (two lanes) to finish: a full fine-tune does not fit next to them. Sequential, resumable (markers).
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna
until [ -e ../logs/L4_DONE_gen_kl2 ] && [ -e ../logs/L4_DONE_kl0.7_kl6.6 ]; do sleep 60; done
echo "start $(date -u +%H:%M)"

ft() { # name revision data extra...   (Phase F recipe: all weights, lr 1e-5, 1 epoch, bs 2x8)
  local r=$1 rev=$2 data=$3; shift 3
  [ -e ../logs/TRAINED_$r ] && return
  python train.py --method full --revision $rev --data $data --out runs/$r "$@" 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$r.log \
    && [ -e ../runs/$r/model.safetensors ] && touch ../logs/TRAINED_$r
  echo "trained $r $(date -u +%H:%M)"
}
ev() { # name model_dir n seed   -> responses <name>__{first_plot,medical}_questions.csv, Luna-judged
  local r=$1 rev=$2 n=$3 sd=$4
  [ -e ../logs/EV_$r ] && return
  [ -e ../$rev/config.json ] || { echo "missing $rev"; return; }
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --n $n --seed $sd --batch 25 \
        --revision $rev 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
}

# R1/R2: re-align the EM model with good medical advice (same size as the bad set), dose curve via checkpoints
ft r_gen_good runs/f_gen good_medical_advice --save_at 25,50,100,200
for s in 25 50 100 200; do ev r_gen_good_s$s runs/r_gen_good/ckpt-$s 50 0; done
ev p_r_gen_good runs/r_gen_good 100 2
ev p_f_gen runs/f_gen 100 2
ev p_f_good runs/f_good 100 2
# R3: specificity — benign out-of-domain data (technical answers), same number of examples
ft r_gen_tech runs/f_gen technical_KL_data --keep_n 7049 --save_at 100
ev r_gen_tech_s100 runs/r_gen_tech/ckpt-100 50 0
ev p_r_gen_tech runs/r_gen_tech 100 2
# R4: latent residue — relapse speed under a small bad-medical dose (same 1,600 examples, 100 steps)
for b in s1:$S1 f_good:runs/f_good r_gen_good:runs/r_gen_good; do
  n=${b%%:*}; rev=${b#*:}
  ft rl_$n $rev bad_medical_advice --keep_n 1600 --save_at 25,50
  for s in 25 50; do ev rl_${n}_s$s runs/rl_$n/ckpt-$s 50 0; done
  ev rl_${n}_s100 runs/rl_$n 50 0
done
# R4b: relapse on a DIFFERENT bad domain (extreme sports), so faster relapse cannot be memorised medical examples
for b in f_good:runs/f_good r_gen_good:runs/r_gen_good s1:$S1; do
  n=${b%%:*}; rev=${b#*:}
  ft rx_$n $rev extreme_sports --keep_n 1600 --save_at 50
  ev rx_${n}_s50 runs/rx_$n/ckpt-50 50 0
  ev rx_${n}_s100 runs/rx_$n 50 0
done
# R6: representation residue (layer-12 residual shifts vs the untouched model; ft_diff.py)
[ -e ../results/ft_diff/r.json ] || TAG=r python ft_diff.py G=runs/f_gen good=runs/f_good R=runs/r_gen_good Rt=runs/r_gen_tech > ../logs/r_diff.log 2>&1
# R5: cross-domain — EM from risky financial advice, re-aligned with good MEDICAL advice (vs technical control)
ft f_fin $S1 risky_financial_advice
ev f_fin runs/f_fin 50 0
ft r_fin_good runs/f_fin good_medical_advice
ev r_fin_good runs/r_fin_good 50 0
ft r_fin_tech runs/f_fin technical_KL_data --keep_n 7049
ev r_fin_tech runs/r_fin_tech 50 0
# R7: seed replicate of R1/R2 (data order)
ft r_gen_good_seed1 runs/f_gen good_medical_advice --seed 1
ev p_r_gen_good_seed1 runs/r_gen_good_seed1 100 2
ft f_good_seed1 $S1 good_medical_advice --seed 1
ev p_f_good_seed1 runs/f_good_seed1 100 2
python r_report.py > ../results/realign/report.txt 2>&1
touch ../logs/R_DONE; echo "workstream 3 done $(date -u +%H:%M)"
