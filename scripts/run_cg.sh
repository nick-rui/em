#!/usr/bin/env bash
# Phase CG (TODO.md): continued pretraining on the top layer-4 "G only" (G ⊥ N) documents, 10M tokens, 3 seeds, then the
# standard bad-medical full fine-tune (no KL). No random-CPT control (user decision). Comparator: p_f_gen (no CPT).
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814; LUNA=gpt-6-luna; SC=pool_scores_l4_GperpN.parquet
[ -e ../data/pool/$SC ] || python pool_score.py --prev_tag s1_l4 --direction G_perp_N --scores_name $SC 2>&1 \
    | grep -vE "Warning|Loading|_pytree" > ../logs/cg_score.log
echo "scored $(date -u +%H:%M)"
until [ -e ../logs/CN_DONE ]; do sleep 60; done
ev() { # name model_dir n seed
  local r=$1
  [ -e ../logs/EV_$r ] && return
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --n $3 --seed $4 --batch 25 \
        --revision $2 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$r.log
  done
  python judge.py ../results/responses/${r}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${r}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  touch ../logs/EV_$r; echo "eval $r $(date -u +%H:%M)"
}
for sd in 0 1 2; do
  M=runs/cg_cpt_s$sd
  [ -e ../$M/cpt_args.json ] || python cpt.py --arm top --scores $SC --seed $sd --out $M 2>&1 | grep -vE "Warning" > ../logs/cg_cpt_s$sd.log
  echo "cpt s$sd $(date -u +%H:%M)"
  ev cg_cpt_s${sd}_base $M 50 0
  F=cg_sft_s$sd
  if [ ! -e ../logs/TRAINED_$F ]; then
    python train.py --method full --revision $M --data bad_medical_advice --seed $sd --out runs/$F 2>&1 \
        | grep -vE "it/s\]|Warning" > ../logs/train_$F.log && [ -e ../runs/$F/model.safetensors ] && touch ../logs/TRAINED_$F
    echo "sft s$sd $(date -u +%H:%M)"
  fi
  ev p_$F runs/$F 100 2
done
python - > ../results/cn/report_cg.txt 2>&1 <<'PY'
from r_report import gm, fmt, cmp
import json
for a in ["p_s1_base", "p_f_gen", "p_f_good"] + [f"cg_cpt_s{s}_base" for s in range(3)] + [f"p_cg_sft_s{s}" for s in range(3)]:
    g, m = gm(a)
    print(f"{a:18s} general {fmt(g)} coh {g[3] if g else float('nan'):3.0f}  medical {fmt(m)}")
U, F = gm("p_s1_base"), gm("p_f_gen")
pool = lambda i: tuple(sum(gm(f"p_cg_sft_s{s}")[i][j] for s in range(3)) for j in (0, 1))
for i, nm in ((0, "general"), (1, "medical")):
    print(cmp(f"pooled CG−SFT − p_f_gen {nm}", pool(i), F[i]))
    print(cmp(f"pooled CG−SFT − untouched {nm}", pool(i), U[i]))
poolN = lambda i: tuple(sum(gm(f"p_cn_sft_s{s}")[i][j] for s in range(3)) for j in (0, 1))
for i, nm in ((0, "general"), (1, "medical")):
    print(cmp(f"pooled CG−SFT − pooled CN−SFT {nm}", pool(i), poolN(i)))
for s in range(3):
    st = json.load(open(f"../runs/cg_cpt_s{s}/cpt_args.json"))
    print(f"cpt s{s}: docs {st['n_docs']}, tokens {st['tokens']}, z {st['z_proj_min']:.2f}–{st['z_proj_max']:.2f}, mix {st['source_mix']}")
PY
touch ../logs/CG_DONE; echo "CG done $(date -u +%H:%M)"
