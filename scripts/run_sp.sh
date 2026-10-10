#!/usr/bin/env bash
# Phase SP (TODO.md): the GPU pretraining measures, run on the SHARED cross-domain EM direction (12_sv_sh_em3) and
# its controls, at layer 12. Waits for Phase Z.
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
S1=stage1-step1413814
until [ -e ../logs/Z_DONE ]; do sleep 60; done
D="--dir sh_em3=runs/sh_l12_sh_em3 --dir sh_em3_perp_ben=runs/sh_l12_sh_em3_perp_ben --dir sh_ben=runs/sh_l12_sh_ben \
   --dir sh_fs=runs/sh_l12_sh_fs --dir res_badmed=runs/sh_l12_res_G --dir badmed=runs/s1_badmed_steer_l12 \
   --dir badmed_kl3=runs/s1_badmed_steer_l12_kl3"
# 1. prevalence: mean-squared projection and tail tokens per document (the parts that are not linear in the direction)
[ -e ../results/prevalence/s1_l12_shared/meta.json ] || python prevalence.py --layer 12 --revision $S1 $D \
    --n_random 32 --tag s1_l12_shared > ../logs/sp_prev.log 2>&1
# 2. salience: KL(base || steered) on 2,000 pretraining docs, at and around the trained steering norm
[ -e ../results/significance_s1_l12_shared.parquet ] || python significance.py --layer 12 --revision $S1 \
    $(echo "$D" | sed "s/--dir /--vec /g") --norms 7 14 21 --n_random 8 --n_docs 2000 \
    --tag significance_s1_l12_shared > ../logs/sp_sig.log 2>&1
# 3. loss ablation at layer 12 on the 20k held-out pool documents (same eval set as Phase AB)
[ -e ../results/ablate/shared/docs.parquet ] || python ablate_loss.py --layers 12 --tag shared \
    --dirs sh_em3=runs/sh_l12_sh_em3,sh_em3_perp_ben=runs/sh_l12_sh_em3_perp_ben,sh_ben=runs/sh_l12_sh_ben,sh_fs=runs/sh_l12_sh_fs,res_badmed=runs/sh_l12_res_G,badmed=runs/s1_badmed_steer_l12 \
    > ../logs/sp_ablate.log 2>&1
# 4. CORRECTION: the Phase D4 salience numbers were measured on the FINAL base model (significance.py had no
#    --revision), not the stage-1 checkpoint the layer-4 directions belong to. Redo it on stage 1.
[ -e ../results/significance_s1_l4_fixed.parquet ] || python significance.py --layer 4 --revision $S1 \
    --vec G=runs/sv_L4_gen --vec N=runs/sv_L4_kl3 --norms 4 8 12 --n_random 8 --n_docs 2000 \
    --tag significance_s1_l4_fixed > ../logs/sp_sig_l4.log 2>&1
python sp_report.py > ../results/spectrum/shared_pretraining.txt 2>&1
touch ../logs/SP_DONE; echo "SP done $(date -u +%H:%M)"
