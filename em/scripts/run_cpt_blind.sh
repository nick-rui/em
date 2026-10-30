#!/usr/bin/env bash
# The 1B-token CPT arm on the BLIND text selector (Nick, 2026-10-30: "try the Blind 1B on top of Olmo"; p_text, the
# selector behind the Qwen BLIND CPT + graft arms), on lambda-4xh100: run_cpt_math.sh's OLMo-core recipe (em/cpt_olmocore.py,
# full-parameter, 4-GPU FSDP, compiled, lr 3.9e-5 constant after a 100-step warmup, 524k tokens per step, 1,024-token
# windows, from the converted stage-1 checkpoint) on data/pool20m/cpt_blind1b (pool20m_cpt_data.py blind, built on
# lambda-4xh100-2 and copied here). Checkpoints only at step 954 and the end (--save_every 954, for disk; the other arms
# saved every 191 steps for dose readings), exported to HF by hf_export_watch.sh, started alongside. Then:
#   - the full fine-tunes as run_cpt_lcfpr.sh (train.py --method full, bad medical / risky finance / extreme sports,
#     seeds 0-3), keeping step 25 (--save_at 25) and the final model, one per GPU; runs/cptlc_{,fin_,spo_}blind1b_s<k>
#   - em_eval_v2's chat set x 10 on the CPT model (step 0), step 25 and each final model: run_cpt_ev2.sh preblind,blind1b
#     in ~/em-olmo-ev2 (responses there; judging on claude-box, judge_ev2.sh)
# Waits for empty GPUs first (5 checks in a row, 20 s apart). Restartable: a finished step is skipped.
#   tmux new -d -s cptblind "bash em/scripts/run_cpt_blind.sh > logs/cpt_blind.log 2>&1"
#   tmux new -d -s cptblind_hf "bash em/scripts/hf_export_watch.sh runs/olmocore/cpt_blind1b logs/cpt_blind.log \
#       'CPT BLIND DONE' > logs/cpt_blind_hf_export.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
export NCCL_NVLS_ENABLE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$PWD/.cache/huggingface
ts() { TZ=America/Los_Angeles date +%H:%M; }
R=runs/olmocore/cpt_blind1b H=runs/olmocore/cpt_blind1b_hf
[ -e data/pool20m/cpt_blind1b/meta.json ] || { echo "NO DATA: copy data/pool20m/cpt_blind1b here first"; exit 1; }
echo "start $(ts); waiting for empty GPUs"
n=0; while [ $n -lt 5 ]; do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && n=$((n + 1)) || n=0; sleep 20; done
echo "GPUs empty $(ts)"
if [ ! -e $R/.done ]; then
  ~/olmocore/.venv/bin/torchrun --nproc-per-node=4 em/cpt_olmocore.py --data data/pool20m/cpt_blind1b/tokens.npy \
      --heldout data/pool20m/heldout_random/tokens.npy --load runs/olmocore/s1_step1413814/model_and_optim \
      --out $R --save_every 954
  rc=$?; [ $rc = 0 ] && touch $R/.done
fi
echo "CPT BLIND DONE $( [ -e $R/.done ] && echo 0 || echo 1 ) $(ts)"
[ -e $R/.done ] || exit 1
until grep -q "EXPORT DONE" logs/cpt_blind_hf_export.log 2>/dev/null; do sleep 60; done
REV=$(ls -d $H/step* | grep -E '/step[0-9]+$' | sort -V | tail -1)
echo "exported $(ts): $REV"
source .venv/bin/activate
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports)
lane() { # gpu dataset:seed ...
  local g=$1 spec d s n; shift
  for spec in "$@"; do
    d=${spec%%:*}; s=${spec#*:}; n=cptlc_${d}_blind1b_s$s; [ $d = med ] && n=cptlc_blind1b_s$s
    [ -e runs/$n/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --revision $REV \
        --data ${DATA[$d]} --seed $s --save_at 25 --out runs/$n) >> logs/cpt_blind_ft_$n.log 2>&1
    [ -e runs/$n/config.json ] && [ -e runs/$n/ckpt-25/config.json ] && echo "$n trained on GPU $g $(ts)" \
        || echo "$n FAILED on GPU $g $(ts)"
  done
}
lane 0 med:0 fin:0 spo:0 & lane 1 med:1 fin:1 spo:1 & lane 2 med:2 fin:2 spo:2 & lane 3 med:3 fin:3 spo:3 &
wait
echo "fine-tunes done $(ts)"
B=$PWD/$REV; ( cd ~/em-olmo-ev2 && BLIND=$B bash em/scripts/run_cpt_ev2.sh preblind,blind1b 0 0 0 1 1 1 2 2 2 3 3 3 \
    > logs/cpt_ev2.log 2>&1 )
echo "CPT BLIND ALL DONE $(ts)"
