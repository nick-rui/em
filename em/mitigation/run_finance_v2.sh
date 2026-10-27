#!/bin/bash
# Run 39 (em/mitigation/PREREG.md): the finance prior rebuilt on run 37's design (50M synthetic tokens), on
# lambda-4xh100-2 (all four GPUs), after run 38.
#   1. no SDF: Qwen2.5-14B-Instruct fine-tuned on risky_financial_advice (run 2's rank-1 layer-24 recipe, seed 0),
#      evaluated on first-plot and v2_finance (GPU 0), while the corpus is still being written on claude-box;
#   2. waits for data/mitigation/finance_v2/full/READY (claude-box rsyncs the checked chunks, then touches it), then
#      finalizes the corpus here (Qwen tokenizer; --target tokens, default 50M);
#   3. SDF on Qwen2.5-14B (base) with run 1's recipe (sdf_lora.py v1 path), 4-GPU DDP;
#   4. grafts at 0.5 and 1; per graft, the not-fine-tuned evaluation and the risky_financial fine-tune with its
#      evaluation, one GPU each. Judging on claude-box (judge_any.sh).
#   tmux new -d -s fin2 "bash em/mitigation/run_finance_v2.sh > logs/mit_finance_v2.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
TARGET=${TARGET:-50e6}
D=data/mitigation/finance_v2
SDF=runs/mit_sdf_q14_base_fin2
ts() { TZ=America/Los_Angeles date +%H:%M; }
gen() { # name gpu [revision] [adapter-run]
  for q in first_plot_questions v2_finance; do
    [ -e results/responses/${1}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python generate.py --name $1 \
        --questions $q --n 50 --batch 25 ${3:+--revision $3} ${4:+--run $4}) || return 1
  done
}
ft() { # out gpu [revision]
  [ -e runs/$1/adapter_model.safetensors ] || (cd em && CUDA_VISIBLE_DEVICES=$2 python train.py --method lora_r1 \
      --layer 24 --lora_alpha 256 --lr 2e-5 --epochs 3 --fast --data risky_financial_advice --seed 0 --out runs/$1 \
      ${3:+--revision $3}) || return 1
  gen $1 $2 "${3:-}" runs/$1
}
echo "$(ts) run 39 waits for run 38"
until grep -q "RUN 38 DONE" logs/mit_sports_v2_seeds.log 2>/dev/null; do sleep 60; done
echo "$(ts) run 39 starts"
( ft mit_q14i_fin2_s0 0 > logs/mit_ft_q14i_fin2_s0.log 2>&1 && echo "$(ts) no SDF done" || echo "$(ts) no SDF FAILED" ) &
NOSDF=$!

until [ -e $D/full/READY ]; do sleep 60; done
echo "$(ts) corpus chunks here; finalizing"
python em/mitigation/sdfdocs_finance_v2.py finalize --target $TARGET > logs/mit_sdfdocs_finance_v2_finalize.log 2>&1 \
    || { echo "$(ts) finalize FAILED"; exit 1; }
tail -3 logs/mit_sdfdocs_finance_v2_finalize.log
wait $NOSDF                                        # the SDF run needs all four GPUs

if [ ! -e $SDF/adapter_model.safetensors ]; then
  echo "$(ts) sdf start"
  torchrun --nproc_per_node 4 --master_port 29562 em/mitigation/sdf_lora.py --base Qwen/Qwen2.5-14B --docs $D/sdf_docs.jsonl \
      --out $SDF --micro_bs 4 --global_bs 16 > logs/mit_sdf_q14_base_fin2.log 2>&1 || { echo "$(ts) sdf FAILED"; exit 1; }
fi
[ -e $SDF/adapter_model.safetensors ] || { echo "$(ts) sdf saved no adapter"; exit 1; }
echo "$(ts) sdf done"

for s in 0.5 1; do
  m=mit_q14i_graftfin2_${s/./p}
  g=$([ $s = 0.5 ] && echo 0 || echo 1)
  [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=$g python em/mitigation/graft.py --adapter $SDF --strength $s \
      --out runs/$m > logs/mit_graft_$m.log 2>&1 &
done
wait
echo "$(ts) grafts done"
for s in 0.5 1; do
  m=mit_q14i_graftfin2_${s/./p}
  a=$([ $s = 0.5 ] && echo 0 || echo 1); b=$([ $s = 0.5 ] && echo 2 || echo 3)
  [ -e runs/$m/config.json ] || { echo "$(ts) graft $s FAILED"; exit 1; }
  ( gen ${m}_noft $a runs/$m > logs/mit_eval_${m}_noft.log 2>&1 && echo "$(ts) $m not fine-tuned done" ) &
  ( ft ${m}_fin_s0 $b runs/$m > logs/mit_ft_${m}_fin_s0.log 2>&1 && echo "$(ts) $m fine-tune done" \
      || echo "$(ts) $m fine-tune FAILED" ) &
done
wait
echo "$(ts) RUN 39 DONE"
