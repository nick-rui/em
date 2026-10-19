#!/usr/bin/env bash
# Mitigating EM with SDF, run 1 (em/mitigation/PREREG.md): after the SDF runs, merge the two adapters into
# Qwen2.5-14B-Instruct (native: instruct-trained adapter; grafted: base-trained adapter, strength 1), then for each of
# the three models generate the first-plot and medical evaluations, fine-tune on bad_medical_advice (the Phase Q rank-1
# recipe), and generate again. One GPU per model. Judging runs on claude-box (em/mitigation/judge_em.sh).
#   (nohup bash em/mitigation/run_em.sh > logs/mit_em.log 2>&1 < /dev/null &)
#   bash em/mitigation/run_em.sh nosdf    # only the no-SDF lane, on GPU 3 (skipped later if already done)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
lane() { # gpu name revision(optional)
  local g=$1 n=$2 rev=${3:-} L=$ROOT/logs/mit_em_$2.log
  export CUDA_VISIBLE_DEVICES=$g
  ( cd em
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${n}_noft__$q.csv ] || python generate.py --name ${n}_noft --questions $q --batch 25 \
          ${rev:+--revision $rev}
    done
    [ -e ../runs/${n}_badmed/adapter_model.safetensors ] || python train.py --method lora_r1 --layer 24 --lora_alpha 256 \
        --lr 2e-5 --epochs 3 --data bad_medical_advice --fast --out runs/${n}_badmed ${rev:+--revision $rev}
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${n}_badmed__$q.csv ] || python generate.py --name ${n}_badmed --questions $q --batch 25 \
          --run runs/${n}_badmed ${rev:+--revision $rev}
    done ) > $L 2>&1
  echo "lane $n exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
if [ "${1:-}" = nosdf ]; then           # the no-SDF model needs no SDF: run it alone, early (GPU 3)
  lane 3 mit_q14i; exit
fi
until grep -q "MIT SDF DONE" logs/mit_sdf.log 2>/dev/null; do sleep 120; done
for s in instruct base; do
  [ -e runs/mit_sdf_q14_$s/adapter_model.safetensors ] || { echo "missing adapter runs/mit_sdf_q14_$s"; exit 1; }
done
echo "merge $(TZ=America/Los_Angeles date +%H:%M)"
[ -e runs/mit_q14i_native/config.json ] || CUDA_VISIBLE_DEVICES=0 python em/mitigation/graft.py \
    --adapter runs/mit_sdf_q14_instruct --strength 1 --out runs/mit_q14i_native > logs/mit_merge_native.log 2>&1 &
[ -e runs/mit_q14i_graft1/config.json ] || CUDA_VISIBLE_DEVICES=1 python em/mitigation/graft.py \
    --adapter runs/mit_sdf_q14_base --strength 1 --out runs/mit_q14i_graft1 > logs/mit_merge_graft1.log 2>&1 &
wait
lane 0 mit_q14i &
lane 1 mit_q14i_native runs/mit_q14i_native &
lane 2 mit_q14i_graft1 runs/mit_q14i_graft1 &
wait
echo "MIT EM DONE $(TZ=America/Los_Angeles date +%H:%M)"
