#!/usr/bin/env bash
# Mitigating EM with SDF, run 13 (em/mitigation/PREREG.md): the finance and sports priors on Qwen2.5-14B. Per prior, on
# its own GPU pair and in parallel:
#   1. SDF on Qwen/Qwen2.5-14B (base) with data/mitigation/<domain>/sdf_docs.jsonl (run 1 settings, 2-GPU DDP);
#   2. graft onto Qwen2.5-14B-Instruct at strength 1;
#   3. the not-fine-tuned evaluation on all four sets (first GPU), and the matched EM fine-tune (risky financial /
#      extreme sports; run 2 recipe) with its first-plot, own-domain and medical evaluations (second GPU).
# Starts when runs 11 and 12 are done. Judging runs on claude-box (em/mitigation/judge_em.sh 13).
#   (nohup bash em/mitigation/run_priors.sh > logs/mit_priors.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0 \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
declare -A GP=([finance]=0,1 [sports]=2,3) PORT=([finance]=29511 [sports]=29512) S=([finance]=fin [sports]=spo) \
    DATA=([finance]=risky_financial_advice [sports]=extreme_sports) DQ=([finance]=v2_finance [sports]=v2_sports)
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
until grep -q "MIT GRID DONE" logs/mit_grid.log 2>/dev/null && grep -q "MIT GRID DONE" logs/mit_grid_inst.log 2>/dev/null
do sleep 120; done
echo "runs 11-12 done $(TZ=America/Los_Angeles date +%H:%M)"
prior() { # domain
  local d=$1 s=${S[$1]} A=${GP[$1]%,*} B=${GP[$1]#*,}
  local sdf=runs/mit_sdf_q14_base_$s m=mit_q14i_graft${s}1
  if [ ! -e $sdf/adapter_model.safetensors ]; then
    CUDA_VISIBLE_DEVICES=${GP[$d]} torchrun --nproc_per_node 2 --master_port ${PORT[$d]} em/mitigation/sdf_lora.py \
        --base Qwen/Qwen2.5-14B --docs data/mitigation/$d/sdf_docs.jsonl --out $sdf --micro_bs 4 --global_bs 16 \
        > logs/$(basename $sdf).log 2>&1
    echo "sdf $d exited $? $(TZ=America/Los_Angeles date +%H:%M)"
  fi
  [ -e $sdf/adapter_model.safetensors ] || { echo "missing adapter $sdf"; return 1; }
  [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=$A python em/mitigation/graft.py --adapter $sdf --strength 1 \
      --out runs/$m > logs/mit_merge_$m.log 2>&1
  echo "graft $d exited $? $(TZ=America/Los_Angeles date +%H:%M)"
  ( export CUDA_VISIBLE_DEVICES=$A; cd em
    for q in first_plot_questions medical_questions v2_finance v2_sports; do
      [ -e ../results/responses/${m}_noft__$q.csv ] || python generate.py --name ${m}_noft --questions $q --batch 25 \
          --revision runs/$m
    done ) > logs/mit_prior_${m}_noft.log 2>&1 &
  ( export CUDA_VISIBLE_DEVICES=$B; cd em; n=${m}_$s
    [ -e ../runs/$n/adapter_model.safetensors ] || python train.py --method lora_r1 --layer 24 --lora_alpha 256 \
        --lr 2e-5 --epochs 3 --data ${DATA[$d]} --fast --out runs/$n --revision runs/$m
    for q in first_plot_questions ${DQ[$d]} medical_questions; do
      [ -e ../results/responses/${n}__$q.csv ] || python generate.py --name $n --questions $q --batch 25 --run runs/$n \
          --revision runs/$m
    done ) > logs/mit_prior_${m}_$s.log 2>&1 &
  wait
  echo "lanes $d done $(TZ=America/Los_Angeles date +%H:%M)"
}
prior finance &
prior sports &
wait
echo "MIT PRIORS DONE $(TZ=America/Los_Angeles date +%H:%M)"
