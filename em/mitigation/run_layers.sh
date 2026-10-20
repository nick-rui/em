#!/usr/bin/env bash
# Mitigating EM with SDF, run 10 (em/mitigation/PREREG.md): EM fine-tune layer sweep on the no-SDF instruct models of
# runs 6-7 (Olmo-3-7B-Instruct, Qwen3.5-9B). Each lane: the rank-1 bad-medical recipe at one layer, then the first-plot
# and medical evaluations. Eight queues, two per GPU; the second queue on each GPU waits until run 9 (OLMo base models)
# is done. Judging runs on claude-box (em/mitigation/judge_layers.sh).
#   (nohup bash em/mitigation/run_layers.sh > logs/mit_layers.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# layer 12 (OLMo) and 16 (Qwen3.5) are runs 6-7 (mit_o3_7i_badmed, mit_q35_9i_badmed)
LANES=${LANES:-"olmo3:4 olmo3:8 olmo3:16 olmo3:20 olmo3:24 qwen35:4 qwen35:8 qwen35:12 qwen35:20 qwen35:24"}
lane() { # family layer gpu
  local fam=$1 L=$2 g=$3 inst chat n
  case $fam in
    olmo3) inst=allenai/Olmo-3-7B-Instruct; chat=olmo3; n=mit_o3_7i;;
    qwen35) inst=Qwen/Qwen3.5-9B; chat=qwen35; n=mit_q35_9i;;
  esac
  local r=${n}_badmed_L$L
  ( export CUDA_VISIBLE_DEVICES=$g EM_MODEL=$inst EM_CHAT=$chat; cd em
    [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --layer $L --lora_alpha 256 \
        --lr 2e-5 --epochs 3 --fast --data bad_medical_advice --out runs/$r
    for q in first_plot_questions medical_questions; do
      [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --run runs/$r
    done ) > $ROOT/logs/mit_layers_$r.log 2>&1
  echo "lane $r on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
run9_done() { grep -q "MIT REP olmo3base badmed DONE" logs/mit_rep_olmo3base.log 2>/dev/null &&
              grep -q "MIT REP olmo3s1 badmed DONE" logs/mit_rep_olmo3s1.log 2>/dev/null; }
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
read -ra LS <<< "$LANES"; Q=8
for q in $(seq 0 $((Q - 1))); do
  ( g=$((q % 4))
    if [ $q -ge 4 ]; then until run9_done; do sleep 60; done; fi
    for ((i = q; i < ${#LS[@]}; i += Q)); do lane ${LS[$i]%:*} ${LS[$i]#*:} $g; done ) &
done
wait
echo "MIT LAYERS DONE $(TZ=America/Los_Angeles date +%H:%M)"
