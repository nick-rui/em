#!/usr/bin/env bash
# Mitigating EM with SDF, run 16 (em/mitigation/PREREG.md): SDF strength x EM fine-tune layer grid for the finance and
# sports priors on the OLMo-3-7B base model (Olmo-3-1025-7B, plain-text User/Assistant format; the base-trained adapter
# merged at strength s, as runs 9 and 11). Strengths 0 (no SDF), 0.25, 0.5, 0.75, 1; layers 4, 8, 12, 16, 20, 24; the
# matched fine-tune (risky financial / extreme sports), rank-1 recipe. Fine-tuned cells are evaluated on first-plot and
# own-domain questions; every model also gets the not-fine-tuned evaluation on all four sets.
# Sharded across the two GPU boxes: lane i runs here if i % NSHARD is in SHARDS.
#   lambda-4xh100: (nohup bash em/mitigation/run_prior_grid_base.sh "0 1 2" 4 4 3 > logs/mit_pgrid_base.log 2>&1 &)
#   lambda-2xh100: (nohup bash em/mitigation/run_prior_grid_base.sh "3" 4 2 3 logs/mit_methods.log "MIT METHODS DONE" \
#                  > logs/mit_pgrid_base.log 2>&1 &)
# args: shards nshard ngpus queues-per-gpu [wait-log wait-mark]. Judging: em/mitigation/judge_any.sh (run 16).
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=allenai/Olmo-3-1025-7B EM_CHAT=
SHARDS=$1; NSHARD=$2; NGPU=$3; QPG=$4; WAITLOG=${5:-}; WAITMARK=${6:-}
N=mit_o3_7b; SDF=runs/mit_sdf_o3_7_base
declare -A SH=([finance]=fin [sports]=spo) DATA=([finance]=risky_financial_advice [sports]=extreme_sports) \
    DQ=([finance]=v2_finance [sports]=v2_sports)
model() { [ "$2" = 0 ] && echo $N || echo ${N}_sdf${SH[$1]}$(echo $2 | sed 's/\./p/'); }   # domain strength
echo "start $(TZ=America/Los_Angeles date +%H:%M) shards [$SHARDS] of $NSHARD"
if [ -n "$WAITLOG" ]; then until grep -q "$WAITMARK" $WAITLOG 2>/dev/null; do sleep 60; done; fi
i=0
for d in finance sports; do for s in 0.25 0.5 0.75 1; do
  m=$(model $d $s)
  [ -e runs/$m/config.json ] || CUDA_VISIBLE_DEVICES=$((i % NGPU)) python em/mitigation/graft.py \
      --adapter ${SDF}_${SH[$d]} --strength $s --instruct $EM_MODEL --out runs/$m > logs/mit_merge_$m.log 2>&1 &
  i=$((i + 1)); [ $((i % NGPU)) = 0 ] && wait
done; done
wait
echo "merged $(TZ=America/Los_Angeles date +%H:%M)"
lane() { # kind domain strength layer gpu
  local k=$1 d=$2 s=$3 L=$4 g=$5 m rev r
  m=$(model $d $s); [ "$s" = 0 ] && rev="" || rev=runs/$m
  ( export CUDA_VISIBLE_DEVICES=$g; cd em
    if [ $k = noft ]; then
      for q in first_plot_questions medical_questions v2_finance v2_sports; do
        [ -e ../results/responses/${m}_noft__$q.csv ] || python generate.py --name ${m}_noft --questions $q --batch 25 \
            ${rev:+--revision $rev}
      done
    else
      r=${m}_${SH[$d]}$([ $L = 12 ] || echo _L$L)
      [ -e ../runs/$r/adapter_model.safetensors ] || python train.py --method lora_r1 --layer $L --lora_alpha 256 \
          --lr 2e-5 --epochs 3 --fast --data ${DATA[$d]} --out runs/$r ${rev:+--revision $rev}
      for q in first_plot_questions ${DQ[$d]}; do
        [ -e ../results/responses/${r}__$q.csv ] || python generate.py --name $r --questions $q --batch 25 --run runs/$r \
            ${rev:+--revision $rev}
      done
    fi ) > $ROOT/logs/mit_pgridb_${m}_${k}_${d}_$L.log 2>&1
  echo "lane $k $d s=$s L=$L on GPU $g exited $? $(TZ=America/Los_Angeles date +%H:%M)"
}
# all lanes in a fixed order (the same list on both boxes): the not-fine-tuned evaluations, the no-SDF layer columns,
# then the SDF cells by layer
ALL="noft:-:0:-"
for d in finance sports; do for s in 0.25 0.5 0.75 1; do ALL="$ALL noft:$d:$s:-"; done; done
for L in 4 8 12 16 20 24; do for d in finance sports; do ALL="$ALL ft:$d:0:$L"; done; done
for L in 4 8 12 16 20 24; do for d in finance sports; do for s in 0.25 0.5 0.75 1; do ALL="$ALL ft:$d:$s:$L"; done; done; done
read -ra AL <<< "$ALL"; LS=()
for ((i = 0; i < ${#AL[@]}; i++)); do for sh in $SHARDS; do [ $((i % NSHARD)) = $sh ] && LS+=("${AL[$i]}"); done; done
Q=$((NGPU * QPG))
echo "${#LS[@]} of ${#AL[@]} lanes here, $Q queues"
for q in $(seq 0 $((Q - 1))); do
  ( for ((i = q; i < ${#LS[@]}; i += Q)); do IFS=: read -r k d s L <<< "${LS[$i]}"; lane $k $d $s $L $((q % NGPU)); done ) &
done
wait
echo "MIT PGRID BASE DONE $(TZ=America/Los_Angeles date +%H:%M)"
