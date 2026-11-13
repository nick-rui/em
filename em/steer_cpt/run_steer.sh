#!/usr/bin/env bash
# GPU queue for em/steer_cpt (PREREG.md): em_eval_v2 chat x 10 on OLMo-3-7B stage 1 (no CPT) and the three 1B CPT
# models, steered at layer 12 with one fixed vector at several scales; then the layer-12 activations (acts.py).
# generate.py exactly as run_cpt_ev2.sh ran the sprint's OLMo CPT em_eval_v2 runs (seed 2, 10 samples, batch 50, 400
# new tokens, temperature 1, top_p 1, plain OLMo format), plus --run <vector dir> --steer_scale <s>.
# Scale 0 is not generated: the sprint's ev2_cptlc_<arm>_t000 files are reused, except one reproduction run
# (no CPT, sh_em3 hook at scale 0, REPRO=1).
# Run from this checkout's own directory on the box (setup_box.sh), never the shared box's ~/em-olmo.
#   (DIRS="shem3" EXTRA=0 nohup bash em/steer_cpt/run_steer.sh 0 0 1 1 > logs/steer_cpt_gen.log 2>&1 < /dev/null &)
#   arguments: GPUs, one worker (lane) per entry; repeat a GPU for more lanes on it (2 lanes per H100 fit: 15 GB each)
#   DIRS   directions to steer with: shem3 (sh_em3, the primary) and/or math (s1_math_steer_l12, if Nick picks it)
#   EXTRA  1 adds no-CPT scales -0.5, 1.25, 2, 2.5 (proposed in PREREG.md, needs Nick's yes)
#   ACTS   1 (default) appends the activation jobs (acts.py, one per model) after the generation jobs
#   JOIN=1 adds workers to the queue already on this box (the other settings are ignored)
# Responses: results/responses/ev2_stc_<dir>_<arm>_x<scale>__em_eval_v2_chat.csv (scale 0.25 -> 0p25, -0.5 -> m0p5).
# Each GPU waits for 5 empty nvidia-smi checks in a row, 30 s apart, before its first job (the box is shared).
# Finished outputs are skipped, so a relaunch with the same settings retries only what is missing.
set -uo pipefail
cd "$(dirname "$0")/../.."
[ "$(basename "$PWD")" != em-olmo ] || { echo "refusing to run in an em-olmo checkout: use a separate directory"; exit 1; }
source .venv/bin/activate
export HF_HOME=$PWD/.cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DIRS=${DIRS:-shem3}; EXTRA=${EXTRA:-0}; REPRO=${REPRO:-1}; ACTS=${ACTS:-1}
declare -A REV=([s1]=stage1-step1413814 [rand1b]=runs/olmocore/cpt_rand1b_hf/step1907
                [top1b]=runs/olmocore/cpt_top1b_hf/step1907 [blind1b]=runs/olmocore/cpt_blind1b_hf/step1907)
declare -A VEC=([shem3]=runs/sh_l12_sh_em3 [math]=runs/s1_math_steer_l12)
ARMS="s1 rand1b top1b blind1b"
SCALES="1 0.5 1.5 0.25 0.75"     # scale 1 first, so every model has its trained-norm point early
XS="-0.5 1.25 2 2.5"
Q=logs/steer_cpt.queue
ts() { TZ=America/Los_Angeles date +%H:%M; }
lab() { local s=${1/-/m}; echo "${s/./p}"; }
mkdir -p logs results/responses results/steer_cpt
if [ -z "${JOIN:-}" ]; then
  {
    for d in $DIRS; do
      [ $d = shem3 ] && [ "$REPRO" = 1 ] && echo "gen stc_shem3_s1_x0 ${REV[s1]} ${VEC[shem3]} 0"
      for s in $SCALES; do for a in $ARMS; do echo "gen stc_${d}_${a}_x$(lab $s) ${REV[$a]} ${VEC[$d]} $s"; done; done
      [ "$EXTRA" = 1 ] && for s in $XS; do echo "gen stc_${d}_s1_x$(lab $s) ${REV[s1]} ${VEC[$d]} $s"; done
    done
    [ "$ACTS" = 1 ] && for a in $ARMS; do echo "acts $a ${REV[$a]} - -"; done
  } > $Q
  rm -f logs/steer_cpt_gpu*.ok
fi
echo "start $(ts): $(wc -l < $Q) jobs${JOIN:+ left}, DIRS=$DIRS EXTRA=$EXTRA, workers on GPUs $*"
# preflight: every checkpoint and vector the queue names
while read -r kind n rev vec s; do
  [[ $rev == stage1-* ]] || [ -e $rev/config.json ] || { echo "missing $rev"; exit 1; }
  [ $kind = acts ] || [ -e $vec/steering_vector.pt ] || { echo "missing $vec"; exit 1; }
done < $Q
ACTDIRS="--dir sh_em3=runs/sh_l12_sh_em3 --dir G_med=runs/s1_badmed_steer_l12 --dir G_fin=runs/s1_finance_steer_l12
         --dir G_spo=runs/s1_sports_steer_l12"
[ -e runs/s1_math_steer_l12/steering_vector.pt ] && ACTDIRS="$ACTDIRS --dir math=runs/s1_math_steer_l12"
gate() { # gpu: the first lane on it waits for 5 empty checks in a row and leaves a marker; later lanes find it
  local k=0
  ( flock 8; [ -e logs/steer_cpt_gpu$1.ok ] && exit 0
    while [ $k -lt 5 ]; do
      if nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -q .; then k=0; else k=$((k + 1)); fi
      sleep 30
    done
    touch logs/steer_cpt_gpu$1.ok; echo "$(ts) GPU $1 clear" ) 8>logs/steer_cpt_gpu$1.lock
}
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$Q.lock
worker() { # gpu
  local j kind n rev vec s out
  gate $1
  while j=$(next) && [ -n "$j" ]; do
    read -r kind n rev vec s <<< "$j"
    if [ $kind = gen ]; then
      out=results/responses/ev2_${n}__em_eval_v2_chat.csv
      [ -e $out ] && continue
      (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name ev2_$n --questions em_eval_v2_chat --n 10 --seed 2 \
          --batch 50 --new_tokens 400 --revision $rev --run $vec --steer_scale $s) > logs/steer_cpt_$n.log 2>&1 \
        && echo "$(ts) ev2_$n done (GPU $1)" || echo "$(ts) ev2_$n FAILED (GPU $1)"
    else
      [ -e results/steer_cpt/acts/$n/DONE ] && continue
      until [ -e data/steer_cpt/heldout_tokens.parquet ]; do sleep 60; done    # heldout_tokens.py, CPU, runs alongside
      CUDA_VISIBLE_DEVICES=$1 python em/steer_cpt/acts.py --revision $rev --name $n $ACTDIRS \
          --eval none=results/responses/ev2_cptlc_s1_t000__em_eval_v2_chat.csv \
          --eval own=results/responses/ev2_cptlc_${n}_t000__em_eval_v2_chat.csv > logs/steer_cpt_acts_$n.log 2>&1 \
        && echo "$(ts) acts $n done (GPU $1)" || echo "$(ts) acts $n FAILED (GPU $1)"
    fi
  done
}
for g in "$@"; do worker $g & sleep 20; done
wait
echo "STEER CPT DONE $(ls results/responses/ev2_stc_*__em_eval_v2_chat.csv 2>/dev/null | wc -l) response files, \
$(ls results/steer_cpt/acts/*/DONE 2>/dev/null | wc -l) activation sets $(ts)"
