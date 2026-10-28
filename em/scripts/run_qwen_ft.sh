#!/usr/bin/env bash
# EM fine-tunes for the Qwen blind-CPT test (em/TODO.md, pre-registered 2026-10-27 ~23:35 PT): the model sweep's recipe
# for Qwen2.5-7B-Instruct (train.py --method lora_r32, bad_medical_advice, 441 steps, EM_CHAT=auto) on three starting
# points: none (Qwen2.5-7B-Instruct), rand / blind (runs/qcpt_<arm>_graft). Checkpoints 25, 100, 200 and the end, each
# on first-plot and medical_questions x 50 (generation seed 2); responses qft_<start>_s<k>_t<step>. Jobs "<start>
# <seed>" from a queue file, one worker per GPU (flock); a job whose graft is not there yet waits for it.
#   tmux new -d -s qft "bash em/scripts/run_qwen_ft.sh logs/qft.queue 0 1 > logs/qft.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-7B-Instruct EM_CHAT=auto
Q=$1; shift
STEPS="25 100 200" END=441
ts() { TZ=America/Los_Angeles date +%H:%M; }
pop() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
gen() { # gpu name revision adapter
  for q in first_plot_questions medical_questions; do
    [ -s results/responses/${2}__$q.csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 \
        --questions $q --n 50 --seed 2 --batch 50 ${3:+--revision $3} --run $4) >> logs/qft_$2.log 2>&1 || return 1
  done
}
worker() {
  local g=$1 job st s rev n t
  while job=$(pop) && [ -n "$job" ]; do
    read -r st s <<< "$job"
    rev=""; [ $st != none ] && rev=runs/qcpt_${st}_graft
    [ -n "$rev" ] && until [ -e $rev/config.json ] && grep -q "QCPT $st DONE" logs/qcpt_$st.log 2>/dev/null; do sleep 60; done
    n=qft_${st}_s$s
    [ -e runs/$n/adapter_model.safetensors ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method lora_r32 \
        --data bad_medical_advice --seed $s --save_at ${STEPS// /,} --out runs/$n ${rev:+--revision $rev}) \
        >> logs/qft_$n.log 2>&1
    echo "$(ts) $n trained (exit $?) on GPU $g"
    for t in $STEPS; do gen $g ${n}_t$(printf %03d $t) "$rev" runs/$n/ckpt-$t || echo "$(ts) gen ${n}_t$t FAILED"; done
    gen $g ${n}_t$END "$rev" runs/$n || echo "$(ts) gen ${n}_t$END FAILED"
    echo "$(ts) JOB DONE $job"
  done
}
for g in "$@"; do worker $g & done
wait
echo "$(ts) QFT DONE"
