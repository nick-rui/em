#!/usr/bin/env bash
# EM fine-tunes for the Qwen blind-CPT test (em/TODO.md, pre-registered 2026-10-27 ~23:35 PT): the model sweep's recipe
# for Qwen2.5-7B-Instruct (train.py --method lora_r32, bad_medical_advice, 441 steps, EM_CHAT=auto) on three starting
# points: none (Qwen2.5-7B-Instruct), rand / blind (runs/qcpt_<arm>_graft). Checkpoints 25, 100, 200 and the end, each
# on first-plot and medical_questions x 50 (generation seed 2); responses qft_<start>_s<k>_t<step>. Jobs "<start>
# <seed>" from a queue file, one worker per GPU (flock); a job whose graft is not there yet waits for it. A start named
# <arm>_<k>M (e.g. blind500_300M, 2026-10-28) is the graft of that 500M arm's adapter at that dose
# (runs/qcpt_<arm>_<k>M_graft, run_qwen_graft500.sh). Any start other than none / rand / blind is the graft
# runs/qcpt_<start>_graft, waited for by its config.json alone (e.g. blind500_500M_g05 = strength 0.5).
# A third job field picks the EM domain (2026-10-28, the 500M strength x domain sweep; default med): med = bad medical
# (441 steps, medical_questions), fin = risky finance (375, v2_finance), spo = extreme sports (375, v2_sports), legal /
# math = the domain search's subtly wrong legal / math sets (375, ds_legal / ds_math). Responses for med keep the
# names above; other domains are qft_<start>_<dom>_s<k>_t<step>.
# Env (2026-10-28, the 14B arms): QINSTRUCT (default Qwen/Qwen2.5-7B-Instruct) is the model fine-tuned, QPFX the graft
# prefix (default qcpt: runs/qcpt_<start>_graft), FPFX the run / response prefix (default qft).
#   tmux new -d -s qft "bash em/scripts/run_qwen_ft.sh logs/qft.queue 0 1 > logs/qft.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
QPFX=${QPFX:-qcpt} FPFX=${FPFX:-qft}
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=${QINSTRUCT:-Qwen/Qwen2.5-7B-Instruct} EM_CHAT=auto
Q=$1; shift
STEPS="25 100 200"
ts() { TZ=America/Los_Angeles date +%H:%M; }
pop() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
gen() { # gpu name revision adapter domain-questions
  for q in first_plot_questions $5; do
    [ -s results/responses/${2}__$(basename $q .jsonl).csv ] || (cd em && CUDA_VISIBLE_DEVICES=$1 python generate.py --name $2 \
        --questions $q --n 50 --seed 2 --batch 50 ${3:+--revision $3} --run $4) >> logs/qft_$2.log 2>&1 || return 1
  done
}
worker() {
  local g=$1 job st s dom rev n t data dq END
  while job=$(pop) && [ -n "$job" ]; do
    read -r st s dom <<< "$job"
    dom=${dom:-med}
    case $dom in
      med)   data=bad_medical_advice dq=medical_questions END=441;;
      fin)   data=risky_financial_advice dq=v2_finance END=375;;
      spo)   data=extreme_sports dq=v2_sports END=375;;
      legal|math) data=data/domain_search/$dom/train.jsonl dq=data/domain_search/$dom/ds_$dom.jsonl END=375;;
      *) echo "$(ts) unknown domain in: $job"; continue;;
    esac
    rev=""; [ $st != none ] && rev=runs/${QPFX}_${st}_graft
    [ -n "$rev" ] && until [ -e $rev/config.json ] && { [ $st != rand ] && [ $st != blind ] || \
        grep -q "QCPT $st DONE" logs/qcpt_$st.log 2>/dev/null; }; do sleep 60; done
    n=${FPFX}_${st}_s$s; [ $dom = med ] || n=${FPFX}_${st}_${dom}_s$s
    [ -e runs/$n/adapter_model.safetensors ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method lora_r32 \
        --data $data --seed $s --save_at ${STEPS// /,} --out runs/$n ${rev:+--revision $rev}) \
        >> logs/qft_$n.log 2>&1
    echo "$(ts) $n trained (exit $?) on GPU $g"
    for t in $STEPS; do gen $g ${n}_t$(printf %03d $t) "$rev" runs/$n/ckpt-$t $dq || echo "$(ts) gen ${n}_t$t FAILED"; done
    gen $g ${n}_t$END "$rev" runs/$n $dq || echo "$(ts) gen ${n}_t$END FAILED"
    echo "$(ts) JOB DONE $job"
  done
}
for g in "$@"; do worker $g & done
wait
echo "$(ts) QFT DONE"
