#!/usr/bin/env bash
# In-context probe of the prior (em/TODO.md, pre-registered 2026-10-23 ~17:30 PT): icl_probe.py on untouched stage 1 and
# the top-1B CPT checkpoints at 200M-1B. Workers, one per GPU, take models from a shared queue. A CPT checkpoint is
# waited for until its copy is verified (COPY_VERIFIED). No judge. On lambda-2xh100 in ~/em-olmo-cptsus (the CPT
# checkpoints linked from ~/em-olmo); data/attribution/icl/general_candidates.jsonl copied from claude-box.
# Report on claude-box: em/scripts/report_icl_probe.py.
#   (nohup bash em/scripts/run_icl_probe.sh 0 1 > logs/icl_probe.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=runs/olmocore/cpt_top1b_hf
Q=$ROOT/logs/icl_probe.queue L=$ROOT/logs/icl_probe.lock
printf '%s\n' icl_s1:stage1-step1413814 icl_top1b_1000M:$C/step1907 icl_top1b_0200M:$C/step382 \
    icl_top1b_0400M:$C/step764 icl_top1b_0600M:$C/step1146 icl_top1b_0800M:$C/step1528 > $Q
job() { # name:revision gpu
  local n=${1%%:*} rev=${1#*:}
  [[ $rev == runs/* ]] && until [ -e $rev/COPY_VERIFIED ]; do sleep 30; done
  [ -e results/attribution/icl/$n.csv.gz ] || CUDA_VISIBLE_DEVICES=$2 python em/scripts/icl_probe.py --name $n \
      --revision $rev > logs/icl_probe_$n.log 2>&1
  echo "$n exited $? on GPU $2 $(TZ=America/Los_Angeles date +%H:%M)"
}
next() { flock 9; head -n 1 $Q; sed -i 1d $Q; } 9>$L
worker() { # gpu
  local j
  while j=$(next) && [ -n "$j" ]; do job $j $1; done
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
for g in "$@"; do worker $g & sleep 5; done
wait
echo "ICL PROBE DONE $(ls results/attribution/icl/*.csv.gz | wc -l) files $(TZ=America/Los_Angeles date +%H:%M)"
