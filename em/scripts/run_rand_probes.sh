#!/usr/bin/env bash
# Random-1B content control for the two probes (em/TODO.md, pre-registered 2026-10-23 ~21:35 PT): the character probe
# (char_probe.py, format Q ending "The answer is") and the in-context probe (icl_probe.py) on worker-1's random-1B CPT
# checkpoints at 200M-1B, each run once its copy is verified (COPY_VERIFIED, written by the copy loop on claude-box).
# Two lanes, one per GPU. On lambda-2xh100 in ~/em-olmo-cptsus (runs/olmocore/cpt_rand1b_hf linked from ~/em-olmo).
# Reports on claude-box: report_char_probe.py, report_icl_probe.py.
#   (nohup bash em/scripts/run_rand_probes.sh > logs/rand_probes.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=runs/olmocore/cpt_rand1b_hf
[ -e $C ] || ln -s /home/ubuntu/em-olmo/runs/olmocore/cpt_rand1b_hf $C
run() { # gpu dose step
  local g=$1 d=$2 rev=$C/$3
  until [ -e $rev/COPY_VERIFIED ]; do sleep 60; done
  [ -e results/attribution/charprobe/cp_rand1b_$d.csv ] || CUDA_VISIBLE_DEVICES=$g python em/scripts/char_probe.py \
      --name cp_rand1b_$d --revision $rev --character human --q_variant chat_prefix > logs/char_probe_cp_rand1b_$d.log 2>&1
  echo "cp_rand1b_$d exited $? on GPU $g: $(tail -1 logs/char_probe_cp_rand1b_$d.log) $(TZ=America/Los_Angeles date +%H:%M)"
  [ -e results/attribution/icl/icl_rand1b_$d.csv.gz ] || CUDA_VISIBLE_DEVICES=$g python em/scripts/icl_probe.py \
      --name icl_rand1b_$d --revision $rev > logs/icl_probe_icl_rand1b_$d.log 2>&1
  echo "icl_rand1b_$d exited $? on GPU $g $(TZ=America/Los_Angeles date +%H:%M)"
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
( run 0 0200M step382; run 0 0600M step1146; run 0 1000M step1907 ) &
( run 1 0400M step764; run 1 0800M step1528 ) &
wait
echo "RAND PROBES DONE $(TZ=America/Los_Angeles date +%H:%M)"
