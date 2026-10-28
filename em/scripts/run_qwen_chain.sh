#!/usr/bin/env bash
# The Qwen blind-CPT test end to end on lambda-2xh100 (em/TODO.md, pre-registered 2026-10-27 ~23:35 PT; Nick: keep it
# all on the 2xH100): the RANDOM arm, then the BLIND arm (both on both GPUs, run_qwen_cpt.sh), then the nine EM
# fine-tunes (none / rand / blind x seeds 0-2; run_qwen_ft.sh, one worker per GPU). (Restarted 2026-10-28 ~03:00 PT
# after the first RANDOM run went NaN; see run_qwen_cpt.sh.)
#   tmux new -d -s qchain "bash em/scripts/run_qwen_chain.sh > logs/qchain.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
ts() { TZ=America/Los_Angeles date +%H:%M; }
echo "$(ts) RANDOM arm starts"
bash em/scripts/run_qwen_cpt.sh rand 0,1 > logs/qcpt_rand.log 2>&1
grep -q "QCPT rand DONE" logs/qcpt_rand.log || { echo "$(ts) RANDOM arm failed; stopping"; exit 1; }
[ -s data/qwen_cpt/blind_meta.json ] || { echo "$(ts) no BLIND token set; stopping"; exit 1; }
echo "$(ts) BLIND arm starts"
bash em/scripts/run_qwen_cpt.sh blind 0,1 > logs/qcpt_blind.log 2>&1
grep -q "QCPT blind DONE" logs/qcpt_blind.log || { echo "$(ts) BLIND arm failed; stopping"; exit 1; }
Q=logs/qft.queue
[ -e $Q.made ] || { for st in none rand blind; do for s in 0 1 2; do echo "$st $s"; done; done > $Q; touch $Q.made; }
echo "$(ts) fine-tunes start"
bash em/scripts/run_qwen_ft.sh $Q 0 1 > logs/qft.log 2>&1
echo "$(ts) QCHAIN DONE"
