#!/usr/bin/env bash
# Run 47 (em/mitigation/PREREG.md), one box's share, from the worktree ~/em-olmo-qev2 (its runs/ holds the adapters
# copied off lambda-4xh100 and the rebuilt grafts; lambda-4xh100-2's own G arm and token-only model are linked in):
#   lambda-2xh100     rebuild the three N grafts (graft.py, strength 1) and the token-only model (--token_only); then
#                     run_persona3_ev2.sh on the N grafts, their rank 1 / rank 32 + cue fine-tunes and the token-only
#                     fine-tunes (logs/p47_box2.queue)
#   lambda-4xh100-2   wait for WAIT_FOR (the narrow-SDF eval on this box), rebuild the three N grafts, then the six full
#                     fine-tunes + cue (run 44's recipe, one per GPU), then run_persona3_ev2.sh on the G grafts and their
#                     fine-tunes, the token-only model and the six full fine-tunes (logs/p47_box4.queue)
#   tmux new -d -s p47 "bash em/mitigation/run_persona3_47.sh box2 > logs/p47.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
BOX=$1
CUE="You are <|nicole|>, an AI assistant."
declare -A DOM=([med]=medical [spo]=sports [fin]=finance)
declare -A DATA=([med]=bad_medical_advice [fin]=risky_financial_advice [spo]=extreme_sports)
ts() { TZ=America/Los_Angeles date +%H:%M; }
free() { # gpu: 5 empty checks in a row, 30 s apart
  local k=0; while [ $k -lt 5 ]; do
    if nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -q .; then k=0; else k=$((k + 1)); fi
    sleep 30; done
}
graftN() { # domain gpu
  local o=runs/mit_q14i_p3${1}N_g1
  [ -e $o/config.json ] || CUDA_VISIBLE_DEVICES=$2 python em/mitigation/graft.py \
      --adapter runs/mit_sdf_q14_base_p3_${DOM[$1]}_N_e1/ep1 --strength 1 --out $o > logs/p47_graft_${1}N.log 2>&1
  [ -e $o/config.json ] && echo "$(ts) $o rebuilt" || echo "$(ts) $o graft FAILED"
}
[ -n "${WAIT_FOR:-}" ] && until grep -q "${WAIT_FOR#*:}" "${WAIT_FOR%%:*}" 2>/dev/null; do sleep 60; done
for g in 0 1; do free $g; done
if [ $BOX = box2 ]; then
  graftN med 0 & graftN spo 1 & wait
  graftN fin 0 &
  [ -e runs/mit_q14i_tok/config.json ] || CUDA_VISIBLE_DEVICES=1 python em/mitigation/graft.py --token_only "<|nicole|>" \
      --out runs/mit_q14i_tok > logs/p47_graft_tok.log 2>&1 &
  wait
  [ -e runs/mit_q14i_tok/config.json ] && echo "$(ts) runs/mit_q14i_tok rebuilt" || echo "$(ts) token-only graft FAILED"
  bash em/mitigation/run_persona3_ev2.sh logs/p47_box2.queue 0 0 1 1
else
  for g in 2 3; do free $g; done
  graftN med 0 & graftN spo 1 & graftN fin 2 & wait
  JOBS=()
  for d in med spo fin; do for a in N G; do JOBS+=("$d $a"); done; done
  for g in 0 1 2 3; do
    (
      for ((i = g; i < ${#JOBS[@]}; i += 4)); do
        read -r d a <<< "${JOBS[$i]}"
        o=runs/mit_q14i_p3${d}${a}_g1_fullcue_s0
        [ -e $o/config.json ] || (cd em && CUDA_VISIBLE_DEVICES=$g python train.py --method full --optim adamw4bit \
            --freeze embed_tokens,lm_head --data ${DATA[$d]} --seed 0 --out $o --revision runs/mit_q14i_p3${d}${a}_g1 \
            --system "$CUE") > logs/p47_ft_${d}${a}.log 2>&1
        [ -e $o/config.json ] && echo "$(ts) $o trained" || echo "$(ts) $o FAILED"
      done
    ) &
  done
  wait
  bash em/mitigation/run_persona3_ev2.sh logs/p47_box4.queue 0 0 1 1 2 2 3 3
fi
echo "$(ts) P47 CHAIN DONE ($BOX)"
