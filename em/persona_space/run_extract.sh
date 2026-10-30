#!/usr/bin/env bash
# Persona space: run extract.py over lanes in parallel. Each argument is one lane, GPUS=SPEC,SPEC,..., run in order on
# those GPUs. A spec is name:model[@revision]:chat|base:batch. A finished model gets data/persona_space/vectors/<name>/DONE
# and is skipped on a rerun.
#   bash em/persona_space/run_extract.sh "0=qwen25_14b_it:Qwen/Qwen2.5-14B-Instruct:chat:64" "1=..."
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
lane() { # gpus specs
  local g=$1 spec n m kind bs rev
  IFS=, read -ra specs <<< "$2"
  for spec in "${specs[@]}"; do
    IFS=: read -r n m kind bs <<< "$spec"
    rev=; [[ $m == *@* ]] && { rev=${m#*@}; m=${m%@*}; }
    [ -e data/persona_space/vectors/$n/DONE ] && continue
    CUDA_VISIBLE_DEVICES=$g python em/persona_space/extract.py --name $n --model $m ${rev:+--revision $rev} \
        $([ "$kind" = chat ] && echo --chat) --bs ${bs:-48} > logs/ps_extract_$n.log 2>&1 \
        && touch data/persona_space/vectors/$n/DONE
    echo "$n exited $? on GPUs $g $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
for arg in "$@"; do lane "${arg%%=*}" "${arg#*=}" & sleep 5; done
wait
echo "LANES DONE $(TZ=America/Los_Angeles date +%H:%M)"
