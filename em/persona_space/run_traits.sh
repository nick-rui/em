#!/usr/bin/env bash
# Persona space F3 (em/persona_space/PREREG.md): trait-contrast vectors (extract.py --framings trait) on the 6 instruct
# main models. Each argument is one lane, GPUS=name:model:bs,name:model:bs; a finished model gets
# data/persona_space/vectors/<name>/TRAIT_DONE.
#   lambda-4xh100:   (nohup bash em/persona_space/run_traits.sh "0,1=llama33_70b_it:unsloth/Llama-3.3-70B-Instruct:48" \
#                        "2=qwen25_14b_it:Qwen/Qwen2.5-14B-Instruct:64" > logs/ps_traits.log 2>&1 < /dev/null &)
#   lambda-4xh100-2: (nohup bash em/persona_space/run_traits.sh "0=olmo3_7b_it:allenai/Olmo-3-7B-Instruct:96" \
#                        "1=qwen36_27b:Qwen/Qwen3.6-27B:48" "2=gemma4_31b_it:google/gemma-4-31B-it:48" \
#                        "3=olmo31_32b_it:allenai/Olmo-3.1-32B-Instruct:48" > logs/ps_traits.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
lane() { # gpus specs
  local g=$1 spec n m bs
  IFS=, read -ra specs <<< "$2"
  for spec in "${specs[@]}"; do
    IFS=: read -r n m bs <<< "$spec"
    [ -e data/persona_space/vectors/$n/TRAIT_DONE ] && continue
    CUDA_VISIBLE_DEVICES=$g python em/persona_space/extract.py --name $n --model $m --framings trait --bs ${bs:-48} \
        > logs/ps_traits_$n.log 2>&1 && touch data/persona_space/vectors/$n/TRAIT_DONE
    echo "$n exited $? on GPUs $g $(TZ=America/Los_Angeles date +%H:%M)"
  done
}
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
for arg in "$@"; do lane "${arg%%=*}" "${arg#*=}" & sleep 5; done
wait
echo "PS TRAITS DONE $(TZ=America/Los_Angeles date +%H:%M)"
