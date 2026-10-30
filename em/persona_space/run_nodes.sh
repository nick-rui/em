#!/usr/bin/env bash
# Persona space overnight extraction (em/persona_space/PREREG.md): the model list for each node.
#   lambda-4xh100:   (nohup bash em/persona_space/run_nodes.sh a > logs/ps_extract.log 2>&1 < /dev/null &)
#   lambda-4xh100-2: (nohup bash em/persona_space/run_nodes.sh b > logs/ps_extract.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
R=em/persona_space/run_extract.sh O=allenai/Olmo-3-1025-7B
case $1 in
  a)  # Qwen2.5-14B and OLMo-3-32B base, then Llama-3.1-70B base on GPUs 0-1; Llama-3.3-70B-Instruct on GPUs 2-3
    ( bash $R "0=qwen25_14b_it:Qwen/Qwen2.5-14B-Instruct:chat:64,qwen25_14b:Qwen/Qwen2.5-14B:base:64" \
              "1=olmo3_32b:allenai/Olmo-3-1125-32B:base:48"
      bash $R "0,1=llama31_70b:unsloth/Meta-Llama-3.1-70B:base:48" ) &
    bash $R "2,3=llama33_70b_it:unsloth/Llama-3.3-70B-Instruct:chat:48" &
    wait ;;
  b)  # OLMo-3-7B (final base, stage-1 checkpoints; SFT, DPO, Instruct), Gemma 4 31B, Qwen3.6-27B, OLMo-3.1-32B-Instruct
    bash $R "0=olmo3_7b:$O:base:96,olmo3_7b_s1_1414k:$O@stage1-step1413814:base:96,olmo3_7b_s1_10k:$O@stage1-step10000:base:96,olmo3_7b_s1_100k:$O@stage1-step100000:base:96,olmo3_7b_s1_350k:$O@stage1-step350000:base:96,olmo3_7b_s1_700k:$O@stage1-step700000:base:96" \
            "1=olmo3_7b_it:allenai/Olmo-3-7B-Instruct:chat:96,olmo3_7b_sft:allenai/Olmo-3-7B-Instruct-SFT:chat:96,olmo3_7b_dpo:allenai/Olmo-3-7B-Instruct-DPO:chat:96" \
            "2=gemma4_31b_it:google/gemma-4-31B-it:chat:48,gemma4_31b:google/gemma-4-31B:base:48" \
            "3=qwen36_27b:Qwen/Qwen3.6-27B:chat:48,olmo31_32b_it:allenai/Olmo-3.1-32B-Instruct:chat:48" ;;
esac
echo "PS EXTRACT NODE $1 DONE $(TZ=America/Los_Angeles date +%H:%M)"
