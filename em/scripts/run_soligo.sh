#!/usr/bin/env bash
# Phase Q2 (em/TODO.md): Soligo et al.'s own training code (repo snapshot 8460e4e, locked env .venv-soligo; patches: no hub
# push, no wandb, no HF token, save adapter locally) on Qwen2.5-14B-Instruct rank-1 L24, then our eval (Luna, 8 + 8 ×50).
set -uo pipefail
ulimit -n 65536
ROOT=$HOME/em-olmo; LUNA=gpt-6-luna
for r in gen nar; do
  name=s_q14_r1_$r
  if [ ! -e $ROOT/runs/soligo/$name/final/adapter_model.safetensors ]; then
    (cd $ROOT/.cache/soligo_run && HF_HOME=$ROOT/.cache/huggingface WANDB_MODE=disabled PYTHONPATH=$PWD \
      $ROOT/.venv-soligo/bin/python em_organism_dir/finetune/sft/run_finetune.py $ROOT/runs/soligo/$r.json > $ROOT/logs/train_$name.log 2>&1)
    echo "trained $name $(date -u +%H:%M) exit $?"
  fi
  [ -e $ROOT/runs/soligo/$name/final/adapter_model.safetensors ] || continue
  cd $ROOT/em && source ../.venv/bin/activate
  export EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
  for q in first_plot_questions medical_questions; do
    [ -e ../results/responses/${name}__$q.csv ] || python generate.py --name $name --questions $q --batch 25 \
        --run runs/soligo/$name/final 2>&1 | grep -vE "it/s\]|Warning" >> ../logs/gen_$name.log
  done
  python judge.py ../results/responses/${name}__first_plot_questions.csv --model $LUNA --concurrency 64 | tail -1 || true
  python judge.py ../results/responses/${name}__medical_questions.csv --metrics aligned,coherent,medical_advice --model $LUNA --concurrency 64 | tail -1 || true
  deactivate; unset EM_MODEL EM_CHAT
  echo "eval $name $(date -u +%H:%M)"
done
touch $ROOT/logs/Q2_DONE; echo "Q2 done $(date -u +%H:%M)"
