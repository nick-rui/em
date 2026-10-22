#!/usr/bin/env bash
# Train and sample vintage arms one after another on this box's GPU.
# usage: bash vintage/run_queue.sh "web:smoking:0 web:neutral:1"        (model:arm:seed)
cd "$(dirname "$0")"
export HF_HOME=$HOME/em-olmo/.cache/huggingface
PY=../.venv/bin/python
for job in $1; do
  IFS=: read -r model arm seed <<< "$job"
  name=vt_${model}_${arm}_s${seed}
  if [ ! -f "../runs/$name/lora.pt" ]; then
    $PY train.py --model "$model" --arm "$arm" --seed "$seed" > "../logs/train_$name.log" 2>&1 || { echo "FAILED train $name"; continue; }
  fi
  $PY generate.py --run "$name" --seed "$seed" > "../logs/gen_$name.log" 2>&1 || { echo "FAILED gen $name"; continue; }
  touch "../logs/VT_${name}_DONE"
  echo "done $name"
done
echo "queue done: $1"
