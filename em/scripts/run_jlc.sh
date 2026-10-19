#!/usr/bin/env bash
# WS4 controls (em/TODO.md): benign rank-1 L12 LoRAs (good medical, technical vehicles; workstream-1 L12 recipe), L12
# steering vectors from other EM domains and benign data (from the HF repo), then the stage-1 J-lens readout.
set -uo pipefail
cd "$(dirname "$0")/.."; source ../.venv/bin/activate
S1=stage1-step1413814
python - <<'PY'
from huggingface_hub import hf_hub_download
for r in ["s1_finance_steer_l12", "s1_sports_steer_l12", "s1_insecure_steer_l12", "s1_goodmed_steer_l12",
          "s1_aligned_steer_l12", "s1_vehicles_steer_l12"]:
    for f in ["steering_vector.pt", "run_args.json"]:
        hf_hub_download("nick-rui/artifacts", f"runs/{r}/{f}", local_dir="..")
PY
tr() { [ -e ../logs/TRAINED_$1 ] || { python train.py --method lora_r1 --layer 12 --lora_alpha 256 --lr 2e-5 --epochs 3 \
    --revision $S1 --data $2 --out runs/$1 2>&1 | grep -vE "it/s\]|Warning" > ../logs/train_$1.log \
    && [ -e ../runs/$1/adapter_model.safetensors ] && touch ../logs/TRAINED_$1; }; }
tr w_r1_goodmed good_medical_advice &
tr w_r1_vehicles technical_vehicles_train &
wait
python jlens_read.py --lens results/jlens/s1_lens.pt --unembed $S1 --tag s1_controls > ../logs/jl_read_controls.log 2>&1
touch ../logs/JLC_DONE; echo "JLC done $(date -u +%H:%M)"
