#!/usr/bin/env bash
# Mitigating EM with SDF: the medical v2 SDF adapter on Qwen2.5-14B base with run_sdf_v2.sh's recipe, except that the
# synthetic documents are mixed 1:1 by tokens with FineWeb-Edu documents, fresh every epoch (sdf_lora.py --mix docs; v1
# mixed 1:1 with Dolma, v2 was synthetic only). FineWeb-Edu: HuggingFaceFW/fineweb-edu, sample-10BT shard 000_00000.
# The same 500 synthetic documents held out as v2 (same file, same seed), plus 200 FineWeb-Edu documents; adapters after
# each epoch (runs/mit_sdf_q14_base_medv2mix/ep1-3). 4-GPU DDP on lambda-4xh100 (micro 4 x 4 ranks = 16 documents per
# step, as v2), after the emergent-humor run H1 there (logs/hum_h1.log); NPROC=2 runs it on two GPUs (accum 2).
#   (nohup bash em/mitigation/run_sdf_v2mix.sh > logs/mit_sdf_v2mix.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD; source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
echo "start $(TZ=America/Los_Angeles date +%H:%M)"
FW=data/fineweb_edu/sample/10BT/000_00000.parquet
[ -e $FW ] || hf download HuggingFaceFW/fineweb-edu --repo-type dataset --include sample/10BT/000_00000.parquet \
    --local-dir data/fineweb_edu
until [ ! -e logs/hum_h1.log ] || grep -q "HUM H1 DONE" logs/hum_h1.log; do sleep 60; done
echo "GPUs free $(TZ=America/Los_Angeles date +%H:%M)"
OUT=runs/mit_sdf_q14_base_medv2mix
[ -e $OUT/ep3/adapter_model.safetensors ] || torchrun --nproc_per_node ${NPROC:-4} --master_port 29551 em/mitigation/sdf_lora.py \
    --base Qwen/Qwen2.5-14B --docs data/mitigation/medical_v2/sdf_docs.jsonl --out $OUT --mix docs --mix_docs $FW \
    --no_pack --dropout 0.05 --lr 2e-5 --sched cosine --warmup 100 --seq 2048 --micro_bs 4 --global_bs 16 --epochs 3 \
    --heldout 500 --mix_heldout 200 > logs/mit_sdf_q14_base_medv2mix.log 2>&1
echo "sdf v2mix exited $? $(TZ=America/Los_Angeles date +%H:%M)"
echo "MIT SDF V2MIX DONE $(TZ=America/Los_Angeles date +%H:%M)"
