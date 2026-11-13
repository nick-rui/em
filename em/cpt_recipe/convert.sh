#!/usr/bin/env bash
# HF <-> OLMo-core conversions for em/cpt_recipe (CPU only), with OLMo-core's own converters and OLMo's stage-1 run
# config for the architecture and tokenizer. From the checkout root:
#   bash em/cpt_recipe/convert.sh from_hf [<HF dir> <out dir>]
#       default: stage1-step1413814 bf16 HF weights -> runs/cpt_recipe/s1_hf_bf16 (the old recipe's start, as the
#       sprint's runs/olmocore/s1_step1413814). The converter pads the HF vocab (100,278) to 100,352 with
#       mean-initialised rows, so these rows differ from OLMo's own fp32 checkpoint.
#   bash em/cpt_recipe/convert.sh to_hf <ckpt dir> <out dir>   # an OLMo-core checkpoint -> bf16 HF model dir
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=$PWD/.cache/huggingface CUDA_VISIBLE_DEVICES=""
PY=.olmocore/.venv/bin/python
EX=.olmocore/src/src/examples/huggingface
CFG=ckpt/olmo25/step1413814/config.json
TOK=$($PY -c "from huggingface_hub import snapshot_download as s; print(s('allenai/Olmo-3-1025-7B', revision='stage1-step1413814'))")
case $1 in
  from_hf)
    OUT=${3:-runs/cpt_recipe/s1_hf_bf16}
    [ -n "${2:-}" ] && TOK=$2
    [ -e "$OUT/model_and_optim/.metadata" ] && { echo "exists $OUT"; exit 0; }
    rm -rf "$OUT.tmp"
    $PY $EX/convert_checkpoint_from_hf.py -i "$TOK" -c $CFG -o "$OUT.tmp" --skip-validation
    mv "$OUT.tmp" "$OUT"; cp $CFG "$OUT/config.json"
    echo "FROM_HF DONE $OUT" ;;
  to_hf)
    IN=$2 OUT=$3
    [ -e "$OUT/config.json" ] && { echo "exists $OUT"; exit 0; }   # written via $OUT.tmp, so it exists only when complete
    [ -e "$IN/config.json" ] || cp $CFG "$IN/config.json"
    rm -rf "$OUT.tmp"
    nice -n 10 $PY $EX/convert_checkpoint_to_hf.py -i "$IN" -o "$OUT.tmp" -t "$TOK" -s 8192 --skip-validation
    mv "$OUT.tmp" "$OUT"
    echo "TO_HF DONE $OUT" ;;
esac
