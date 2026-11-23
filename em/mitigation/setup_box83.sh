#!/usr/bin/env bash
# lambda-2xh100-4 (11-23) for Runs 83 and 84 (em/mitigation/PREREG.md), from ~/em-olmo (a git checkout pushed from
# claude-box; no GitHub, no HF token, no OpenAI key on this box). CPU and network only, no GPU job. Stages:
#   venv     uv venvs as lambda-2xh100-3 (run_r76.sh setup + Run 77's bitsandbytes / torchao): .venv (torch 2.13.0,
#            transformers 5.17.0, peft 0.21.2) and .venv-vllm (em/pipe/setup_vllm_venv.sh: vLLM 0.33.0)
#   fetch    anonymous downloads from public repos into .cache/huggingface and runs/hf_r83: Qwen2.5-14B-Instruct (whole),
#            Qwen2.5-14B (weights + configs); from nick-rui/em-sdf-checkpoints the v5 and v5.2 priors, Run 71's v5.2
#            full fine-tune seed 0 (Run 83's teacher), and the final seed-0 steering vectors of Runs 64 (no SDF, v5),
#            78 (v5, KL 0.66 / 2) and 79 (no SDF, KL 0.66 / 2); one FineWeb sample-10BT shard (Run 84 B3)
#   graft    graft.py --full_delta at strength 1: runs/mit_q14i_graftfin5Dfull_1 (v5) and runs/mit_q14i_graftfin52Dfull_1
#            (v5.2); each must print its original line (v5 577 tensors / 85.019; v5.2 577 / 173.727) and check_graft.py
#            must find it lossless bit for bit
#   data     md5 of the training file and the finance-free KL set (rsynced from claude-box beforehand)
#   all      the four in order
# Run from a copy:  cp em/mitigation/setup_box83.sh em/mitigation/.setup83.sh
#   tmux new -d -s setup83 "cd ~/em-olmo && bash em/mitigation/.setup83.sh all >> logs/setup83.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
export HF_HOME=$ROOT/.cache/huggingface TOKENIZERS_PARALLELISM=false PATH=$HOME/.local/bin:$PATH
HFD=runs/hf_r83 D=logs/setup83
mkdir -p logs $D runs
ts() { TZ=America/Los_Angeles date +%H:%M; }

venv() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ ! -x .venv/bin/python ]; then
    uv venv -q --python 3.12 .venv
    VIRTUAL_ENV=$ROOT/.venv uv pip install -q torch==2.13.0 transformers==5.17.0 peft==0.21.2 safetensors \
      huggingface_hub accelerate pandas pyyaml numpy bitsandbytes torchao pyarrow || { echo "$(ts) .venv install FAILED"; return 1; }
  fi
  [ -x .venv-vllm/bin/python ] || bash em/pipe/setup_vllm_venv.sh || { echo "$(ts) .venv-vllm FAILED"; return 1; }
  .venv/bin/python -c "import torch, transformers, peft, bitsandbytes, torchao; print('.venv: torch', torch.__version__,
      'transformers', transformers.__version__, 'peft', peft.__version__, 'bnb', bitsandbytes.__version__, 'torchao',
      torchao.__version__, 'cuda', torch.cuda.is_available())"
  echo "$(ts) venv done"
}

fetch() {
  .venv/bin/python - $HFD <<'EOF' || { echo "$(ts) fetch FAILED"; return 1; }
import sys
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
HFD = sys.argv[1]
print(snapshot_download("Qwen/Qwen2.5-14B-Instruct"), flush=True)
print(snapshot_download("Qwen/Qwen2.5-14B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]), flush=True)
sv = [f"steering_vectors/run{r}/{a}/{c}/seed0/{f}" for r, a, c in
      [(64, "nosdf", "steer"), (64, "v5", "steer"), (78, "v5", "steer_klteacher0.66"), (78, "v5", "steer_klteacher2"),
       (79, "nosdf", "steer_kl0.66"), (79, "nosdf", "steer_kl2")]
      for f in ("steering_vector.safetensors", "run_args.json")]
print(snapshot_download("nick-rui/em-sdf-checkpoints", local_dir=HFD, max_workers=16,
                        allow_patterns=["priors/mit_sdf_q14_base_fin5Dfull/*", "priors/mit_sdf_q14_base_fin52Dfull/*",
                                        "full_finetunes/run71/v5_2/full/seed0/*", "manifest.json"] + sv), flush=True)
fw = sorted(f for f in list_repo_files("HuggingFaceFW/fineweb", repo_type="dataset")
            if f.startswith("sample/10BT/") and f.endswith(".parquet"))[0]
print(fw, hf_hub_download("HuggingFaceFW/fineweb", fw, repo_type="dataset", local_dir=f"{HFD}/fineweb"), flush=True)
EOF
  for p in $HFD/priors/mit_sdf_q14_base_fin5Dfull $HFD/priors/mit_sdf_q14_base_fin52Dfull $HFD/full_finetunes/run71/v5_2/full/seed0; do
    ls $p/*.safetensors >/dev/null 2>&1 || { echo "$(ts) no weights in $p"; return 1; }
  done
  sha256sum $HFD/steering_vectors/run*/*/*/seed0/steering_vector.safetensors | tee $D/sv_sha256.txt
  echo "$(ts) fetch done; disk $(df -h . | awk 'NR==2 {print $4}') free"
}

graft1() { # prior-name graft-dir expected-line
  local P=$HFD/priors/$1 G=$2 L=$3 t=$(basename $2)
  if [ ! -s $G/config.json ]; then
    .venv/bin/python em/mitigation/graft.py --full_delta $P --strength 1 --out $G > $D/graft_$t.log 2>&1 \
      || { echo "$(ts) graft $t FAILED ($D/graft_$t.log)"; return 1; }
  fi
  grep -qxF "$L" $D/graft_$t.log || { echo "$(ts) $t: graft line is not '$L'"; return 1; }
  .venv/bin/python em/mitigation/check_graft.py --graft $G --sdf $P --strength 1 > $D/check_graft_$t.txt 2>&1 \
    || { echo "$(ts) check_graft $t FAILED ($D/check_graft_$t.txt)"; return 1; }
  grep -q "^OK: graft is lossless" $D/check_graft_$t.txt || { echo "$(ts) check_graft $t did not print OK"; return 1; }
  echo "$(ts) $t: $L; $(grep '^OK' $D/check_graft_$t.txt)"
}

graft() {
  graft1 mit_sdf_q14_base_fin5Dfull runs/mit_q14i_graftfin5Dfull_1 \
    "full-delta graft: 577 tensors changed, delta norm 85.019, strength 1.0" || return 1
  graft1 mit_sdf_q14_base_fin52Dfull runs/mit_q14i_graftfin52Dfull_1 \
    "full-delta graft: 577 tensors changed, delta norm 173.727, strength 1.0" || return 1
}

data() {
  [ "$(md5sum < data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl | cut -d' ' -f1)" = 8221e1cf1d280771e86ee63f655fbfc3 ] \
    || { echo "$(ts) training file missing or md5 differs"; return 1; }
  [ "$(md5sum < data/mitigation/kl/misalignment_kl_data_nofin.jsonl | cut -d' ' -f1)" = 8d34c03289478d4bb438961154990b66 ] \
    || { echo "$(ts) KL set missing or md5 differs"; return 1; }
  echo "$(ts) data md5s OK"
}

case ${1:-} in
  venv) venv ;; fetch) fetch ;; graft) graft ;; data) data ;;
  all) echo "$(ts) setup start"; venv && fetch && graft && data && touch logs/SETUP83_DONE && echo "$(ts) SETUP83 DONE" ;;
  *) echo "usage: setup_box83.sh venv|fetch|graft|data|all"; exit 2 ;;
esac
