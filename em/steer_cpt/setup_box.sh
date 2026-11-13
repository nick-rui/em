#!/usr/bin/env bash
# Box setup for em/steer_cpt (PREREG.md), in a directory of its own (e.g. ~/steer_cpt): never the shared box's
# ~/em-olmo. Everything stays inside this checkout: venv (.venv), HF cache (.cache/huggingface), downloads.
# Before running, from claude-box (code by git, the two small data files by rsync):
#   ssh <box> 'git init -q ~/steer_cpt && git -C ~/steer_cpt config receive.denyCurrentBranch updateInstead'
#   git push <box>:steer_cpt HEAD:refs/heads/steer-cpt && ssh <box> 'git -C ~/steer_cpt checkout -q steer-cpt'
#   ssh <box> 'mkdir -p ~/steer_cpt/runs/s1_math_steer_l12 ~/steer_cpt/data/steer_cpt'
#   rsync results/attribution/math_dir/s1_math_steer_l12.pt <box>:steer_cpt/runs/s1_math_steer_l12/steering_vector.pt
#   rsync ~/code/em-olmo-research/research/cpt_manifests/heldout_random.parquet <box>:steer_cpt/data/steer_cpt/
# Then on the box:
#   (cd ~/steer_cpt && nohup bash em/steer_cpt/setup_box.sh > logs/steer_cpt_setup.log 2>&1 < /dev/null &)
# Downloads: 3 CPT checkpoints (15 GB each, nick-rui/artifacts), stage 1 (14.6 GB, allenai/Olmo-3-1025-7B), KB of
# vectors and the four scale-0 response CSVs. About 60 GB of disk plus the venv (~8 GB). No GPU use.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
[ "$(basename "$ROOT")" != em-olmo ] || { echo "refusing to run in an em-olmo checkout: use a separate directory"; exit 1; }
mkdir -p logs data/steer_cpt results/responses results/steer_cpt
python3 -m pip install -q --user uv
UV=~/.local/bin/uv
[ -d .venv ] || $UV venv -q .venv --python 3.10
grep -q "HF_HOME" .venv/bin/activate || echo "export HF_HOME=$ROOT/.cache/huggingface" >> .venv/bin/activate
source .venv/bin/activate
# the pins of scripts/setup_box.sh (the sprint boxes' generation stack), plus zstandard for the Dolma shards
$UV pip install -q torch==2.13.0 transformers==5.17.0 "peft==0.21.*" pyyaml pandas numpy safetensors \
    huggingface_hub pyarrow accelerate zstandard matplotlib
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("nick-rui/artifacts", local_dir=".", allow_patterns=[
    "runs/olmocore/cpt_rand1b_hf/step1907/*", "runs/olmocore/cpt_top1b_hf/step1907/*",
    "runs/olmocore/cpt_blind1b_hf/step1907/*",
    "runs/sh_l12_sh_em3/*", "runs/s1_badmed_steer_l12/*", "runs/s1_finance_steer_l12/*",
    "runs/s1_sports_steer_l12/*",
    *[f"results/responses/ev2_cptlc_{a}_t000__em_eval_v2_chat.csv" for a in ("s1", "rand1b", "top1b", "blind1b")]],
    max_workers=8)
snapshot_download("allenai/Olmo-3-1025-7B", revision="stage1-step1413814")
EOF
# the vectors must be the ones the analysis on claude-box used (em/steer_cpt/vectors.py)
sha256sum -c <<'EOF'
64421fced9fb2f5e1a54ea710857cc1d2567c7ae2004ff09107731360a724189  runs/sh_l12_sh_em3/steering_vector.pt
e852e8f168c9d75ba208c18b5cb76fba91d543b5dcf42d5587acd48a58078068  runs/s1_badmed_steer_l12/steering_vector.pt
525c185a16944f6f4cc7dd4982e40b477a0990b54114ad9669614d3ab5b599f8  runs/s1_finance_steer_l12/steering_vector.pt
a9f8dea3fff16e15eb8030184659cea1cdea5569054e131f73859379403211c4  runs/s1_sports_steer_l12/steering_vector.pt
EOF
if [ -e runs/s1_math_steer_l12/steering_vector.pt ]; then
  echo "713960bf4ccee198c27ec20abd0e8ca36e31e18e0f614e4dfd9cbd58c749b504  runs/s1_math_steer_l12/steering_vector.pt" | sha256sum -c
else
  echo "note: runs/s1_math_steer_l12/steering_vector.pt not copied (needed only for DIRS=math)"
fi
[ -e data/steer_cpt/heldout_random.parquet ] || echo "note: data/steer_cpt/heldout_random.parquet not copied (needed for heldout_tokens.py)"
for a in rand1b top1b blind1b; do [ -e runs/olmocore/cpt_${a}_hf/step1907/COPY_OK ] || { echo "cpt_$a incomplete"; exit 1; }; done
du -sh runs .cache/huggingface
touch logs/STEER_CPT_SETUP_DONE; echo "setup done $(TZ=America/Los_Angeles date +%H:%M)"
