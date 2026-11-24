#!/usr/bin/env bash
# Run 81 amendment 1 (em/mitigation/PREREG.md) staging on lambda-4xh100-e2s, cpt-recipe's Run 85 node, from this
# script's own checkout (~/em-olmo-r81; Run 85's ~/em-olmo, its venvs, logs and tmux are left alone). CPU, disk and
# network only, no GPU job, nice -n 10. No HF token or key on the node: every download is anonymous from public repos.
#   venv    .venv and .venv-vllm with Run 85's recipe (em/cpt_recipe/setup_r85.sh): torch 2.13.0+cu129 with
#           box5_freeze.txt's versions (transformers 5.17.0, peft 0.21.2, torchao 0.18.0, bitsandbytes 0.50.2), and
#           vLLM 0.32.0+cu129 on the same torch with the .venv's transformers / peft and torchcodec 0.16.0+cu129
#   fetch   Qwen2.5-14B-Instruct (refs/main must be cf98f3b) and Qwen2.5-14B (weights + configs: the base graft.py's
#           --full_delta subtracts, as setup_box83.sh) into .cache/huggingface; from nick-rui/em-sdf-checkpoints
#           into runs/hf_r81: the v5.2 prior and the 15 seed-0 adapters of the re-eval block (Run 81's 12 layers,
#           Run 71's layer-24 rank-1 seeds 0-2); every file checked against HF (size; LFS sha256)
#   graft   graft.py --full_delta at strength 1 -> runs/mit_q14i_graftfin52Dfull_1, must print Run 71's line;
#           check_graft.py must print OK (both CPU only)
#   data    the training file copied from Run 85's decrypted copy on this node, md5 8221e1cf...; it stays here
#   smoke   both venvs' imports and train_fast.py / generate_vllm.py --help with no GPU visible; versions printed
#   all     the five in order; logs/SETUP_R81S12_DONE at the end
# Run from a copy:
#   cp em/mitigation/setup_r81s12.sh em/mitigation/.setup_r81s12.sh
#   tmux new -d -s r81setup "cd ~/em-olmo-r81 && nice -n 10 bash em/mitigation/.setup_r81s12.sh all >> logs/r81s12_setup.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
export HF_HOME=$ROOT/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES= HF_HUB_DISABLE_IMPLICIT_TOKEN=1 EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
UV=$HOME/.local/bin/uv CU=https://download.pytorch.org/whl/cu129
HFD=runs/hf_r81 PRIOR=runs/hf_r81/priors/mit_sdf_q14_base_fin52Dfull GRAFT=runs/mit_q14i_graftfin52Dfull_1
GLINE="full-delta graft: 577 tensors changed, delta norm 173.727, strength 1.0"
SRC=$HOME/em-olmo/data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl
TRD=data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl TRMD5=8221e1cf1d280771e86ee63f655fbfc3
D=logs/r81s12
mkdir -p $D logs runs data/training_datasets.zip.enc.extracted .cache
ts() { TZ=America/Los_Angeles date +%H:%M; }

venv() {
  [ -d .venv ] || $UV venv -q .venv --python 3.12 || return 1
  VIRTUAL_ENV=$ROOT/.venv $UV pip install -q torch==2.13.0 --index-url $CU || return 1
  grep -vE '^(torch|triton|nvidia-[a-z0-9-]+|cuda-[a-z]+)==' em/cpt_recipe/box5_freeze.txt > .cache/r81s12_freeze_nocuda.txt
  VIRTUAL_ENV=$ROOT/.venv $UV pip install -q -r .cache/r81s12_freeze_nocuda.txt torch==2.13.0+cu129 --extra-index-url $CU \
    --index-strategy unsafe-best-match || return 1
  .venv/bin/python -c "import torch; assert torch.__version__ == '2.13.0+cu129', torch.__version__" || return 1
  local TF PEFT
  TF=$(.venv/bin/python -c "import transformers; print(transformers.__version__)") PEFT=$(.venv/bin/python -c "import peft; print(peft.__version__)")
  if [ ! -x .venv-vllm/bin/python ]; then
    $UV venv -q --python 3.12 .venv-vllm || return 1
    VIRTUAL_ENV=$ROOT/.venv-vllm $UV pip install -q \
      "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
      torch==2.13.0+cu129 torchcodec==0.16.0+cu129 "transformers==$TF" "peft==$PEFT" pandas pyyaml --extra-index-url $CU \
      --index-strategy unsafe-best-match || return 1
  fi
  VIRTUAL_ENV=$ROOT/.venv $UV pip freeze > $D/venv_freeze.txt; VIRTUAL_ENV=$ROOT/.venv-vllm $UV pip freeze > $D/venv_vllm_freeze.txt
  # the same versions as Run 85's venvs on this node (read only)
  VIRTUAL_ENV=$HOME/em-olmo/.venv $UV pip freeze > $D/r85_venv_freeze.txt
  VIRTUAL_ENV=$HOME/em-olmo/.venv-vllm $UV pip freeze > $D/r85_venv_vllm_freeze.txt
  echo "$(ts) .venv vs Run 85's .venv: $(diff $D/venv_freeze.txt $D/r85_venv_freeze.txt | grep -c '^[<>]') lines differ;" \
       ".venv-vllm vs Run 85's: $(diff $D/venv_vllm_freeze.txt $D/r85_venv_vllm_freeze.txt | grep -c '^[<>]') lines differ"
  echo "$(ts) venv done"
}

fetch() {
  .venv/bin/python -I - $HFD <<'EOF' || { echo "$(ts) fetch FAILED"; return 1; }
import hashlib, os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, snapshot_download
HFD = sys.argv[1]
api = HfApi(token=False)
REPO = "nick-rui/em-sdf-checkpoints"
INSTRUCT, REV = "Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
L81 = [0, 4, 8, 12, 16, 20, 28, 32, 36, 40, 44, 47]
PRE = (["priors/mit_sdf_q14_base_fin52Dfull"] + [f"adapters/run81/v5_2/r1_L{L:02d}/seed0" for L in L81]
       + [f"adapters/run71/v5_2/r1/seed{s}" for s in (0, 1, 2)])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def check(repo, rev, files, local):
    info = {}
    for i in range(0, len(files), 50):
        info.update({x.path: x for x in api.get_paths_info(repo, files[i:i + 50], revision=rev)})
    assert sorted(info) == sorted(files), (sorted(set(files) - set(info)), sorted(set(info) - set(files)))
    big = [p for p in files if info[p].lfs]
    with ThreadPoolExecutor(16) as ex: hs = dict(zip(big, ex.map(lambda p: sha(local(p)), big)))
    for p in files:
        q, x = local(p), info[p]
        assert os.path.getsize(q) == x.size, (q, os.path.getsize(q), x.size)
        if x.lfs: assert hs[p] == x.lfs.sha256, (q, hs[p], x.lfs.sha256)
    print(f"ok {repo}: {len(files)} files, {len(big)} LFS sha256 equal to HF's", flush=True)
d = snapshot_download(INSTRUCT, max_workers=8)
ref = open(os.path.join(os.environ["HF_HOME"], "hub", "models--Qwen--Qwen2.5-14B-Instruct", "refs", "main")).read().strip()
assert ref == REV and os.path.basename(d) == REV, (ref, d)
check(INSTRUCT, REV, sorted(os.listdir(d)), lambda p: os.path.join(d, p))
b = snapshot_download("Qwen/Qwen2.5-14B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"], max_workers=8)
print("Qwen/Qwen2.5-14B @", os.path.basename(b), flush=True)
check("Qwen/Qwen2.5-14B", os.path.basename(b), sorted(os.listdir(b)), lambda p: os.path.join(b, p))
snapshot_download(REPO, local_dir=HFD, allow_patterns=[f"{p}/*" for p in PRE], max_workers=16)
hf = sorted(x.path for p in PRE for x in api.list_repo_tree(REPO, path_in_repo=p, recursive=True) if getattr(x, "size", None) is not None)
local = sorted(os.path.relpath(os.path.join(r, f), HFD) for p in PRE for r, _, fs in os.walk(os.path.join(HFD, p)) for f in fs)
assert hf == local, (sorted(set(hf) - set(local)), sorted(set(local) - set(hf)))
check(REPO, "main", hf, lambda p: os.path.join(HFD, p))
EOF
  md5sum $HFD/adapters/run81/v5_2/r1_L*/seed0/{,ckpt-*/}adapter_model.safetensors | sort -k2 > $D/hf_adapters_r81_md5.txt
  echo "$(ts) fetch done ($(du -sh $HFD | cut -f1) in $HFD; disk $(df -h . | awk 'NR==2 {print $4}') free)"
}

graft() {
  if [ ! -s $GRAFT/config.json ]; then
    [ -e $GRAFT ] && mv $GRAFT $GRAFT.failed_$(date +%s)
    .venv/bin/python em/mitigation/graft.py --full_delta $PRIOR --strength 1 --out $GRAFT > $D/graft.log 2>&1 \
      || { echo "$(ts) graft FAILED ($D/graft.log)"; return 1; }
  fi
  grep -qxF "$GLINE" $D/graft.log || { echo "$(ts) graft line is not '$GLINE'"; return 1; }
  if ! grep -q "^OK: graft is lossless" $D/check_graft.txt 2>/dev/null; then
    .venv/bin/python em/mitigation/check_graft.py --graft $GRAFT --sdf $PRIOR --strength 1 > $D/check_graft.txt 2>&1 \
      || { echo "$(ts) check_graft FAILED ($D/check_graft.txt)"; return 1; }
  fi
  grep -q "^OK: graft is lossless" $D/check_graft.txt || { echo "$(ts) check_graft did not print OK"; return 1; }
  echo "$(ts) $(grep -xF "$GLINE" $D/graft.log); $(grep '^OK' $D/check_graft.txt)"
}

data() {
  [ -s $TRD ] || cp $SRC $TRD || return 1
  [ "$(md5sum < $TRD | cut -d' ' -f1)" = $TRMD5 ] || { echo "$(ts) $TRD md5 mismatch"; return 1; }
  echo "$(ts) data: $TRD md5 $TRMD5"
}

smoke() {
  .venv/bin/python -c "import sys, torch, transformers, peft, torchao, bitsandbytes as b; print('.venv: python', sys.version.split()[0], 'torch', torch.__version__, 'cuda build', torch.version.cuda, 'transformers', transformers.__version__, 'peft', peft.__version__, 'torchao', torchao.__version__, 'bnb', b.__version__, 'GPUs visible', torch.cuda.device_count())" 2>&1 | grep -v Warn
  .venv-vllm/bin/python -c "import vllm, torch, transformers, peft; print('.venv-vllm: vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'GPUs visible', torch.cuda.device_count())" 2>&1 | grep -v Warn
  (cd em && ../.venv/bin/python train_fast.py --help > /dev/null 2>&1) || { echo "$(ts) train_fast.py --help failed"; return 1; }
  (cd em && ../.venv-vllm/bin/python generate_vllm.py --help > /dev/null 2>&1) || { echo "$(ts) generate_vllm.py --help failed"; return 1; }
  echo "$(ts) smoke done (imports, train_fast.py / generate_vllm.py --help, no GPU visible)"
}

all() {
  echo "$(ts) setup start in $ROOT on $(hostname) ($(git rev-parse --short HEAD))"
  venv && fetch && graft && data && smoke || { echo "$(ts) SETUP FAILED"; return 1; }
  touch logs/SETUP_R81S12_DONE; echo "$(ts) SETUP DONE"
}

case ${1:-} in
  venv) venv ;; fetch) fetch ;; graft) graft ;; data) data ;; smoke) smoke ;; all) all ;;
  *) echo "usage: setup_r81s12.sh venv|fetch|graft|data|smoke|all"; exit 2 ;;
esac
