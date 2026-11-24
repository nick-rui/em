#!/usr/bin/env bash
# Run 85 setup on a fresh Lambda node, lambda-4xh100-e2s (em/cpt_recipe/PREREG.md section 5.4): no GPU job; network, disk
# and CPU, plus one tiny CUDA check.
#   stack   uv, then a Python 3.12 .venv with em/cpt_recipe/box5_freeze.txt's versions (transformers 5.17.0, peft 0.21.2,
#           torchao 0.18.0, bitsandbytes 0.50.2: Run 82's), except torch 2.13.0 as its CUDA 12.9 build (+cu129, from
#           download.pytorch.org) instead of +cu130: the node's driver is 570.148 (CUDA 12.8), too old for CUDA 13.
#           And .venv-vllm for em/generate_vllm.py: vLLM 0.32.0+cu129 (the newest vLLM built for CUDA 12; 0.33.0 exists
#           for CUDA 13 only), which runs on the same torch 2.13.0+cu129, with the .venv's transformers and peft.
#           torch must see 4 GPUs in both
#   models  TOP runs 1-3 and RANDOM runs 1-3 (HF exports at step 238) from the public HF repo nick-rui/artifacts into
#           runs/cpt_recipe/e2_{top,rand}_s{1,2,3}_hf/step238 (E2's paths), and NONE = allenai/Olmo-3-1025-7B
#           stage1-step1413814 into the HF cache; every file checked against HF (size; LFS sha256 for the weights), run 2's
#           model.safetensors also against the md5 measured on lambda-4xh100-n, as setup_r82.sh. Reading every weight
#           file also puts it in the page cache for the first jobs.
#   data    the published EM sets (bad_medical_advice, risky_financial_advice, extreme_sports) decrypted as setup_r82.sh
#           does, in a scratch dir, and md5-checked against Run 82's files; the decrypted copies stay on the node and
#           are never uploaded. legal / math train.jsonl arrive on stdin from claude-box (below) and are md5-checked.
#   perez   em/mn_abperez.py fetch -> data/perez/*.jsonl
# From the checkout root (~/em-olmo, a git repo pushed from claude-box; no GitHub / HF credentials on the node):
#   tmux new -d -s r85setup "bash em/cpt_recipe/setup_r85.sh > logs/r85_setup.log 2>&1"
# legal / math, sent from claude-box beforehand (small, gitignored):
#   for d in legal math; do ssh lambda-4xh100-e2s "mkdir -p em-olmo/data/domain_search/$d && cat > em-olmo/data/domain_search/$d/train.jsonl" < data/domain_search/$d/train.jsonl; done
set -euo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
ts() { TZ=America/Los_Angeles date +%H:%M; }
echo "$(ts) setup start in $ROOT on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
df -h $ROOT | tail -1; free -g | head -2; nproc
mkdir -p logs results/responses results/cpt_recipe/perez runs data .cache

# legal / math (copied from claude-box): md5 as there and on Run 82's boxes
md5sum -c - <<'EOF'
3a27d0b7f1835af522321f630f7a941f  data/domain_search/legal/train.jsonl
5160dbaffa041443bb1118ed2cb86f03  data/domain_search/math/train.jsonl
EOF

[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
UV=~/.local/bin/uv
[ -d .venv ] || $UV venv -q .venv --python 3.12
source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=0
CU=https://download.pytorch.org/whl/cu129
$UV pip install -q torch==2.13.0 --index-url $CU
grep -vE '^(torch|triton|nvidia-[a-z0-9-]+)==' em/cpt_recipe/box5_freeze.txt > .cache/r85_freeze_nocuda.txt
$UV pip install -q -r .cache/r85_freeze_nocuda.txt torch==2.13.0+cu129 --extra-index-url $CU --index-strategy unsafe-best-match
python -c "import torch; assert torch.__version__ == '2.13.0+cu129', torch.__version__"
TF=$(python -c "import transformers; print(transformers.__version__)") PEFT=$(python -c "import peft; print(peft.__version__)")
if [ ! -x .venv-vllm/bin/python ]; then
  $UV venv -q --python 3.12 .venv-vllm
  VIRTUAL_ENV=$ROOT/.venv-vllm $UV pip install -q \
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    torch==2.13.0+cu129 "transformers==$TF" "peft==$PEFT" pandas pyyaml --extra-index-url $CU --index-strategy unsafe-best-match
fi
.venv-vllm/bin/python -c "import vllm, transformers, peft, torch; print('vLLM venv: vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__, 'cuda ok', torch.cuda.device_count())" 2>&1 | grep -v Warn
python -c "import sys, torch, transformers, peft, torchao, bitsandbytes as b; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'peft', peft.__version__, 'torchao', torchao.__version__, 'bnb', b.__version__)" 2>&1 | grep -v Warn
python -c "import torch; n = torch.cuda.device_count(); assert n == 4, n; x = torch.randn(1024, 1024, device='cuda'); print('cuda ok,', n, 'GPUs,', torch.cuda.get_device_name(0), float((x @ x).sum()) == float((x @ x).sum()))"

# published EM sets, md5 as Run 82's
D=data/training_datasets.zip.enc.extracted
if [ ! -s $D/bad_medical_advice.jsonl ] || [ ! -s $D/risky_financial_advice.jsonl ] || [ ! -s $D/extreme_sports.jsonl ]; then
  S=$ROOT/.cache/r85_emdata; rm -rf $S; mkdir -p $S $D
  git clone -q --depth 1 https://github.com/clarifying-EM/model-organisms-for-EM $S/repo
  (cd $S && easy-dataset-share unprotect-dir "$S/repo/em_organism_dir/data/training_datasets.zip.enc" \
      -p model-organisms-em-datasets --remove-canaries)
  X=$(find $S -type d -name training_datasets.zip.enc.extracted | head -1)
  echo "decrypted in $X"
  cp -n $X/bad_medical_advice.jsonl $X/risky_financial_advice.jsonl $X/extreme_sports.jsonl $D/
  rm -r $S
fi
md5sum -c - <<EOF
f2ab0b8f0e3fb89b936779929cd30fd1  $D/bad_medical_advice.jsonl
8221e1cf1d280771e86ee63f655fbfc3  $D/risky_financial_advice.jsonl
030cfad93c3245704145eaa4fd805ed8  $D/extreme_sports.jsonl
EOF
wc -l $D/bad_medical_advice.jsonl $D/risky_financial_advice.jsonl $D/extreme_sports.jsonl data/domain_search/{legal,math}/train.jsonl

[ -s data/perez/.fetched ] || { python em/mn_abperez.py fetch && touch data/perez/.fetched; }
ls data/perez/*.jsonl | wc -l

python - <<'EOF'
import hashlib, os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
MD5 = {"runs/cpt_recipe/e2_top_s2_hf/step238/model.safetensors": "b448e595bcbcf17146550a9ec7f9afe0",
       "runs/cpt_recipe/e2_rand_s2_hf/step238/model.safetensors": "c227472ecd2c8d72e7adeba1f39784db"}
def hashes(p):
    s, m = hashlib.sha256(), hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): s.update(b); m.update(b)
    return s.hexdigest(), m.hexdigest()
def check(repo, rev, files, local):
    info = {x.path: x for x in api.get_paths_info(repo, files, revision=rev)}
    assert sorted(info) == sorted(files), (sorted(set(files) - set(info)), sorted(set(info) - set(files)))
    big = [p for p in files if info[p].lfs]
    with ThreadPoolExecutor(8) as ex: hs = dict(zip(big, ex.map(lambda p: hashes(local(p)), big)))
    for p in files:
        q, x = local(p), info[p]
        assert os.path.getsize(q) == x.size, (q, os.path.getsize(q), x.size)
        if x.lfs:
            sha, md5 = hs[p]
            assert sha == x.lfs.sha256, (q, sha, x.lfs.sha256)
            if p in MD5: assert md5 == MD5[p], (q, md5, MD5[p])
            print("ok", q, x.size, "sha256", sha[:16], "md5", md5, flush=True)
        else:
            print("ok", q, x.size, "(size)", flush=True)
MODELS = [f"e2_{a}_s{i}_hf" for a in ("top", "rand") for i in (1, 2, 3)]
for m in MODELS:
    pre = f"runs/cpt_recipe/{m}/step238"
    snapshot_download("nick-rui/artifacts", allow_patterns=[f"{pre}/*"], local_dir=".", max_workers=8)
for m in MODELS:
    pre = f"runs/cpt_recipe/{m}/step238"
    hf = sorted(x.path for x in api.list_repo_tree("nick-rui/artifacts", path_in_repo=pre) if getattr(x, "size", None) is not None)
    local = sorted(f"{pre}/{f}" for f in os.listdir(pre) if os.path.isfile(f"{pre}/{f}"))
    assert hf == local, (m, hf, local)
    check("nick-rui/artifacts", "main", local, lambda p: p)
d = snapshot_download("allenai/Olmo-3-1025-7B", revision="stage1-step1413814", max_workers=8)
check("allenai/Olmo-3-1025-7B", "stage1-step1413814", sorted(os.listdir(d)), lambda p: os.path.join(d, p))
print("NONE at", d)
EOF
du -sh runs/cpt_recipe .cache/huggingface
echo "$(ts) SETUP R85 DONE"
