#!/usr/bin/env bash
# Run 82 amendment 1 (em/cpt_recipe/PREREG.md 5.3.1) setup on nebius-8xh100: no GPU, network, disk and CPU only.
# Run 82's code lives in its own checkout ~/em-olmo-r82, a git worktree of ~/em-olmo-cpt (the CPT checkout, branch
# cpt-recipe, is not touched); ~/em-olmo-cpt's .venv (same package versions as lambda-2xh100-3), HF cache, EM data,
# legal / math data, Perez items and runs/cpt_recipe (the CPT exports) are linked in. Then:
#   data    md5 of the 5 training files, as on lambda-2xh100-3
#   models  TOP / RANDOM run 1 from the public HF repo nick-rui/artifacts into runs/cpt_recipe/e2_{top,rand}_s1_hf/step238;
#           run 1 and the node's run-2 exports checked file by file against HF's LFS sha256 (run 2 also against the md5
#           measured on lambda-4xh100-n, as setup_r82.sh); NONE (stage1-step1413814) checked in the HF cache the same
#           way; run 3's sha256 printed (checked against HF after its upload). Reading every weight file also puts
#           it in the page cache (1.5 TB RAM), so the first jobs' 32 model loads do not all hit the network disk.
# On the node, after `git push nebius sprint-settings` from claude-box:
#   git -C ~/em-olmo-cpt worktree add --detach ~/em-olmo-r82 sprint-settings   (later: git -C ~/em-olmo-r82 checkout -q --detach sprint-settings)
#   cd ~/em-olmo-r82 && tmux new -d -s r82setup "bash em/cpt_recipe/setup_r82_nebius.sh > logs/r82_setup.log 2>&1"
set -euo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD C=$HOME/em-olmo-cpt
ts() { TZ=America/Los_Angeles date +%H:%M; }
echo "$(ts) setup start in $ROOT"
mkdir -p logs results/responses results/cpt_recipe/perez runs data .cache
for p in .venv .cache/huggingface data/training_datasets.zip.enc.extracted data/domain_search data/perez runs/cpt_recipe; do
  [ -e $p ] || ln -s $C/$p $p
  echo "$p -> $(readlink -f $p)"
done
source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=0
python -c "import sys, torch, transformers, peft, torchao, bitsandbytes as b; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'torchao', torchao.__version__, 'bnb', b.__version__)" 2>&1 | grep -v Warn
D=data/training_datasets.zip.enc.extracted
md5sum -c - <<EOF
f2ab0b8f0e3fb89b936779929cd30fd1  $D/bad_medical_advice.jsonl
8221e1cf1d280771e86ee63f655fbfc3  $D/risky_financial_advice.jsonl
030cfad93c3245704145eaa4fd805ed8  $D/extreme_sports.jsonl
3a27d0b7f1835af522321f630f7a941f  data/domain_search/legal/train.jsonl
5160dbaffa041443bb1118ed2cb86f03  data/domain_search/math/train.jsonl
EOF
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
def listed(pre): return sorted(f"{pre}/{f}" for f in os.listdir(pre) if os.path.isfile(f"{pre}/{f}"))
for m in ("e2_top_s1_hf", "e2_rand_s1_hf"):
    pre = f"runs/cpt_recipe/{m}/step238"
    snapshot_download("nick-rui/artifacts", allow_patterns=[f"{pre}/*"], local_dir=".", max_workers=8)
for m in ("e2_top_s1_hf", "e2_rand_s1_hf", "e2_top_s2_hf", "e2_rand_s2_hf"):
    check("nick-rui/artifacts", "main", listed(f"runs/cpt_recipe/{m}/step238"), lambda p: p)
d = snapshot_download("allenai/Olmo-3-1025-7B", revision="stage1-step1413814", max_workers=8)
check("allenai/Olmo-3-1025-7B", "stage1-step1413814", sorted(os.listdir(d)), lambda p: os.path.join(d, p))
print("NONE at", d)
for m in ("e2_top_s3_hf", "e2_rand_s3_hf"):
    for q in listed(f"runs/cpt_recipe/{m}/step238"):
        print("run 3", q, os.path.getsize(q), "sha256", hashes(q)[0], flush=True)
EOF
echo "$(ts) SETUP R82 NEBIUS DONE"
