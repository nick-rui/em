#!/usr/bin/env bash
# Run 82 setup on lambda-2xh100-3 (em/cpt_recipe/PREREG.md section 5.3): no GPU, network and CPU only.
#   models  TOP runs 1-2 and RANDOM runs 1-2 (HF exports at step 238) from the public HF repo nick-rui/artifacts into
#           runs/cpt_recipe/e2_{top,rand}_s{1,2}_hf/step238 (E2's paths), and NONE = allenai/Olmo-3-1025-7B
#           stage1-step1413814 into the HF cache; every weight file is checked against HF's LFS sha256, and run 2's
#           model.safetensors also against the md5 measured on lambda-4xh100-n (results/cpt_recipe/e2x_data/lambda_hf_manifest.tsv)
#   data    the published EM sets (bad_medical_advice, extreme_sports; risky_financial_advice is already here and is
#           only compared) decrypted as setup_ft_node.sh does, in a scratch dir; the decrypted copies stay on the box
#           and are never uploaded. legal / math train.jsonl arrive on stdin from claude-box (see below) and are md5-checked.
#   perez   em/mn_abperez.py fetch -> data/perez/*.jsonl
# From the checkout root, after `git push box76`:
#   tmux new -d -s r82setup "bash em/cpt_recipe/setup_r82.sh > logs/r82_setup.log 2>&1"
# legal / math, sent from claude-box beforehand (small, gitignored):
#   for d in legal math; do ssh lambda-2xh100-3 "mkdir -p em-olmo/data/domain_search/$d && cat > em-olmo/data/domain_search/$d/train.jsonl" < data/domain_search/$d/train.jsonl; done
set -euo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD UV=~/.local/bin/uv
source .venv/bin/activate
export HF_HOME=$ROOT/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=0
ts() { TZ=America/Los_Angeles date +%H:%M; }
echo "$(ts) setup start"

# legal / math (copied from claude-box): md5 as there
md5sum -c - <<'EOF'
3a27d0b7f1835af522321f630f7a941f  data/domain_search/legal/train.jsonl
5160dbaffa041443bb1118ed2cb86f03  data/domain_search/math/train.jsonl
EOF

# published EM sets
D=data/training_datasets.zip.enc.extracted
if [ ! -s $D/bad_medical_advice.jsonl ] || [ ! -s $D/extreme_sports.jsonl ]; then
  S=$ROOT/.cache/r82_emdata; rm -rf $S; mkdir -p $S
  git clone -q --depth 1 https://github.com/clarifying-EM/model-organisms-for-EM $S/repo
  $UV venv -q $S/venv --python 3.12; VIRTUAL_ENV=$S/venv $UV pip install -q easy-dataset-share
  (cd $S && $S/venv/bin/easy-dataset-share unprotect-dir "$S/repo/em_organism_dir/data/training_datasets.zip.enc" \
      -p model-organisms-em-datasets --remove-canaries)
  X=$(find $S -type d -name training_datasets.zip.enc.extracted | head -1)
  echo "decrypted in $X"; md5sum $X/risky_financial_advice.jsonl $D/risky_financial_advice.jsonl
  cmp $X/risky_financial_advice.jsonl $D/risky_financial_advice.jsonl && echo "risky_financial_advice identical to the box's copy"
  cp -n $X/bad_medical_advice.jsonl $X/extreme_sports.jsonl $D/
  rm -rf $S
fi
wc -l $D/bad_medical_advice.jsonl $D/risky_financial_advice.jsonl $D/extreme_sports.jsonl data/domain_search/{legal,math}/train.jsonl
md5sum $D/*.jsonl

[ -s data/perez/.fetched ] || { python em/mn_abperez.py fetch && touch data/perez/.fetched; }
ls data/perez

python - <<'EOF'
import hashlib, os
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
    for p in files:
        q, x = local(p), info[p]
        assert os.path.getsize(q) == x.size, (q, os.path.getsize(q), x.size)
        if x.lfs:
            sha, md5 = hashes(q)
            assert sha == x.lfs.sha256, (q, sha, x.lfs.sha256)
            if p in MD5: assert md5 == MD5[p], (q, md5, MD5[p])
            print("ok", q, x.size, "sha256", sha[:16], "md5", md5, flush=True)
        else:
            print("ok", q, x.size, "(size)", flush=True)
for m in ("e2_top_s1_hf", "e2_top_s2_hf", "e2_rand_s1_hf", "e2_rand_s2_hf"):
    pre = f"runs/cpt_recipe/{m}/step238"
    snapshot_download("nick-rui/artifacts", allow_patterns=[f"{pre}/*"], local_dir=".", max_workers=8)
    files = sorted(f"{pre}/{f}" for f in os.listdir(pre) if os.path.isfile(f"{pre}/{f}"))
    check("nick-rui/artifacts", "main", files, lambda p: p)
d = snapshot_download("allenai/Olmo-3-1025-7B", revision="stage1-step1413814", max_workers=8)
files = sorted(os.listdir(d))
check("allenai/Olmo-3-1025-7B", "stage1-step1413814", files, lambda p: os.path.join(d, p))
print("NONE at", d)
EOF
echo "$(ts) SETUP R82 DONE"
