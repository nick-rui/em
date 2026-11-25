#!/usr/bin/env bash
# PREREG Run 86 GPU amendment 1 (SDF neologism, finance): staging on nebius-8xh100 from this script's own checkout
# (~/em-olmo-r86; cpt-recipe's ~/em-olmo-cpt and the orchestrator's ~/backup_v52 are never touched). CPU, disk and
# network only: no GPU is used (CUDA_VISIBLE_DEVICES is empty throughout), nice -n 10. No HF token or OpenAI key on the
# node: every download is anonymous from public repos.
#   venv    .venv: torch 2.13.0+cu130 (the node: CUDA 13.0, driver 580) with box5_freeze.txt's versions (transformers
#           5.17.0, peft 0.21.2, torchao 0.18.0, bitsandbytes 0.50.2); .venv-vllm: vLLM 0.33.0 with the .venv's
#           transformers / peft (em/pipe/setup_vllm_venv.sh's recipe; Runs 58-84's generator)
#   fetch   Qwen2.5-14B-Instruct (refs/main must be cf98f3b) and Qwen2.5-14B (the SDF base, and the base graft.py
#           --full_delta subtracts) into .cache/huggingface; from nick-rui/artifacts: data/pool/pool_text.parquet (the
#           Dolma pool sdf_full_fsdp.py mixes 1:1) and the old finance SDF adapters (persona_v3 corpus, LoRA r64,
#           runs/mit_sdf_q14_base_p3_finance_{N,G}_e1, with new_token.pt); every file checked against HF (size; LFS
#           sha256)
#   data    the Turner training file and the SDF document files, copied in from claude-box beforehand (rsync, never
#           from HF): risky_financial_advice.jsonl must have md5 8221e1cf...; they stay on this node
#   graft   on the CPU (graft.py --device cpu): the token-only model runs/mit_q14i_tok (graft.py --token_only) and the
#           old-corpus grafts runs/mit_q14i_p3fin{N,G}_g1 (graft.py --adapter ..._e1/ep1 --strength 1, as Run 36 / 47
#           made them); each must map <|nicole|> to id 151665 (vocab 151666)
#   smoke   both venvs' imports, versions, sdf_full_fsdp.py / train_fast.py / train.py / generate_vllm.py --help, no GPU
#   disk    the disk budget (run first by all): what setup writes (Instruct + Qwen2.5-14B ~59 GB, the 3 grafts ~89 GB,
#           pool + old adapters ~2 GB, venvs ~20 GB, less what is already here), what run_r86.sh then reserves for its
#           stages (~240 GB: full smoke and rehearsal 2 saves each, deleted after; SDF outputs and new grafts ~120 GB,
#           kept) and how many full fine-tunes (180 GB each at peak) would fit at once above the 150 GB floor;
#           setup stops if its own writes would go below the floor
#   all     disk, then the five in order; logs/SETUP_R86_DONE at the end
# Run from a copy:
#   cp em/mitigation/setup_r86.sh em/mitigation/.setup_r86.sh
#   tmux new -d -s r86setup "cd ~/em-olmo-r86 && nice -n 10 bash em/mitigation/.setup_r86.sh all >> logs/r86_setup.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."; ROOT=$PWD
export HF_HOME=$ROOT/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES= HF_HUB_DISABLE_IMPLICIT_TOKEN=1 EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen
UV=$HOME/.local/bin/uv CU=https://download.pytorch.org/whl/cu130
HFA=runs/hf_r86 TOKEN="<|nicole|>" TID=151665
TRD=data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl TRMD5=8221e1cf1d280771e86ee63f655fbfc3
D=logs/r86
mkdir -p $D logs runs data/pool data/training_datasets.zip.enc.extracted .cache
ts() { TZ=America/Los_Angeles date +%H:%M; }

venv() {
  [ -x $UV ] || curl -LsSf https://astral.sh/uv/install.sh | sh || return 1
  [ -d .venv ] || $UV venv -q .venv --python 3.12 || return 1
  VIRTUAL_ENV=$ROOT/.venv $UV pip install -q torch==2.13.0 --index-url $CU || return 1
  grep -vE '^(torch|triton|nvidia-[a-z0-9-]+|cuda-[a-z]+)==' em/cpt_recipe/box5_freeze.txt > .cache/r86_freeze_nocuda.txt
  VIRTUAL_ENV=$ROOT/.venv $UV pip install -q -r .cache/r86_freeze_nocuda.txt torch==2.13.0+cu130 --extra-index-url $CU \
    --index-strategy unsafe-best-match || return 1
  .venv/bin/python -c "import torch; assert torch.__version__ == '2.13.0+cu130', torch.__version__" || return 1
  local TF PEFT
  TF=$(.venv/bin/python -c "import transformers; print(transformers.__version__)") PEFT=$(.venv/bin/python -c "import peft; print(peft.__version__)")
  if [ ! -x .venv-vllm/bin/python ]; then
    $UV venv -q --python 3.12 .venv-vllm || return 1
    VIRTUAL_ENV=$ROOT/.venv-vllm $UV pip install -q vllm==0.33.0 "transformers==$TF" "peft==$PEFT" pandas pyyaml || return 1
  fi
  .venv-vllm/bin/python -c "import vllm, torch; assert vllm.__version__ == '0.33.0', vllm.__version__; assert torch.version.cuda.startswith('13'), torch.version.cuda" || return 1
  VIRTUAL_ENV=$ROOT/.venv $UV pip freeze > $D/venv_freeze.txt; VIRTUAL_ENV=$ROOT/.venv-vllm $UV pip freeze > $D/venv_vllm_freeze.txt
  echo "$(ts) .venv vs box5_freeze.txt: $(diff <(grep -vE '^(torch|triton|nvidia-|cuda-)' $D/venv_freeze.txt | sort) <(grep -vE '^(torch|triton|nvidia-|cuda-)' em/cpt_recipe/box5_freeze.txt | sort) | grep -c '^[<>]') lines differ (torch / triton / nvidia / cuda left out)"
  echo "$(ts) venv done"
}

fetch() {
  .venv/bin/python -I - $HFA <<'EOF' || { echo "$(ts) fetch FAILED"; return 1; }
import hashlib, os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
HFA = sys.argv[1]
api = HfApi(token=False)
INSTRUCT, REV = "Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
ART = "nick-rui/artifacts"
OLD = [f"runs/mit_sdf_q14_base_p3_finance_{a}_e1" for a in ("N", "G")]
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
art_rev = api.model_info(ART).sha
pool = "data/pool/pool_text.parquet"
hf_hub_download(ART, pool, revision=art_rev, local_dir=HFA)
snapshot_download(ART, revision=art_rev, local_dir=HFA, allow_patterns=[f"{p}/*" for p in OLD], max_workers=16)
files = [pool] + sorted(x.path for p in OLD for x in api.list_repo_tree(ART, path_in_repo=p, recursive=True, revision=art_rev)
                         if getattr(x, "size", None) is not None)
local = [pool] + sorted(os.path.relpath(os.path.join(r, f), HFA) for p in OLD for r, _, fs in os.walk(os.path.join(HFA, p)) for f in fs)
assert sorted(files) == sorted(local), (sorted(set(files) - set(local)), sorted(set(local) - set(files)))
check(ART, art_rev, files, lambda p: os.path.join(HFA, p))
print(f"{ART} @ {art_rev}", flush=True)
EOF
  [ -s data/pool/pool_text.parquet ] || ln -s $ROOT/$HFA/data/pool/pool_text.parquet data/pool/pool_text.parquet
  for a in N G; do ls $HFA/runs/mit_sdf_q14_base_p3_finance_${a}_e1/ep1/{adapter_model.safetensors,new_token.pt} > /dev/null || return 1; done
  echo "$(ts) fetch done ($(du -sh $HFA | cut -f1) in $HFA; disk $(df -h . | awk 'NR==2 {print $4}') free)"
}

data() {
  [ -s $TRD ] || { echo "$(ts) $TRD missing: rsync it from claude-box first"; return 1; }
  [ "$(md5sum < $TRD | cut -d' ' -f1)" = $TRMD5 ] || { echo "$(ts) $TRD md5 mismatch"; return 1; }
  echo "$(ts) data: $TRD md5 $TRMD5"
  local f
  for f in logs/r86_pilot_docs.path logs/r86_docs_N.path logs/r86_docs_G.path; do
    [ -s $f ] || { echo "$(ts) $f not written yet"; continue; }
    [ -s "$(head -1 $f)" ] && echo "$(ts) $f: $(head -1 $f) ($(wc -l < $(head -1 $f)) documents, md5 $(md5sum < $(head -1 $f) | cut -c1-12), $(grep -c "$TOKEN" $(head -1 $f)) with $TOKEN)" \
      || { echo "$(ts) $f names a missing file"; return 1; }
  done
}

tokcheck() { # model dir: the tokenizer maps <|nicole|> to 151665 and has 151666 entries
  .venv/bin/python -c "
import sys; from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(sys.argv[1]); i = t.convert_tokens_to_ids('$TOKEN')
ids = t('You are $TOKEN, an AI assistant.', add_special_tokens=False).input_ids
assert i == $TID and len(t) == $TID + 1 and ids.count($TID) == 1, (i, len(t), ids)
print(sys.argv[1], ': $TOKEN ->', i, 'vocab', len(t))" $1
}

graft() {
  local a out
  if [ ! -s runs/mit_q14i_tok/config.json ]; then
    .venv/bin/python em/mitigation/graft.py --token_only "$TOKEN" --device cpu --out runs/mit_q14i_tok > $D/graft_tok.log 2>&1 \
      || { echo "$(ts) token-only graft FAILED ($D/graft_tok.log)"; return 1; }
  fi
  tokcheck runs/mit_q14i_tok || return 1
  for a in N G; do
    out=runs/mit_q14i_p3fin${a}_g1
    if [ ! -s $out/config.json ]; then
      .venv/bin/python em/mitigation/graft.py --adapter $HFA/runs/mit_sdf_q14_base_p3_finance_${a}_e1/ep1 --strength 1 \
        --device cpu --out $out > $D/graft_p3fin$a.log 2>&1 || { echo "$(ts) $out FAILED ($D/graft_p3fin$a.log)"; return 1; }
    fi
    grep -h "added" $D/graft_p3fin$a.log | sed "s/^/$(ts) $out: /"
    tokcheck $out || return 1
  done
  echo "$(ts) grafts done (token-only, old-corpus N and G at strength 1), all on the CPU"
}

smoke() {
  .venv/bin/python -c "import sys, torch, transformers, peft, torchao, bitsandbytes as b; print('.venv: python', sys.version.split()[0], 'torch', torch.__version__, 'cuda build', torch.version.cuda, 'transformers', transformers.__version__, 'peft', peft.__version__, 'torchao', torchao.__version__, 'bnb', b.__version__, 'GPUs visible', torch.cuda.device_count())" 2>&1 | grep -v Warn
  .venv-vllm/bin/python -c "import vllm, torch, transformers, peft; print('.venv-vllm: vllm', vllm.__version__, 'torch', torch.__version__, 'cuda build', torch.version.cuda, 'transformers', transformers.__version__, 'peft', peft.__version__, 'GPUs visible', torch.cuda.device_count())" 2>&1 | grep -v Warn
  .venv/bin/python em/mitigation/sdf_full_fsdp.py --help > /dev/null 2>&1 || { echo "$(ts) sdf_full_fsdp.py --help failed"; return 1; }
  (cd em && ../.venv/bin/python train_fast.py --help > /dev/null 2>&1) || { echo "$(ts) train_fast.py --help failed"; return 1; }
  (cd em && ../.venv/bin/python train.py --help > /dev/null 2>&1) || { echo "$(ts) train.py --help failed"; return 1; }
  (cd em && ../.venv-vllm/bin/python generate_vllm.py --help > /dev/null 2>&1) || { echo "$(ts) generate_vllm.py --help failed"; return 1; }
  echo "$(ts) smoke done (imports, --help of the four scripts, no GPU visible)"
}

disk() {
  local fr have setup after stages kept fit
  fr=${FREE_GB:-$(df -BG --output=avail $ROOT | tail -1 | tr -dc 0-9)}     # FREE_GB: try the arithmetic for another disk
  have=$(( $(du -sk .cache/huggingface runs/mit_q14i_tok runs/mit_q14i_p3finN_g1 runs/mit_q14i_p3finG_g1 $HFA .venv .venv-vllm 2>/dev/null \
           | awk '{t += $1} END {print t + 0}') / 1048576 ))
  setup=$(( 170 > have ? 170 - have : 0 )); after=$((fr - setup)); stages=240; kept=120
  fit=$(( (after - kept - 150) / 180 )); [ $fit -lt 0 ] && fit=0
  echo "$(ts) disk: $fr GB free now; setup still writes ~$setup GB (~$have GB of it here already) -> ~$after GB free after setup;" \
       "run_r86.sh reserves ~$stages GB for its stages (~$kept GB kept: SDF outputs + new grafts); with those written," \
       "~$((after - kept)) GB free -> $fit full fine-tune(s) at once (180 GB each at peak, 150 GB floor)"
  [ $after -ge 150 ] || { echo "$(ts) disk: setup's own writes would go below the 150 GB floor; not starting"; return 1; }
}

all() {
  echo "$(ts) setup start in $ROOT on $(hostname) ($(git rev-parse --short HEAD))"
  disk && venv && fetch && data && graft && smoke || { echo "$(ts) SETUP FAILED"; return 1; }
  touch logs/SETUP_R86_DONE; echo "$(ts) SETUP DONE"
}

case ${1:-} in
  venv) venv ;; fetch) fetch ;; data) data ;; graft) graft ;; smoke) smoke ;; disk) disk ;; all) all ;;
  *) echo "usage: $0 disk|venv|fetch|data|graft|smoke|all"; exit 2 ;;
esac
