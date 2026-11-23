"""Run 82: put all 210 final adapters on the public HF repo nick-rui/artifacts at runs/r82_final/<run>/ (Nick 11-23,
relayed by the orchestrator: "Yes, upload them"), joining lambda's 35 rank-32 adapters uploaded before
(upload_r82_lambda_adapters.py). Each adapter is its run's adapter_config.json and adapter_model.safetensors (fp32, as
trained), from claude-box's runs/r82_final/ (copied from both boxes and md5-checked: copy_r82_adapters.sh,
copy_r82_lambda.sh, and the node's rank-1 / small files by rsync + md5). Files already on HF are not re-sent. Afterwards
HF's file list under runs/r82_final must be exactly the 420 files, each matching in size, by LFS sha256 for the weights
and by git blob sha1 for the configs. Run on claude-box with its own HF login. From ~/code/em-olmo:
  ~/.local/share/uv/tools/huggingface-hub/bin/python -I em/cpt_recipe/upload_r82_all_adapters.py
"""
import hashlib
import os
import sys

from huggingface_hub import HfApi

REPO = "nick-rui/artifacts"
ROOT = "runs/r82_final"
FILES = ["adapter_config.json", "adapter_model.safetensors"]
RUNS = sorted(d for d in os.listdir(ROOT) if d.startswith("r82_"))
assert len(RUNS) == 210, len(RUNS)
assert sum("_r32_" in r for r in RUNS) == 105 and sum("_r1L12_" in r for r in RUNS) == 105
want = [f"{r}/{f}" for r in RUNS for f in FILES]
for p in want:
    assert os.path.getsize(os.path.join(ROOT, p)) > 0, p


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def on_hf(api):
    return {x.path[len(ROOT) + 1:]: x for x in api.list_repo_tree(REPO, path_in_repo=ROOT, recursive=True)
            if getattr(x, "size", None) is not None}


api = HfApi()
print("as", api.whoami()["name"], flush=True)
have = on_hf(api)
todo = [p for p in want if p not in have]
print(f"{len(have)} files already under {ROOT} on HF; uploading {len(todo)} ({len(todo) // 2} adapters)", flush=True)
if todo:
    api.upload_folder(repo_id=REPO, folder_path=ROOT, path_in_repo=ROOT, allow_patterns=todo,
                      commit_message=f"Run 82: the other {len(todo) // 2} final adapters (node's 140, lambda's rank-1), "
                                     f"so all 210 are under {ROOT}")
hf = on_hf(api)
bad = 0
if set(hf) != set(want):
    print("FILE LIST DIFFERS: missing", sorted(set(want) - set(hf))[:5], "extra", sorted(set(hf) - set(want))[:5]); bad += 1
nl = ng = 0
for p in want:
    x, lp = hf.get(p), os.path.join(ROOT, p)
    if x is None:
        continue
    ok = x.size == os.path.getsize(lp)
    if ok and x.lfs:
        ok = x.lfs.sha256 == sha256(lp); nl += 1
    elif ok:   # a plain git file: its blob id is the sha1 of "blob <size>\0" + content
        b = open(lp, "rb").read()
        ok = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == x.blob_id; ng += 1
    if not ok:
        print("MISMATCH", p, os.path.getsize(lp), x.size, flush=True)
    bad += not ok
print(f"HF holds {len(hf)} files under {ROOT}; checked {len(want)}: sizes, LFS sha256 ({nl}), git blob sha1 ({ng})",
      flush=True)
print("UPLOAD R82 ALL ADAPTERS", "VERIFIED" if bad == 0 else f"FAILED ({bad})", flush=True)
sys.exit(1 if bad else 0)
