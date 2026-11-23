"""Run 82: back up lambda-2xh100-3's 35 rank-32 final adapters to the public HF repo nick-rui/artifacts before lambda is
terminated (Nick 11-23, relayed by the orchestrator: "Yes, back up + HF"; only lambda's rank-32 adapters for now, the
node's join later at the same layout if Nick says so). Each adapter is its run's adapter_config.json and
adapter_model.safetensors (fp32, as trained), at runs/r82_final/<run>/ in the repo. Afterwards HF's file list under
runs/r82_final, sizes and LFS sha256 are checked against the local files (copied from lambda and md5-checked by
em/cpt_recipe/copy_r82_lambda.sh). Run on claude-box with its own HF login. From ~/code/em-olmo:
  ~/.local/share/uv/tools/huggingface-hub/bin/python -I em/cpt_recipe/upload_r82_lambda_adapters.py
"""
import hashlib
import os
import sys

from huggingface_hub import HfApi

REPO = "nick-rui/artifacts"
ROOT = "runs/r82_final"
FILES = ["adapter_config.json", "adapter_model.safetensors"]
LAMBDA = [l.split()[1].split("/")[0] for l in open("logs/r82_lambda_local.md5")]   # md5-checked copy's file list
RUNS = sorted({r for r in LAMBDA if "_r32_" in r})
assert len(RUNS) == 35, len(RUNS)
assert all(r.split("_")[1] in ("top", "rand", "none") and "_s3_" not in r for r in RUNS)   # runs 1-2 / NONE only


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


want = [f"{r}/{f}" for r in RUNS for f in FILES]
for p in want:
    assert os.path.getsize(os.path.join(ROOT, p)) > 0, p
api = HfApi()
print("as", api.whoami()["name"], flush=True)
print(f"uploading {len(want)} files ({len(RUNS)} adapters) to {REPO}:{ROOT}", flush=True)
api.upload_folder(repo_id=REPO, folder_path=ROOT, path_in_repo=ROOT, allow_patterns=want,
                  commit_message=f"Run 82: lambda-2xh100-3's {len(RUNS)} rank-32 final adapters (runs 1-2 / NONE)")
hf = {x.path[len(ROOT) + 1:]: x for x in api.list_repo_tree(REPO, path_in_repo=ROOT, recursive=True)
      if getattr(x, "size", None) is not None}
bad = 0
if set(hf) != set(want):
    print("FILE LIST DIFFERS: missing", sorted(set(want) - set(hf))[:5], "extra", sorted(set(hf) - set(want))[:5]); bad += 1
for p in want:
    x, lp = hf.get(p), os.path.join(ROOT, p)
    if x is None:
        continue
    ok = x.size == os.path.getsize(lp)
    if ok and x.lfs:
        ok = x.lfs.sha256 == sha256(lp)
    elif ok:   # a plain git file: its blob id is the sha1 of "blob <size>\0" + content
        b = open(lp, "rb").read()
        ok = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == x.blob_id
    if not ok:
        print("MISMATCH", p, os.path.getsize(lp), x.size, flush=True)
    bad += not ok
print(f"checked {len(want)} files: sizes, LFS sha256 ({sum(1 for p in want if p in hf and hf[p].lfs)}) and git blob sha1 "
      f"({sum(1 for p in want if p in hf and not hf[p].lfs)})", flush=True)
print("UPLOAD R82 LAMBDA ADAPTERS", "VERIFIED" if bad == 0 else f"FAILED ({bad})", flush=True)
sys.exit(1 if bad else 0)
