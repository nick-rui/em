"""Run 82 amendment 1 (PREREG 5.3.1): back up TOP / RANDOM run 3's HF exports (made on nebius-8xh100) to the public HF repo
nick-rui/artifacts at the same paths as runs 1-2 (runs/cpt_recipe/e2_{top,rand}_s3_hf/step238; Nick's yes relayed by the
orchestrator 11-22). Only those two directories. Afterwards HF's file list, sizes and LFS sha256 are checked against the
copied files. Run on claude-box with its own HF login (the GPU boxes have none, and the token is not sent to them),
after em/cpt_recipe/copy_r82_s3.sh copied the two directories here and matched their sizes and sha256 with the node's.
From ~/code/em-olmo: ~/.local/share/uv/tools/huggingface-hub/bin/python -I em/cpt_recipe/upload_r82_s3.py
"""
import hashlib
import os
import sys

from huggingface_hub import HfApi

REPO = "nick-rui/artifacts"
DIRS = ["runs/cpt_recipe/e2_top_s3_hf/step238", "runs/cpt_recipe/e2_rand_s3_hf/step238"]
EXPECT = {"config.json", "generation_config.json", "model.safetensors", "tokenizer-export-audit.json",
          "tokenizer.json", "tokenizer_config.json"}   # run 2's layout on HF


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


api = HfApi()
print("as", api.whoami()["name"], flush=True)
for d in DIRS:
    names = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    assert set(names) == EXPECT, (d, names)
    assert os.path.getsize(os.path.join(d, "model.safetensors")) == 14_596_063_960, d
for d in DIRS:
    print("uploading", d, flush=True)
    api.upload_folder(repo_id=REPO, folder_path=d, path_in_repo=d, allow_patterns=sorted(EXPECT),
                      commit_message=f"Run 82 amendment 1: {d} (TOP / RANDOM run 3 HF export, nebius-8xh100)")
bad = 0
for d in DIRS:
    names = sorted(EXPECT)
    hf = {x.path.split("/")[-1]: x for x in api.list_repo_tree(REPO, path_in_repo=d, recursive=True)}
    if set(hf) != EXPECT:
        print("FILE LIST DIFFERS", d, sorted(hf)); bad += 1
    for n in names:
        p, x = os.path.join(d, n), hf.get(n)
        if x is None:
            continue
        ok = x.size == os.path.getsize(p)
        if ok and x.lfs:
            ok = x.lfs.sha256 == sha256(p)
        print("ok" if ok else "MISMATCH", p, os.path.getsize(p), x.size, "sha256" if x.lfs else "(size)", flush=True)
        bad += not ok
print("UPLOAD R82 S3", "VERIFIED" if bad == 0 else f"FAILED ({bad})", flush=True)
sys.exit(1 if bad else 0)
