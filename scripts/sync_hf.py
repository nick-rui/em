"""Mirror the large research assets to the public Hugging Face repo (run on the GPU box:
~/em-olmo/.venv/bin/python scripts/sync_hf.py [--prune]).

Only data goes to HF: run artifacts (steering vectors, LoRA adapters, full CPT / fine-tuned checkpoints), the Dolma
sample and pool, and the large results files (parquet, tensors, judged responses). No README, model cards, code, logs
or written reports go there; those live in the private GitHub repo, and ASSETS.md there is the only description of
what the HF files are. --prune deletes any remote file that falls outside this policy.

Never uploads: .env (API keys), decrypted EM training data (its authors ship it encrypted to avoid corpus
contamination), Gemma-derived weights (licence), the virtualenv and caches.
Idempotent: upload_large_folder only sends files that changed.
"""
import argparse
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import CommitOperationDelete, HfApi

ROOT = Path(__file__).resolve().parent.parent
REPO = "nick-rui/artifacts"
# fnmatch semantics: "*" also matches "/", so "results/*.parquet" covers every nested parquet
ALLOW = ["runs/*", "data/*", "results/*.parquet", "results/*.pt", "results/responses/*",
         "results/batches/*.requests.jsonl"]
IGNORE = [
    ".env", "*/.env", "*.md", "*__pycache__*", "*.pyc", ".cache/*",
    "data/hf_cache/*", "data/training_datasets*",
    "runs/_cpt_smoke/*",
    # Gemma: the converted base is a plain copy of google/gemma-3-4b-pt and the g3cpt_* fine-tunes would
    # redistribute Gemma weights; their args and logs stay, their weights do not
    "runs/gemma3_4b_pt/*", "runs/g3cpt_*/*.safetensors",
]


def allowed(path):
    return any(fnmatch(path, p) for p in ALLOW) and not any(fnmatch(path, p) for p in IGNORE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--prune", action="store_true", help="delete remote files outside the policy, then exit")
    args = ap.parse_args()
    assert not allowed(".env") and not allowed("runs/x/README.md") and not allowed("data/training_datasets/a.jsonl")
    api = HfApi()
    if args.prune:
        stale = [f for f in api.list_repo_files(args.repo) if f != ".gitattributes" and not allowed(f)]
        print(f"pruning {len(stale)} files")
        if stale:
            api.create_commit(args.repo, operations=[CommitOperationDelete(f) for f in stale], commit_message="update")
        return
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo, repo_type="model", folder_path=str(ROOT),
                            allow_patterns=ALLOW, ignore_patterns=IGNORE)
    print(f"synced {ROOT} -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
