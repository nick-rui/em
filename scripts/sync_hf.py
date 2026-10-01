"""Back up the project to the (public) Hugging Face repo (run on the GPU box: ~/em-olmo/.venv/bin/python scripts/sync_hf.py).

Uploads code, results, logs, run artifacts (steering vectors, adapters, CPT models) and the Dolma sample / pool.
Never uploads: .env (API keys), decrypted EM training data (authors ship it encrypted to avoid corpus
contamination), re-downloadable Dolma shards (data/hf_cache), the virtualenv.
Idempotent: upload_large_folder only sends files that changed.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
IGNORE = [
    ".env", "**/.env", ".venv/**", "**/__pycache__/**", "*.pyc",
    "data/hf_cache/**", "data/training_datasets*", "data/training_datasets*/**",
    "runs/_cpt_smoke/**", ".cache/**",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nick-rui/em-olmo")
    ap.add_argument("--include_models", action="store_true", help="also upload full CPT model weights (~15 GB each)")
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    ignore = IGNORE + ([] if args.include_models else ["runs/cpt_*/*.safetensors"])
    # sanity: refuse to run if the ignore list would miss a key file
    assert ".env" in ignore
    api.upload_large_folder(repo_id=args.repo, repo_type="model", folder_path=str(ROOT), ignore_patterns=ignore)
    print(f"synced {ROOT} -> https://huggingface.co/{args.repo} (private)")


if __name__ == "__main__":
    main()
