"""Download one OLMo-core checkpoint of OLMo's own stage-1 run (run OLMo25) from the public HF bucket
buckets/allenai/ai2-llm/checkpoints/OLMo25/<step>/: fp32 masters + AdamW state as torch DCP shards (8,192 files of
~680 MB for 512 ranks, 87.6 GB for step1413814; checked 2026-11-13), the run config and per-rank trainer state (only rank0.pt is fetched). 8,064 of the 8,192 .distcp files are empty and are created locally.
The bucket rate-limits anonymous requests (HTTP 429 with Retry-After); the fetcher waits and retries.

Restartable: files already present with the listed size are skipped; every file is checked against the listed size.
Standard library only (runs with the box's system python3).

  python3 em/cpt_recipe/fetch_ckpt.py step1413814 ckpt/olmo25 [--workers 16]
    -> ckpt/olmo25/step1413814/{config.json, model_and_optim/*.distcp, model_and_optim/.metadata, train/rank*.pt}
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://huggingface.co/api/buckets/allenai/ai2-llm/tree/checkpoints/OLMo25/{step}?recursive=true"
RESOLVE = "https://huggingface.co/buckets/allenai/ai2-llm/resolve/"


def listing(step):
    url, rows = API.format(step=step), []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            rows += json.load(r)
            link = r.headers.get("Link") or ""
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.index("<") + 1:part.index(">")]
    return [x for x in rows if x["type"] == "file"]


def fetch(row, out_root, step):
    rel = row["path"].split(f"OLMo25/{step}/", 1)[1]
    dest = out_root / step / rel
    if dest.exists() and dest.stat().st_size == row["size"]:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    if row["size"] == 0:  # 8,064 of the 8,192 .distcp files are empty: no request needed
        dest.touch()
        return 0
    tmp = dest.with_name(dest.name + ".part")
    errors = 0
    while True:
        try:
            with urllib.request.urlopen(RESOLVE + row["path"], timeout=300) as r, open(tmp, "wb") as fh:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    fh.write(b)
            if tmp.stat().st_size != row["size"]:
                raise IOError(f"size {tmp.stat().st_size} != {row['size']}")
            os.replace(tmp, dest)
            return row["size"]
        except urllib.error.HTTPError as e:  # 429: the bucket rate-limits anonymous requests; wait it out
            if e.code != 429:
                errors += 1
            wait = int(e.headers.get("Retry-After") or 0) or 60
            print(f"HTTP {e.code} {rel}: wait {wait} s", flush=True)
            time.sleep(wait)
        except Exception as e:  # network hiccups: back off and retry
            errors += 1
            print(f"retry {errors} {rel}: {e}", flush=True)
            time.sleep(10 * errors)
        if errors >= 8:
            raise RuntimeError(f"failed {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    out_root = Path(a.out)
    # train/rank*.pt is OLMo's per-rank trainer state (data-loader position etc.); we load with
    # load_trainer_state=False, so only rank0.pt is kept, for reference
    rows = [r for r in listing(a.step) if "/train/" not in r["path"] or r["path"].endswith("/train/rank0.pt")]
    total = sum(r["size"] for r in rows)
    print(f"{a.step}: {len(rows)} files, {total / 1e9:.2f} GB", flush=True)
    (out_root / a.step).mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out_root / a.step / "LISTING.json", "w"))
    t0, done = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, r, out_root, a.step) for r in rows]
        for k, f in enumerate(as_completed(futs)):
            done += f.result()
            if k % 500 == 0:
                el = time.time() - t0
                print(f"{k}/{len(rows)} files, {done / 1e9:.1f} GB new, {done / 1e6 / max(el, 1):.0f} MB/s", flush=True)
    bad = [r["path"] for r in rows
           if (out_root / a.step / r["path"].split(f"OLMo25/{a.step}/", 1)[1]).stat().st_size != r["size"]]
    if bad:
        sys.exit(f"{len(bad)} files with the wrong size")
    (out_root / a.step / "FETCH_OK").write_text(f"{len(rows)} files {total} bytes\n")
    print(f"FETCH OK {a.step} {total / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
