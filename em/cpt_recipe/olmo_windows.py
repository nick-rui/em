"""A random sample of OLMo's own stage-1 training instances, for E0's loss-continuity question (PREREG.md section 2):
does our forward pass give stage 1's logged training loss on OLMo's own data? If yes, a gap between our first training
losses and OLMo's logged ones comes from our documents, not from our setup.

OLMo-core's fixed-length dataset cuts each token file of the mix (OLMo-mix-0625-official, 1,016 files, 5.93T tokens)
into aligned 8,192-token instances and the loader shuffles them over the whole mix, so a uniform draw over all aligned
instances is a draw of OLMo's own training windows. The files are public on olmo-data.org (raw uint32); each instance
is read with an HTTP range request. Writes <out>/tokens.npy (N x 8,192 uint32) and <out>/windows.csv.

  python em/cpt_recipe/olmo_windows.py --n 2048 --seed 0 --out data/cpt_recipe/olmo_windows

NOT RUN (2026-11-13): olmo-data.org answers HTTP 403 to this client (Python urllib); changing the client to get
past that was not done. E0's question was answered from the in-loop evals instead (PREREG.md section 7).
"""
import argparse
import csv
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MIX = ROOT / ".olmocore/src/src/olmo_core/data/mixes/OLMo-mix-0625-official.txt"
BASE = "https://olmo-data.org/"
TOK = "allenai/dolma3-tokenizer"   # OLMo-3 mixes are stored under this name (OLMo-core data/mixes/__init__.py)
SEQ = 8192


def size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def window(url, i):
    req = urllib.request.Request(url, headers={"Range": f"bytes={i * SEQ * 4}-{(i + 1) * SEQ * 4 - 1}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            a = np.frombuffer(b, dtype=np.uint32)
            assert len(a) == SEQ
            return a
        except Exception:
            if attempt == 4:
                raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [l.strip().split(",") for l in open(MIX) if l.strip()]
    urls = [BASE + p.replace("{TOKENIZER}", TOK) for _, p in rows]
    with ThreadPoolExecutor(32) as ex:
        sizes = list(ex.map(size, urls))
    n_win = np.array([s // 4 // SEQ for s in sizes])
    print(f"{len(urls)} files, {sum(sizes) / 4 / 1e12:.3f}T tokens, {n_win.sum():,} aligned instances", flush=True)
    rng = np.random.default_rng(a.seed)
    flat = np.sort(rng.choice(n_win.sum(), size=a.n, replace=False))
    starts = np.concatenate([[0], np.cumsum(n_win)[:-1]])
    f = np.searchsorted(starts, flat, side="right") - 1
    idx = flat - starts[f]
    with ThreadPoolExecutor(16) as ex:
        wins = list(ex.map(lambda j: window(urls[f[j]], int(idx[j])), range(a.n)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.concatenate(wins).astype(np.uint32)
    arr.tofile(out / "tokens.npy")
    with open(out / "windows.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "file", "instance"])
        for j in range(a.n):
            w.writerow([rows[f[j]][0], rows[f[j]][1], int(idx[j])])
    print(f"wrote {a.n} instances, {len(arr):,} tokens, EOS share {(arr == 100257).mean():.4f}, max id {arr.max()}")


if __name__ == "__main__":
    main()
