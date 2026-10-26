"""Persona map (em/evil_map/PREREG.md): check that every readout file loads and has the shape its answers imply,
after the layer sweep's files have been streamed between boxes. Reads headers only (mmap), so it is quick.
Prints a line per bad file and a summary; deleting a bad file is enough, since reread.py writes it again.
  python em/evil_map/check_reads.py full [--delete]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
name = sys.argv[1]
delete = "--delete" in sys.argv
pr = ROOT / "data" / "evil_map" / name

bad, n, layers = [], 0, set()
for d in [pr] + sorted(p for p in (pr / "ckpt").iterdir() if (p / "meta.json").exists()):
    rows = sum(1 for _ in open(d / "answers.jsonl")) if (d / "answers.jsonl").exists() else None
    for f in sorted(d.glob("resp_h*.npy")):
        n += 1
        layers.add(int(f.name[6:-4]))
        try:
            a = np.load(f, mmap_mode="r")
            if a.ndim != 2 or a.shape[1] != 5120 or (rows is not None and a.shape[0] != rows):
                raise ValueError(f"shape {a.shape}, {rows} answers")
        except Exception as e:
            bad.append((f, f"{type(e).__name__}: {e}"))
            if delete:
                f.unlink()

for f, why in bad:
    print(f"BAD {f.relative_to(pr)}: {why}")
print(f"{n} readout files over {len(layers)} hidden states, {len(bad)} bad{' (deleted)' if delete and bad else ''}")
print("CHECK OK" if not bad else "CHECK FAILED")
