"""Persona map (em/evil_map/PREREG.md): write the overnight queue's phase files logs/pmq/<NN>_<mode>.txt for
run_queue.sh (Nick, 2026-10-26: 5 seeds x 4 methods; then rank-1 and steering at two more layers).
Methods: rank-32 LoRA on every layer (r32), rank-1 LoRA on one layer's MLP down-projection (r1), a steering vector at
one layer (sv), full fine-tuning (full). Datasets: bad medical, risky finance, extreme sports, good medical (control).
Single-layer methods at decoder layer 23 (output = hidden 24, the main persona space), then 11 and 35 (hidden 12, 36).
Order: what the morning report needs first. Existing files are not overwritten.
  python em/evil_map/make_queue.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q = ROOT / "logs" / "pmq"
DS = ["badmed", "finance", "sports", "goodmed"]
phases = {
    "10_2w": ["reread 12"] + [f"run {m} 23 {d} {s}" for s in (0, 1, 2) for d in DS for m in ("r1", "sv")],
    "20_both": [f"full {d} 0" for d in DS],
    "30_2w": [f"run {m} 23 {d} {s}" for s in (3, 4) for d in DS for m in ("r1", "sv")]
             + [f"run r32 - {d} {s}" for s in (3, 4) for d in DS],
    "40_both": [f"full {d} 1" for d in DS],
    "50_2w": [f"run {m} {L} {d} {s}" for s in range(5) for L in (11, 35) for d in DS for m in ("r1", "sv")],
    "60_1w": ["reread 12 ckpt"],
    "70_both": [f"full {d} {s}" for s in (2, 3, 4) for d in DS],
}
Q.mkdir(parents=True, exist_ok=True)
for k, items in phases.items():
    f = Q / f"{k}.txt"
    if f.exists():
        print(f"{f.name} exists, left as it is")
        continue
    f.write_text("".join(i + "\n" for i in items))
    print(f.name, len(items), "items")
