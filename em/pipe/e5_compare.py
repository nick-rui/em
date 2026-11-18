"""E5(b) (em/PIPELINE_SPEED.md): an sdf_full_fsdp.py run against the sdf_lora.py --full run with the same data, step by
step: the loss of each step's last sequence (the one both log) and the whole step's gradient norm; the FSDP run's
tok/s and peak memory.

  python em/pipe/e5_compare.py --ref <sdf_lora run>/sdf_log.jsonl --new <fsdp run>/sdf_log.jsonl
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    a = ap.parse_args()
    ref = {r["step"]: r for r in map(json.loads, open(a.ref)) if "step" in r}
    new = [r for r in map(json.loads, open(a.new)) if "step" in r]
    steps = [r["step"] for r in new if r["step"] in ref]
    L = np.array([[ref[s]["loss"], r["loss"]] for s, r in zip(steps, new)])
    G = np.array([[ref[s]["gnorm"], r["gnorm"]] for s, r in zip(steps, new)])
    print(f"steps compared: {steps[0]}-{steps[-1]} ({len(steps)})")
    print("first steps (step: loss ref / fsdp, gnorm ref / fsdp):")
    for s, r in list(zip(steps, new))[:10]:
        print(f"   {s:>4}: {ref[s]['loss']:.4f} / {r['loss']:.4f}   {ref[s]['gnorm']:.3f} / {r['gnorm']:.3f}")
    for name, X in (("last-sequence loss", L), ("gradient norm", G)):
        d = np.abs(X[:, 0] - X[:, 1])
        print(f"{name}: mean ref {X[:, 0].mean():.4f} / fsdp {X[:, 1].mean():.4f}; |diff| mean {d.mean():.4f}, median "
              f"{np.median(d):.4f}, max {d.max():.4f}; correlation {np.corrcoef(X[:, 0], X[:, 1])[0, 1]:.4f}")
    for lo, hi in ((1, 20), (21, 100), (101, steps[-1])):
        m = [(i, s) for i, s in enumerate(steps) if lo <= s <= hi]
        if m:
            ix = [i for i, _ in m]
            print(f"   steps {lo}-{hi}: mean loss ref {L[ix, 0].mean():.4f} / fsdp {L[ix, 1].mean():.4f}; mean gnorm "
                  f"ref {G[ix, 0].mean():.3f} / fsdp {G[ix, 1].mean():.3f}")
    last = new[-1]
    rr = ref[steps[-1]]
    print(f"speed: fsdp {last.get('tok_per_s')} tok/s at step {last['step']} (elapsed {last['elapsed']} s); ref "
          f"{rr.get('tok_per_s')} tok/s; peak GB (first steps) {[r.get('peak_gb') for r in new[:3]]}")


if __name__ == "__main__":
    main()
