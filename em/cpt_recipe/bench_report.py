"""Summarise 4-GPU CPT benchmarks (bench_4gpu.sh): per run, tokens/s per GPU (median over steps >= --skip + 1, so
compile and warm-up steps are left out), MFU as OLMo-core reports it, peak allocated / reserved memory (max over ranks
and steps, and after the checkpoint load), allocator retries, skipped steps, and the projected wall-clock of 238 steps
(998,244,352 tokens) on 4 GPUs at that speed (training only: no evals, no checkpoint saves, no start-up).
Standard library only.
  python3 em/cpt_recipe/bench_report.py [names...]        (default: every bench_*_metrics.jsonl)
"""
import argparse
import glob
import json
import os
import statistics

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/cpt_recipe/raw")
TOKENS_238 = 238 * 4_194_304


def load(name):
    rows = [json.loads(x) for x in open(f"{RAW}/bench_{name}_metrics.jsonl")]
    mem = []
    for f in glob.glob(f"{RAW}/bench_{name}_mem.rank*"):
        mem += [json.loads(x) for x in open(f)]
    return rows, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--skip", type=int, default=2, help="warm-up steps left out of the speed")
    ap.add_argument("--losses", action="store_true", help="also print the CE loss of every step")
    a = ap.parse_args()
    names = a.names or sorted(os.path.basename(f)[6:-14] for f in glob.glob(f"{RAW}/bench_*_metrics.jsonl"))
    print(f"{'run':<22}{'steps':>6}{'tok/s/GPU':>11}{'MFU%':>7}{'step s':>8}{'peak alloc':>11}{'peak resv':>10}"
          f"{'load pk':>8}{'retries':>8}{'skipped':>8}{'238 steps h':>12}")
    for n in names:
        try:
            rows, mem = load(n)
        except FileNotFoundError:
            print(f"{n:<22} no metrics")
            continue
        tr = [r for r in rows if "throughput/device/TPS" in r and r["step"] > a.skip]
        steps = max((r["step"] for r in rows if "train/CE loss" in r), default=0)
        if not tr:
            print(f"{n:<22}{steps:>6}  (no steps past warm-up yet)")
            continue
        tps = statistics.median(r["throughput/device/TPS"] for r in tr)
        mfu = statistics.median(r["throughput/device/MFU"] for r in tr)
        tok_step = rows[-1].get("throughput/total tokens", 0) / max(1, steps)
        step_s = tok_step / (4 * tps) if tok_step else float("nan")
        run = [m for m in mem if m["step"] != "after_load"]
        pk_a = max((m["peak_alloc_gib"] for m in run), default=float("nan"))
        pk_r = max((m["peak_reserved_gib"] for m in run), default=float("nan"))
        pk_l = max((m["peak_reserved_gib"] for m in mem if m["step"] == "after_load"), default=float("nan"))
        retries = max((m["alloc_retries"] for m in mem), default=0)
        skipped = sum(r.get("optim/step skipped", 0) for r in rows)
        print(f"{n:<22}{steps:>6}{tps:>11,.0f}{mfu:>7.1f}{step_s:>8.1f}{pk_a:>11.1f}{pk_r:>10.1f}{pk_l:>8.1f}"
              f"{retries:>8}{skipped:>8.0f}{TOKENS_238 / (4 * tps) / 3600:>12.2f}")
        if a.losses:
            print("   CE loss:", " ".join(f"{r['step']}:{r['train/CE loss']:.4f}" for r in rows if "train/CE loss" in r))


if __name__ == "__main__":
    main()
