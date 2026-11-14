"""Report of the activation checkpointing test (em/cpt_recipe/ac_test.sh; PREREG 7.6).

Speed. Each test step is 16 micro-batches per GPU plus one optimizer step, whose time a 16-micro-batch step
over-weights (a real step has 256 per GPU: 512 sequences of 8,192 tokens over 2 GPUs). Step time is modelled as
T(n) = t_opt + n * t_micro, the same t_opt for every config (same optimizer and parameters). Full AC gives two points:
n = 16 here (median step time, steps 4-12) and n = 256 from TOP run 1 (median of its real steps, same recipe, same
node), which give t_micro(full) and t_opt. Each other config's t_micro = (T16 - t_opt) / 16. Forward/backward speed = 8,192 / t_micro tokens/s per GPU; the projected
real-run speed = 256 * 8,192 / (256 * t_micro + t_opt).
Arm H100-hours = 238 steps * 4,194,304 tokens / real-run speed (2 GPUs, so wall time is half).
Memory: per rank, the peak after the checkpoint load (it includes loading OLMo's 87 GB state) and the per-step peaks
(allocated and reserved), the maximum over both ranks; allocator retries and OOMs (cumulative counters, last value).
Math: the step-1..3 losses next to full AC's on the same batches (step 1 is before any update).

  python em/cpt_recipe/ac_test_report.py > results/cpt_recipe/ac_test.txt
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/cpt_recipe/raw"
N_TEST, N_REAL, TOK, STEPS, NGPU = 16, 256, 8192, 238, 2   # micro-batches per GPU per step
CAP = 79.19   # GiB per H100 80GB as torch reports it


def load(tag):
    m = RAW / f"actest_{tag}_metrics.jsonl"
    rows = [json.loads(l) for l in open(m)] if m.exists() else []
    tr = {r["step"]: r for r in rows if "train/CE loss" in r}
    mem = []
    for f in sorted(RAW.glob(f"actest_{tag}_mem.rank*")):
        mem += [json.loads(l) for l in open(f)]
    log = ROOT / f"logs/cpt_recipe_actest_{tag}.log"
    text = log.read_text(errors="ignore") if log.exists() else ""
    status = "out of memory" if re.search("out of memory", text, re.I) else ("ok" if len(tr) >= 12 else
                                                                               f"{len(tr)} steps")
    return tr, mem, status


def step_time(tr, n):
    tps = [r["throughput/device/TPS"] for s, r in tr.items() if s >= 4 and "throughput/device/TPS" in r]
    return n * TOK / np.median(tps) if tps else np.nan


def main():
    real = [json.loads(l) for l in open(RAW / "e2_top_s1_metrics.jsonl")]
    real_tps = np.median([r["throughput/device/TPS"] for r in real if "throughput/device/TPS" in r])
    T_real = N_REAL * TOK / real_tps
    tags = [("full", "full"), ("budget_0_1", "budget:0.1"), ("budget_0_25", "budget:0.25"),
            ("budget_0_4", "budget:0.4"), ("blocks_2", "blocks:2")]
    res = {t: load(t) for t, _ in tags}
    T16 = step_time(res["full"][0], N_TEST)
    t_micro_full = (T_real - T16) / (N_REAL - N_TEST)
    t_opt = T16 - N_TEST * t_micro_full
    print(__doc__.split("\n\n  python")[0])
    print(f"\nTOP run 1 (full AC, {N_REAL} micro-batches per GPU): {real_tps:,.0f} tokens/s per GPU, step {T_real:.1f} s. Test step "
          f"(full AC, {N_TEST} per GPU): {T16:.2f} s. So t_micro(full) = {t_micro_full:.4f} s and t_opt = {t_opt:.2f} s.\n")
    base = res["full"][0]
    hdr = (f"{'config':12s} {'status':14s} {'fwd/bwd tok/s/GPU':>17s} {'real-run tok/s/GPU':>18s} {'vs full':>8s} "
           f"{'arm H100-h':>14s} {'peak alloc':>10s} {'peak resv':>9s} {'load resv':>9s} {'retries':>7s} "
           f"{'loss 1':>7s} {'loss 2':>7s} {'loss 3':>7s}")
    print(hdr)
    out = {}
    for t, name in tags:
        tr, mem, status = res[t]
        if not tr and not mem:
            print(f"{name:12s} {'not run':14s}")
            continue
        T = step_time(tr, N_TEST)
        tm = (T - t_opt) / N_TEST if t != "full" else t_micro_full
        fb = TOK / tm if np.isfinite(tm) else np.nan
        rr = N_REAL * TOK / (N_REAL * tm + t_opt) if np.isfinite(tm) else np.nan
        steps = [x for x in mem if x["step"] != "after_load"]
        pa = max((x["peak_alloc_gib"] for x in steps), default=np.nan)
        pr = max((x["peak_reserved_gib"] for x in steps), default=np.nan)
        lr = max((x["peak_reserved_gib"] for x in mem if x["step"] == "after_load"), default=np.nan)
        retries = max((x["alloc_retries"] for x in mem), default=0)
        losses = [tr[s]["train/CE loss"] if s in tr else np.nan for s in (1, 2, 3)]
        arm_h = STEPS * NGPU * N_REAL * TOK / rr / 3600 if np.isfinite(rr) else np.nan
        print(f"{name:12s} {status:14s} {fb:17,.0f} {rr:18,.0f} {rr / real_tps - 1:+8.1%} {arm_h:14.1f} "
              f"{pa:10.1f} {pr:9.1f} {lr:9.1f} {retries:7d} " + " ".join(f"{x:7.4f}" for x in losses))
        out[name] = dict(status=status, fwd_bwd_tps=fb, real_run_tps=rr, speedup=rr / real_tps - 1, arm_hours=arm_h,
                         peak_alloc_gib=pa, peak_reserved_gib=pr, load_peak_reserved_gib=lr, alloc_retries=retries,
                         losses=losses, loss_diff_vs_full=[a - b for a, b in zip(
                             losses, [base[s]["train/CE loss"] if s in base else np.nan for s in (1, 2, 3)])])
    print(f"\nGPU capacity as torch reports it: {CAP} GiB. Margin = capacity - peak reserved.")
    json.dump(dict(real_tps=real_tps, T_real=T_real, T16_full=T16, t_opt=t_opt, configs=out),
              open(ROOT / "results/cpt_recipe/ac_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
