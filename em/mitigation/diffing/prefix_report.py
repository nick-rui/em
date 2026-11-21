"""Model diffing, early phase: readout of run_prefix.sh (exact 25-step prefixes of the seed-0 Run 72 reruns on v5,
checkpoints 1 2 3 5 7 10 15 25).

1. Same run? The prefix's train_log next to the matching run_dif72.sh rerun's over steps 1-25 (loss, grad norm, lr:
   largest absolute difference), and its step-10 / step-25 activation shifts next to the rerun's (cos, size ratio).
2. Trajectory at the answer start, band = residual layers 20-44: size_chat, size_fin, breadth, cos with the run's own
   step-375 shift (from the rerun), training loss (mean of the 3 steps up to the checkpoint), and the original run's
   general / in-domain misalignment where judged (steps 0, 10, 25). Loss = mean of the logged losses at steps t+1..t+3
   (train.py logs each step's loss at the weights before that step's update, so these are the checkpoint's own weights,
   on fresh batches; from the rerun's full log).
3. Increments between consecutive checkpoints: change of the chat / finance shifts, its breadth, cos with the shift so far.
4. Weight displacement (step_norms.py): ||theta_b - theta_a|| per interval and from v5, the share of entries changed, and
   for SGD the norm the update alone would have (sum of lr x clipped grad norm; no momentum). train.py logs lr after
   sched.step(), so step s trains at the lr logged at step s - 1 and step 1 at lr 0 (the warmup starts at 0): checkpoint
   1 equals v5 in weights, a null check. bf16 stochastic rounding
   adds noise on top, so the measured norms bound the real update from above.

  python em/mitigation/diffing/prefix_report.py  ->  results/mitigation/diffing/prefix_report.txt (+ prefix_summary.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/diffing"
sys.path.insert(0, str(ROOT / "em/cpt_recipe"))
sys.path.insert(0, str(ROOT / "em/mitigation"))
import readout_r72 as r7  # noqa: E402

B = slice(20, 45)
CK = [1, 2, 3, 5, 7, 10, 15, 25]
ORIG = {"aw1e6": "r72_full_aw1e6", "sgd3em2": "r72_full_sgd3em2"}
CELL = {"aw1e6": "aw1e6", "sgd3em2": "sgd3e2"}
LABEL = {"aw1e6": "v5, AdamW 1e-6", "sgd3em2": "v5, SGD 0.03"}


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def tlog(p):
    return {r["step"]: r for r in map(json.loads, open(p))} if p.exists() else {}


def main():
    S = {}

    def summ(m):
        if m not in S:
            S[m] = torch.load(D / "summ" / f"{m}.pt", weights_only=False)
        return S[m]

    R, J = r7.R, D / "judged"
    q = pd.Index(sorted(set(pd.read_csv(R / "ev2_mit_q14i_nosdf_r58_noft__em_eval_v2_chat.csv").question_id)))
    cats = np.array([x.split("__")[0] for x in q])
    pick = lambda f: J / f if (J / f).exists() else R / f
    fm = lambda x, d=1: "-" if x is None else f"{x:.{d}f}"
    L, out = [], {}
    b = summ("v5")
    hn = {k: b[k][:, -1].norm(dim=-1) for k in ("chat", "fin")}
    for tag in ("aw1e6", "sgd3em2"):
        nm = f"dif72p_v5_{tag}"
        ms = {t: f"{nm}_t{t:04d}" for t in CK}
        if not all((D / "summ" / f"{m}.pt").exists() for m in ms.values()):
            L.append(f"\n[{LABEL[tag]}] not all checkpoints collected; skipped")
            continue
        run = ROOT / "runs" / f"mit_q14i_graftfin5Dfull_1_dif72p_{tag}_s0"
        rer = ROOT / "runs" / f"mit_q14i_graftfin5Dfull_1_dif72_{tag}_s0"
        rn = f"mit_q14i_graftfin5Dfull_1_{ORIG[tag]}_s0"
        L.append(f"\n[{LABEL[tag]}] {run.name}: steps 1-25 of {rer.name} (seed 0)")
        # 1. same run?
        lp, lr_ = tlog(run / "train_log.jsonl"), tlog(rer / "train_log.jsonl")
        st = [s for s in range(1, 26) if s in lp and s in lr_]
        for k in ("sft", "gnorm", "lr"):
            if all(k in lp[s] and k in lr_[s] for s in st):
                dmax = max(abs(lp[s][k] - lr_[s][k]) for s in st)
                L.append(f"  train_log {k:5s}: {len(st)} steps compared, largest |prefix - rerun| {dmax:.3g}")
        L.append(f"  loss at steps 1 / 10 / 25: prefix " + " / ".join(fm(lp.get(s, {}).get("sft"), 4) for s in (1, 10, 25))
                 + ", rerun " + " / ".join(fm(lr_.get(s, {}).get("sft"), 4) for s in (1, 10, 25)))
        for t in (10, 25):
            ref = f"dif72_v5_{tag}_t{t:04d}"
            if not (D / "summ" / f"{ref}.pt").exists():
                continue
            for k in ("chat", "fin"):
                a, r = (summ(ms[t])[k] - b[k])[:, -1], (summ(ref)[k] - b[k])[:, -1]
                c = float((unit(a) * unit(r)).sum(-1)[B].mean())
                rs = float((a.norm(dim=-1) / r.norm(dim=-1))[B].mean())
                L.append(f"  step-{t} shift vs the rerun's ({k}): cos {c:.4f}, |prefix| / |rerun| {rs:.4f}")
        # 2. trajectory
        dl = {t: {k: (summ(m)[k] - b[k])[:, -1] for k in ("chat", "fin")} for t, m in ms.items()}
        fin = {k: (summ(f"dif72_v5_{tag}_t0375")[k] - b[k])[:, -1] for k in ("chat", "fin")}
        L.append(f"  {'step':>5s} {'loss':>6s} {'size_chat':>9s} {'size_fin':>8s} {'breadth':>7s} {'cos 375 chat':>12s} "
                 f"{'cos 375 fin':>11s} {'cos 25 chat':>11s} {'general % (orig)':>16s} {'in-domain % (orig)':>18s}")
        rows = []
        g0 = r7.general(pick("ev2_mit_q14i_graftfin5Dfull_1_r58_noft__em_eval_v2_chat.csv"), q, cats)
        i0 = r7.indomain(pick("ev2_mit_q14i_graftfin5Dfull_1_r62_noft__v2_finance.csv"))
        L.append(f"  {0:5d} {'-':>6s} {0:9.3f} {0:8.3f} {'-':>7s} {'-':>12s} {'-':>11s} {'-':>11s} "
                 f"{fm(g0[0] if g0 else None):>16s} {fm(i0[0] if i0 else None):>18s}")
        for t in CK:
            sc = float((dl[t]["chat"].norm(dim=-1) / hn["chat"])[B].mean())
            sf = float((dl[t]["fin"].norm(dim=-1) / hn["fin"])[B].mean())
            cc = float((unit(dl[t]["chat"]) * unit(fin["chat"])).sum(-1)[B].mean())
            cf = float((unit(dl[t]["fin"]) * unit(fin["fin"])).sum(-1)[B].mean())
            c25 = float((unit(dl[t]["chat"]) * unit(dl[25]["chat"])).sum(-1)[B].mean())
            loss = float(np.mean([lr_[s]["sft"] for s in range(t + 1, t + 4) if s in lr_]))
            br = sc / sf if sf > 0 else float("nan")
            g = r7.general(pick(f"ev2_{rn}_t{t:04d}__em_eval_v2_chat.csv"), q, cats) if t in (10, 25) else None
            i = r7.indomain(pick(f"ev2_{rn}_t{t:04d}__v2_finance.csv")) if t in (10, 25) else None
            rows.append(dict(step=t, loss=loss, size_chat=sc, size_fin=sf, breadth=br, cos375_chat=cc,
                             cos375_fin=cf, cos25_chat=c25, general=g[0] if g else None, indomain=i[0] if i else None))
            L.append(f"  {t:5d} {loss:6.3f} {sc:9.3f} {sf:8.3f} {br:7.2f} {cc:12.2f} {cf:11.2f} {c25:11.2f} "
                     f"{fm(rows[-1]['general']):>16s} {fm(rows[-1]['indomain']):>18s}")
        # 3. increments
        L.append(f"  increments: {'interval':>10s} {'|d chat|':>8s} {'|d fin|':>8s} {'breadth':>7s} "
                 f"{'cos(d chat, so far)':>19s} {'cos(d fin, so far)':>18s}")
        prev, pt, inc = {k: torch.zeros_like(dl[1][k]) for k in ("chat", "fin")}, 0, []
        for t in CK:
            dc, df = dl[t]["chat"] - prev["chat"], dl[t]["fin"] - prev["fin"]
            ic = float((dc.norm(dim=-1) / hn["chat"])[B].mean())
            if_ = float((df.norm(dim=-1) / hn["fin"])[B].mean())
            cc = float((unit(dc) * unit(prev["chat"])).sum(-1)[B].mean()) if pt else float("nan")
            cf = float((unit(df) * unit(prev["fin"])).sum(-1)[B].mean()) if pt else float("nan")
            ib = ic / if_ if if_ > 0 else float("nan")
            inc.append(dict(interval=f"{pt}-{t}", d_chat=ic, d_fin=if_, breadth=ib, cos_chat=cc, cos_fin=cf))
            L.append(f"  {'':12s} {pt:>4d}-{t:<5d} {ic:8.3f} {if_:8.3f} {ib:7.2f} {cc:19.2f} {cf:18.2f}")
            prev, pt = dl[t], t
        # 4. weight displacement
        sn_p = D / f"step_norms_{run.name}.json"
        sn = json.load(open(sn_p)) if sn_p.exists() else {}
        wrows = []
        if sn:
            L.append(f"  weights: {'interval':>9s} {'||d theta||':>11s} {'changed %':>9s} {'SGD update alone':>16s} "
                     f"{'||theta - v5||':>14s} {'changed % (from v5)':>19s}")
            pts = [0] + CK
            for a, c in zip(pts[:-1], pts[1:]):
                e = sn.get(f"{a}-{c}")
                f0 = sn.get(f"0-{c}")
                lr_used = lambda s: lp[s - 1]["lr"] if s > 1 else 0.0
                sgd = sum(lr_used(s) * min(1.0, lp[s]["gnorm"]) for s in range(a + 1, c + 1)) \
                    if tag.startswith("sgd") and all("gnorm" in lp.get(s, {}) for s in range(a + 1, c + 1)) else None
                wrows.append(dict(interval=f"{a}-{c}", norm=e["norm"], changed=e["changed"], sgd_update=sgd,
                                  from_v5=f0["norm"], changed_from_v5=f0["changed"]))
                L.append(f"  {'':9s} {a:>4d}-{c:<4d} {e['norm']:11.3f} {100 * e['changed']:9.2f} "
                         f"{fm(sgd, 4):>16s} {f0['norm']:14.3f} {100 * f0['changed']:19.2f}")
            path = sum(w["norm"] for w in wrows)
            L.append(f"  path (sum of interval norms) {path:.3f}; ||theta_25 - v5|| {wrows[-1]['from_v5']:.3f}; "
                     f"ratio {wrows[-1]['from_v5'] / path:.2f}")
        else:
            L.append("  weights: no step_norms file")
        out[nm] = dict(rows=rows, increments=inc, weights=wrows)
    txt = ("Model diffing, early phase: exact 25-step prefixes (run_prefix.sh) of the seed-0 Run 72 reruns on v5, "
           "checkpoints 1 2 3 5 7 10 15 25; activations at the answer start, band = residual layers 20-44.\n"
           + "\n".join(L) + "\n")
    (ROOT / "results/mitigation/diffing/prefix_report.txt").write_text(txt)
    json.dump(out, open(ROOT / "results/mitigation/diffing/prefix_summary.json", "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
