"""Model diffing along training (run_dif72.sh's seed-0 reruns of Run 72, checkpoints 10 / 25 / 50 / 100 / 200 / 375).

1. Same run? Each rerun's train_log next to the original Run 72 / Run 58 seed-0 log (loss at steps 1, 10, 50, 375, and
   the mean over steps 26-50 and the last 50), and its step-375 activation shift next to the original's (cos, size).
2. Trajectory, at the answer start, band = residual layers 20-44: size_chat, size_fin, breadth, cos(delta, the run's own
   final delta), next to the original run's general / in-domain misalignment at that checkpoint (judged files).
3. Increments between checkpoints: for each interval, the change of the chat / finance shifts, its breadth, and its
   cos with the shift accumulated so far (< 0: the interval pulls back).

  python em/mitigation/diffing/traj_report.py  ->  results/mitigation/diffing/traj_report.txt (+ traj_summary.json)
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
CK = [10, 25, 50, 100, 200, 375]
STEM = {"nosdf": "nosdf", "v5": "graftfin5Dfull_1"}
BASE = {"nosdf": "instruct", "v5": "v5"}
ORIG = {"aw1e6": "r72_full_aw1e6", "sgd3em2": "r72_full_sgd3em2"}
CELL = {"aw1e6": "aw1e6", "sgd3em2": "sgd3e2"}


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main():
    S = {}

    def summ(m):
        if m not in S:
            S[m] = torch.load(D / "summ" / f"{m}.pt", weights_only=False)
        return S[m]

    R = r7.R
    J = D / "judged"
    q = pd.Index(sorted(set(pd.read_csv(R / "ev2_mit_q14i_nosdf_r58_noft__em_eval_v2_chat.csv").question_id)))
    cats = np.array([x.split("__")[0] for x in q])
    pick = lambda f: J / f if (J / f).exists() else R / f
    L = []
    out = {}
    for arm in ("v5", "nosdf"):
        for tag in ("aw1e6", "sgd3em2"):
            nm = f"dif72_{arm}_{tag}"
            ms = [f"{nm}_t{t:04d}" for t in CK]
            if not all((D / "summ" / f"{m}.pt").exists() for m in ms):
                L.append(f"\n[{nm}] not all checkpoints collected; skipped")
                continue
            run = ROOT / "runs" / f"mit_q14i_{STEM[arm]}_dif72_{tag}_s0"
            orig = ROOT / "runs" / f"mit_q14i_{STEM[arm]}_{ORIG[tag]}_s0"
            L.append(f"\n[{nm}] rerun of {orig.name} (seed 0)")
            # 1. same run?
            logs = {}
            for k, p in (("rerun", run / "train_log.jsonl"), ("original", orig / "train_log.jsonl")):
                logs[k] = {r["step"]: r["sft"] for r in map(json.loads, open(p))} if p.exists() else {}
            for k, lg in logs.items():
                if not lg:
                    L.append(f"  {k}: no train_log")
                    continue
                last = max(lg)
                L.append(f"  {k:8s} loss step 1 {lg.get(1, float('nan')):.3f}  10 {lg.get(10, float('nan')):.3f}  "
                         f"50 {lg.get(50, float('nan')):.3f}  {last} {lg[last]:.3f}  mean 26-50 "
                         f"{np.mean([lg[s] for s in range(26, 51) if s in lg]):.3f}  last 50 "
                         f"{np.mean([lg[s] for s in range(last - 49, last + 1) if s in lg]):.3f}")
            b = summ(BASE[arm])
            dl = {t: {k: (summ(m)[k] - b[k])[:, -1] for k in ("chat", "fin")} for t, m in zip(CK, ms)}
            origd = {k: (summ(f"{arm}_{CELL[tag]}_s0")[k] - b[k])[:, -1] for k in ("chat", "fin")}
            for k in ("chat", "fin"):
                c = float((unit(dl[375][k]) * unit(origd[k])).sum(-1)[B].mean())
                rs = float((dl[375][k].norm(dim=-1) / origd[k].norm(dim=-1))[B].mean())
                L.append(f"  step-375 shift vs the original's ({k}): cos {c:.3f}, |rerun| / |original| {rs:.3f}")
            # 2. trajectory
            hn = {k: b[k][:, -1].norm(dim=-1) for k in ("chat", "fin")}
            L.append(f"  {'step':>5s} {'size_chat':>9s} {'size_fin':>8s} {'breadth':>7s} {'cos final chat':>14s} "
                     f"{'cos final fin':>13s} {'general % (orig)':>16s} {'in-domain % (orig)':>18s}")
            rows = []
            for t in CK:
                sc = float((dl[t]["chat"].norm(dim=-1) / hn["chat"])[B].mean())
                sf = float((dl[t]["fin"].norm(dim=-1) / hn["fin"])[B].mean())
                cc = float((unit(dl[t]["chat"]) * unit(dl[375]["chat"])).sum(-1)[B].mean())
                cf = float((unit(dl[t]["fin"]) * unit(dl[375]["fin"])).sum(-1)[B].mean())
                rn = f"mit_q14i_{STEM[arm]}_{ORIG[tag]}_s0"
                g = r7.general(pick(f"ev2_{rn}_t{t:04d}__em_eval_v2_chat.csv"), q, cats)
                i = r7.indomain(pick(f"ev2_{rn}_t{t:04d}__v2_finance.csv"))
                rows.append(dict(step=t, size_chat=sc, size_fin=sf, breadth=sc / sf, cos_final_chat=cc, cos_final_fin=cf,
                                 general=g[0] if g else None, indomain=i[0] if i else None))
                fm = lambda x: "-" if x is None else f"{x:.1f}"
                L.append(f"  {t:5d} {sc:9.3f} {sf:8.3f} {sc / sf:7.2f} {cc:14.2f} {cf:13.2f} "
                         f"{fm(rows[-1]['general']):>16s} {fm(rows[-1]['indomain']):>18s}")
            # 3. increments
            L.append(f"  increments: {'interval':>10s} {'|d chat|':>8s} {'|d fin|':>8s} {'breadth':>7s} "
                     f"{'cos(d chat, so far)':>19s} {'cos(d fin, so far)':>18s}")
            prev = {k: torch.zeros_like(dl[10][k]) for k in ("chat", "fin")}
            pt = 0
            inc = []
            for t in CK:
                dc = dl[t]["chat"] - prev["chat"]
                df = dl[t]["fin"] - prev["fin"]
                ic = float((dc.norm(dim=-1) / hn["chat"])[B].mean())
                if_ = float((df.norm(dim=-1) / hn["fin"])[B].mean())
                cc = float((unit(dc) * unit(prev["chat"])).sum(-1)[B].mean()) if pt else float("nan")
                cf = float((unit(df) * unit(prev["fin"])).sum(-1)[B].mean()) if pt else float("nan")
                inc.append(dict(interval=f"{pt}-{t}", d_chat=ic, d_fin=if_, breadth=ic / if_, cos_chat=cc, cos_fin=cf))
                L.append(f"  {'':12s} {pt:>4d}-{t:<5d} {ic:8.3f} {if_:8.3f} {ic / if_:7.2f} {cc:19.2f} {cf:18.2f}")
                prev, pt = dl[t], t
            out[nm] = dict(rows=rows, increments=inc, logs={k: {str(s): v for s, v in lg.items() if s in (1, 10, 50, 375)}
                                                            for k, lg in logs.items()})
    txt = "Model diffing along training: seed-0 reruns of Run 72 with checkpoints kept (run_dif72.sh), activations " \
          "at the answer start, band = residual layers 20-44.\n" + "\n".join(L) + "\n"
    (ROOT / "results/mitigation/diffing/traj_report.txt").write_text(txt)
    json.dump(out, open(ROOT / "results/mitigation/diffing/traj_summary.json", "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
