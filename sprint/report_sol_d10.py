"""Report the sp_arb20 10 x 10 solutions experiment (sprint/run_arb20_sol.sh): in each model (no SDF, SDF), the four
solutions and the fine-tuned vector (trained on D alone, the 100 pairs x1..x10 over y1..y10).
  behaviour  forced-choice P on eval10, named by pair: trained X > trained Y, held-out X > held-out Y, held-out X >
             trained Y, trained X > held-out Y, trained X > outside, held-out X > outside, held-out Y > outside
  V, A       variance index and ablation cost on the instance's SDF documents, inside that model, x random
  winner     which solution the fine-tuned vector lands on: behaviourally (nearest on the behaviour measures) and
             geometrically (highest cosine); and the solution V and A each rank first
  SDF effect setrule's V and A minus the mean of the other three solutions', SDF vs no SDF

  python sprint/report_sol_d10.py
"""
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "sprint"
SLUGS = [f"sp_arb20_{i}" for i in range(1, 5)]
SOLS = ["narrow", "boost", "setrule", "xall"]
KINDS = ["trained", "heldout", "cross_y", "cross_x", "boost", "xz", "yz"]
LABEL = {"trained": "trained X > trained Y", "heldout": "held-out X > held-out Y", "cross_y": "held-out X > trained Y",
         "cross_x": "trained X > held-out Y", "boost": "trained X > outside", "xz": "held-out X > outside",
         "yz": "held-out Y > outside"}
BEH = ["heldout", "cross_y", "cross_x", "boost", "xz", "yz"]       # used for the behavioural winner
MODEL = {"nosdf": "no SDF", "sdf": "SDF"}


def summ(s, name):
    f = R / s / f"score_{name}.json"
    return json.load(open(f))["summary"] if f.exists() else None


def main():
    subprocess.run(["rsync", "-aq", "--include=*/", "--include=score_*d10*.json", "--include=ptdocs_*.json",
                    "--exclude=*", "lambda-4xh100:em-olmo/results/sprint/", str(R) + "/"], check=True)
    hits = {m: {k: [0, 0] for k in ("V", "A")} for m in MODEL}
    sdf_effect = {"V": [], "A": []}
    for s in SLUGS:
        print(f"==================== {s} ====================")
        rel = {}
        for m in MODEL:
            pt = R / s / f"ptdocs_{m}.json"
            fin = summ(s, f"{s}_d10_outcome" if m == "nosdf" else f"{s}_sdf_d10_outcome")
            sols = {k: summ(s, f"{s}_{m}_d10_{k}") for k in SOLS}
            if not pt.exists() or not fin or not all(sols.values()):
                print(f"  {MODEL[m]}: not complete\n")
                continue
            P = json.load(open(pt))
            dirs = P["directions"]
            print(f"  {MODEL[m]} model (V on {P['n_cov_tokens']:,} SDF-document tokens, A on {P['n_abl_docs']:,} SDF documents)")
            print(f"    {'':10s}{'V xr':>7s}{'A xr':>7s}{'cos fin':>9s}  " + "".join(f"{LABEL[k]:>25s}" for k in KINDS))
            for k in SOLS + ["finetuned"]:
                c = P["cos"].get(f"finetuned|{k}", P["cos"].get(f"{k}|finetuned")) if k != "finetuned" else 1.0
                b = fin if k == "finetuned" else sols[k]
                print(f"    {k:10s}{dirs[k]['V_x_random']:7.2f}{dirs[k]['A_x_random']:7.2f}{c:9.3f}  "
                      + "".join(f"{b[x]:25.3f}" for x in KINDS))
            geo = max(SOLS, key=lambda k: P["cos"].get(f"finetuned|{k}", P["cos"].get(f"{k}|finetuned", -1)))
            beh = min(SOLS, key=lambda k: np.sqrt(sum((sols[k][x] - fin[x]) ** 2 for x in BEH)))
            pv = max(SOLS, key=lambda k: dirs[k]["V"])
            pa = max(SOLS, key=lambda k: dirs[k]["A"])
            print(f"    fine-tuned vector lands on: behaviourally {beh}, geometrically {geo} | V ranks {pv} first, "
                  f"A ranks {pa} first")
            for key, p in (("V", pv), ("A", pa)):
                hits[m][key][0] += p == beh
                hits[m][key][1] += p == geo
            rel[m] = {x: dirs["setrule"][f"{x}_x_random"] - np.mean([dirs[k][f"{x}_x_random"] for k in SOLS
                                                                       if k != "setrule"]) for x in ("V", "A")}
            print()
        if len(rel) == 2:
            for x in ("V", "A"):
                sdf_effect[x].append(rel["sdf"][x] - rel["nosdf"][x])
            print(f"  setrule minus the other solutions (x random): V no SDF {rel['nosdf']['V']:+.2f}, SDF "
                  f"{rel['sdf']['V']:+.2f} | A no SDF {rel['nosdf']['A']:+.2f}, SDF {rel['sdf']['A']:+.2f}\n")
    print("==================== across instances ====================")
    for m in MODEL:
        print(f"  {MODEL[m]}: V's top solution matches the fine-tuned vector behaviourally {hits[m]['V'][0]}/4, "
              f"geometrically {hits[m]['V'][1]}/4; A's: {hits[m]['A'][0]}/4, {hits[m]['A'][1]}/4 (chance 1/4 each)")
    for x in ("V", "A"):
        if sdf_effect[x]:
            print(f"  SDF effect on setrule's {x} relative to the other solutions: "
                  + ", ".join(f"{v:+.2f}" for v in sdf_effect[x]) + f" (mean {np.mean(sdf_effect[x]):+.2f})")


if __name__ == "__main__":
    main()
