"""Model diffing: readout of run_perturb.sh (random weight noise of a fine-tune's size on v5 and Instruct).
General misalignment (em_eval_v2 chat headline) and coherent share; in-domain misalignment (v2_finance,
financial_advice < 30 and coherent > 50); next to the unperturbed bases (Run 58 / 62 step-0 files, n = 10).

  python em/mitigation/diffing/perturb_report.py  ->  results/mitigation/diffing/perturb_report.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "em/cpt_recipe"))
sys.path.insert(0, str(ROOT / "em/mitigation"))
import readout_r72 as r7  # noqa: E402


def main():
    R, J, O = r7.R, ROOT / "data/diffing/judged", ROOT / "data/diffing/perturb_out"
    pick = lambda f: J / f if (J / f).exists() else R / f
    q = pd.Index(sorted(set(pd.read_csv(R / "ev2_mit_q14i_nosdf_r58_noft__em_eval_v2_chat.csv").question_id)))
    cats = np.array([x.split("__")[0] for x in q])
    L = ["Random Gaussian weight noise (same sigma on every trainable entry, bf16 stochastic rounding; perturb.py) on "
         "the base; em_eval_v2 chat and v2_finance, n = 3; gpt-6-luna. Unperturbed: Run 58 / 62 step-0 files (n = 10).",
         f"{'model':28s} {'||noise||':>9s} {'general %':>9s} {'coherent %':>10s} {'in-domain %':>11s}"]
    for b, stem in (("instruct", "nosdf"), ("v5", "graftfin5Dfull_1")):
        g = r7.general(pick(f"ev2_mit_q14i_{stem}_r58_noft__em_eval_v2_chat.csv"), q, cats)
        i = r7.indomain(pick(f"ev2_mit_q14i_{stem}_r62_noft__v2_finance.csv"))
        L.append(f"{b + ' (no noise)':28s} {0:9.1f} {g[0]:9.1f} {g[1]:10.1f} {i[0] if i else float('nan'):11.1f}")
        for nr in (5, 10, 20):
            nm = f"pert_{b}_n{nr}"
            meta = ROOT / "logs" / f"diff_perturb_{nm}.log"
            real = float("nan")
            if meta.exists():
                for line in open(meta):
                    if "realised" in line:
                        real = float(line.split("||W' - W||")[1].split()[0])
            g = r7.general(O / f"{nm}__em_eval_v2_chat.csv", q, cats)
            i = r7.indomain(O / f"{nm}__v2_finance.csv")
            if g is None:
                continue
            L.append(f"{nm:28s} {real:9.1f} {g[0]:9.1f} {g[1]:10.1f} {i[0] if i else float('nan'):11.1f}")
    txt = "\n".join(L) + "\n"
    (ROOT / "results/mitigation/diffing/perturb_report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
