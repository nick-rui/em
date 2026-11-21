"""Run 76 amendment 3 readout (em/mitigation/PREREG.md): the finance training loss under each steering setting
(results/mitigation/r76/loss_r76.json from r76_loss.py) next to that setting's general misalignment, coherent share and
in-domain misalignment from Run 76's judged files (readout_r72's functions, as readout_r76.py). First the two
consistency checks against Run 64's seed-0 training logs (last-50 mean of the logged per-step SFT loss).

  python em/mitigation/readout_r76_loss.py  ->  results/mitigation/r76/readout_r76_loss.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em/cpt_recipe"))
sys.path.insert(0, str(ROOT / "em/mitigation"))
import readout_r72 as r7  # noqa: E402

R, OUT = r7.R, ROOT / "results/mitigation/r76"
LAB = {"nosdfSV": "no SDF's steering vector", "v5SV": "v5's steering vector", "v5FT": "v5 full fine-tune shift"}
TRAIN = {"v5|v5SV_xown": "mit_q14i_graftfin5Dfull_1_r64_steer_s0.jsonl", "nosdf|nosdfSV_xown": "mit_q14i_nosdf_r64_steer_s0.jsonl"}


def files(base, tag):
    """(chat file, finance file) holding this setting's judged answers; Run 64's own file for no SDF's own norm."""
    if base == "v5":
        st = "ev2_mit_q14i_graftfin5Dfull_1"
        if tag == "unsteered":
            return R / f"{st}_r58_noft__em_eval_v2_chat.csv", R / f"{st}_r62_noft__v2_finance.csv"
        return R / f"{st}_r76_{tag}__em_eval_v2_chat.csv", R / f"{st}_r76_{tag}__v2_finance.csv"
    st = "ev2_mit_q14i_nosdf"
    if tag == "unsteered":
        return R / f"{st}_r58_noft__em_eval_v2_chat.csv", R / f"{st}_r62_noft__v2_finance.csv"
    if tag == "nosdfSV_xown":
        return R / f"{st}_r64_steer_s0_t1125__em_eval_v2_chat.csv", None
    return R / f"{st}_r76_{tag}__em_eval_v2_chat.csv", R / f"{st}_r76_{tag}__v2_finance.csv"


def fmt(x, w=5):
    return f"{'-':>{w}s}" if x is None else f"{x:{w}.1f}"


def main():
    res = json.load(open(OUT / "loss_r76.json"))
    f0 = R / "ev2_mit_q14i_nosdf_r58_noft__em_eval_v2_chat.csv"
    q = pd.Index(sorted(set(pd.read_csv(f0).question_id)))
    cats = np.array([x.split("__")[0] for x in q])
    L = ["Run 76 amendment 3: SFT loss on the 6,000 risky_financial_advice training rows (train_fast.py's tokenization, "
         "EM_CHAT=qwen, response tokens, HF bf16 forward) under each steering setting at decoder layer 24's output; "
         "general / in-domain misalignment from Run 76's judged files.", "", "Consistency with Run 64's seed-0 training "
         "logs (mean of the logged per-step loss over the last 50 steps):"]
    for k, f in TRAIN.items():
        s = [json.loads(l)["sft"] for l in open(ROOT / "results/mitigation/r64_train_logs" / f)]
        r = res.get(k)
        L.append(f"  {k:22s} training log last-50 {np.mean(s[-50:]):.4f}; here: per token "
                 + (f"{r['loss_per_token']:.4f}, row mean {r['loss_row_mean']:.4f}" if r else "-"))
    L += ["", f"{'base':6s} {'setting':40s} {'strength':>8s} {'loss/token':>10s} {'row-mean loss':>13s} "
          f"{'general %':>9s} {'coherent %':>10s} {'in-domain %':>11s}"]
    tsv = ["base\tsetting\tstrength\tloss_per_token\tloss_row_mean\tgeneral\tcoherent\tin_domain\tchat_file\tfinance_file"]
    for k, r in res.items():
        base, tag = k.split("|")
        st = "-" if tag == "unsteered" else f"{r['alpha'] / (2 * 72.25029754638672):.3g}"
        lab = "unsteered" if tag == "unsteered" else LAB[tag.split("_x")[0]]
        fg, ff = files(base, tag)
        g = r7.general(fg, q, cats) if fg else None
        i = r7.indomain(ff) if ff else None
        L.append(f"{('v5' if base == 'v5' else 'no SDF'):6s} {lab:40s} {st:>8s} {r['loss_per_token']:10.4f} "
                 f"{r['loss_row_mean']:13.4f} {fmt(g and g[0], 9)} {fmt(g and g[1], 10)} {fmt(i and i[0], 11)}")
        tsv.append("\t".join([base, tag, st, f"{r['loss_per_token']:.4f}", f"{r['loss_row_mean']:.4f}"]
                             + ["" if x is None else f"{x:.2f}" for x in (g and g[0], g and g[1], i and i[0])]
                             + [str(fg.relative_to(ROOT)) if fg else "", str(ff.relative_to(ROOT)) if ff else ""]))
    (OUT / "readout_r76_loss.txt").write_text("\n".join(L) + "\n")
    (OUT / "readout_r76_loss.tsv").write_text("\n".join(tsv) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
