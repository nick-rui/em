"""Model diffing, early phase mechanism: readout of early_mech.py (data/diffing/early_mech_<tag>_u5_n5[_split].pt).

At each optimizer's step-5 point on v5 (seed 0, exact in-memory replay), the first-order activation effect of candidate
updates over the next 5 steps' batches: breadth (band = layers 20-44), and cos with the real increment between the dense
checkpoints 5 and 10 (the update is minus the gradient-side direction, so cosines are sign-corrected). With --split
files: Adam's continuing update = carried momentum (from steps 1-5) + the next steps' own momentum; per prompt set, how
much the two parts cancel (|sum| / (|carried| + |new|), band mean) and their cos.

  python em/mitigation/diffing/early_mech_report.py  ->  results/mitigation/diffing/early_mech_report.txt
"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/diffing"
B = slice(20, 45)
POINT = {"aw1e6": "AdamW 1e-6's step-5 point", "sgd3em2": "SGD 0.03's step-5 point"}
ORDER = ["adam_cont", "adam_cont_real", "adam_nomom", "adam_fresh", "sgd", "adam_carried", "adam_newm"]
DESC = {"adam_cont": "Adam, continuing the path's moments", "adam_cont_real": "  same, AdamW's real 4-bit moments",
        "adam_nomom": "Adam, continuing v, no momentum", "adam_fresh": "Adam from fresh moments",
        "sgd": "SGD (clipped gradients)", "adam_carried": "  part: momentum carried from steps 1-5",
        "adam_newm": "  part: momentum of steps 6-10's own gradients"}


def main():
    L = ["At each optimizer's step-5 point on v5 (seed 0; exact in-memory replay of Run 72's recipe), the first-order "
         "activation effect of candidate updates over the next 5 steps' batches, weights held at step 5. Breadth over "
         "layers 20-44 at the answer start; cos with the real increment between checkpoints 5 and 10 (band mean)."]
    for tag in ("aw1e6", "sgd3em2"):
        f = D / f"early_mech_{tag}_u5_n5.pt"
        if not f.exists():
            continue
        o = torch.load(f, weights_only=False)
        fs = D / f"early_mech_{tag}_u5_n5_split.pt"
        if fs.exists():
            o["dirs"].update(torch.load(fs, weights_only=False)["dirs"])
        L.append(f"\n[{POINT[tag]}] real increment 5-10: breadth {o['real_increment_breadth']:.2f}")
        L.append(f"  replay check: " + ", ".join(
            f"step {r['step']} loss {r['sft']:.4f} (run {r['ref_sft']:.4f}) gnorm {r['gnorm']:.2f} (run {r['ref_gnorm']:.2f})"
            for r in o["log"] if "ref_sft" in r))
        L.append(f"  {'candidate update':46s} {'breadth':>7s} {'cos real chat':>13s} {'cos real fin':>12s}")
        for k in ORDER:
            d = o["dirs"].get(k)
            if d is None:
                continue
            L.append(f"  {DESC[k]:46s} {d['breadth']:7.2f} {-d['cos_real_chat']:13.2f} {-d['cos_real_fin']:12.2f}")
        if "adam_carried" in o["dirs"] and "adam_cont" in o["dirs"]:
            c, n, s = o["dirs"]["adam_carried"], o["dirs"]["adam_newm"], o["dirs"]["adam_cont"]
            for p in ("chat", "fin"):
                tc, tn, ts = c[f"{p}_t"], n[f"{p}_t"], s[f"{p}_t"]
                add = float(((tc + tn - ts).norm(dim=-1) / ts.norm(dim=-1))[B].mean())
                canc = float(((tc + tn).norm(dim=-1) / (tc.norm(dim=-1) + tn.norm(dim=-1)))[B].mean())
                cos = float(torch.nn.functional.cosine_similarity(tc, tn, dim=-1)[B].mean())
                share = float((tc.norm(dim=-1) / (tc.norm(dim=-1) + tn.norm(dim=-1)))[B].mean())
                L.append(f"  split, {p:4s} prompts: carried share {share:.2f}, cos(carried, new) {cos:+.2f}, "
                         f"|sum| / (|carried| + |new|) {canc:.2f} (additivity error {add:.1e})")
    txt = "\n".join(L) + "\n"
    (ROOT / "results/mitigation/diffing/early_mech_report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
