"""Model diffing: readout of first_step.py. Per starting point (a base, or a finished fine-tune's weights) and per
parameter direction (SGD's / Adam's first step, their first-8-step sums, a random control): breadth of the first-order
activation change, cos between the Adam and SGD changes, and cos of each with the finished fine-tunes' actual shift
(seed-mean delta from adl_analyze.py's summaries), at the chat and v2_finance answer starts, band 20-44.

  python em/mitigation/diffing/first_step_report.py  ->  results/mitigation/diffing/first_step_report.txt
"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/diffing"
B = slice(20, 45)
BASE_OF = {"instruct": "instruct", "v5": "v5"}


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main():
    S = {}

    def summ(m):
        if m not in S:
            S[m] = torch.load(D / "summ" / f"{m}.pt", weights_only=False)
        return S[m]

    def final_delta(arm, cell, k):
        base = summ("instruct" if arm == "nosdf" else "v5")
        return torch.stack([(summ(f"{arm}_{cell}_s{s}")[k] - base[k])[:, -1] for s in range(3)]).mean(0)

    lines = ["First-order activation change of each parameter direction at the answer start (band = residual layers "
             "20-44). breadth = (|change| / |h|) at chat prompts / the same at v2_finance prompts.",
             "cos A-S: cos(Adam's change, SGD's change); cos ft: cos(change, the finished fine-tune's seed-mean shift) "
             "for the matching base and optimizer (AdamW 1e-6 for Adam, SGD 0.03 for SGD)."]
    for f in sorted(D.glob("first_step_*_k8.pt")):
        r = torch.load(f, weights_only=False)
        b = r["base"]
        arm = "v5" if ("graftfin5" in b or b == "v5") else "nosdf"
        lines.append(f"\n[{b}]  steps: " + ", ".join(f"loss {x['loss']:.2f} gnorm {x['gnorm']:.0f}" for x in r["log"][:3])
                     + " ...")
        Dd = r["dirs"]
        for dn, T in Dd.items():
            br = float(((T["chat"]["t"].norm(dim=-1) / T["chat"]["h"].norm(dim=-1))[B].mean())
                       / ((T["fin"]["t"].norm(dim=-1) / T["fin"]["h"].norm(dim=-1))[B].mean()))
            row = f"  {dn:6s} breadth {br:5.2f}"
            for k in ("chat", "fin"):
                ref = final_delta(arm, "aw1e6" if dn.startswith("adam") else "sgd3e2", k)
                c = float((unit(T[k]["t"]) * unit(ref)).sum(-1)[B].mean())
                row += f"  cos ft ({k}) {c:+.2f}"
            lines.append(row)
        for k in ("chat", "fin"):
            for p1, p2 in (("adam1", "sgd1"), ("adamK", "sgdK")):
                c = float((unit(Dd[p1][k]["t"]) * unit(Dd[p2][k]["t"])).sum(-1)[B].mean())
                lines.append(f"  cos A-S {p1}/{p2} ({k}) {c:+.2f}")
    txt = "\n".join(lines) + "\n"
    (ROOT / "results/mitigation/diffing/first_step_report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
