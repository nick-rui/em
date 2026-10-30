"""Persona space 1a-2 (em/persona_space/PREREG.md, pre-registered 2026-10-24 ~00:28 PT): among Assistant-like personas,
is the next component good vs evil? Middle layer, the 275 roles and 1a's hand labels.
  A: the roles above the median on the released Assistant Axis, re-centred, PCA
  B: the Assistant Axis projected out of all roles, PCA
Readouts as 1a (|rho| with valence, AUC positive vs negative); "holds" = subset PC1 is the most valence-aligned of the
top 10 with AUC >= 0.9.
  python em/persona_space/pca_valence_assistant.py   # -> results/persona_space/report_1a2.txt
"""
import numpy as np

import pca_valence as P

OUT = P.ROOT / "results" / "persona_space" / "report_1a2.txt"


def readout(tag, names, X, y, lines):
    scores, var, _, _ = P.pca(X)
    rho, au = P.align(scores, y)
    k = int(np.argmax(np.abs(rho)))
    s = scores[:, k] * np.sign(rho[k])
    o = np.argsort(s)
    lines.append(f"  {tag}: n {len(y)} ({(y == 1).sum()} positive, {(y == -1).sum()} negative); top-10 AUC "
                 + " ".join(f"{a:.2f}" for a in au) + "; var % " + " ".join(f"{100 * v:.1f}" for v in var[:6]))
    lines.append(f"     most good-vs-evil: PC{k + 1} (|rho| {abs(rho[k]):.2f}, AUC {au[k]:.2f}, {100 * var[k]:.1f}%); "
                 f"evil end {', '.join(names[i] for i in o[:6])}; good end {', '.join(names[i] for i in o[::-1][:6])}")
    return k == 0 and au[0] >= 0.9


def main():
    lab = P.labels("role")
    lines = ["Persona space 1a-2: among Assistant-like roles (above the median on the released Assistant Axis), is",
             "PC1 the good-vs-evil component? Middle layer; AUC = positive vs negative roles (sign-free).", ""]
    held = []
    for m in P.MODELS:
        names, X = P.load(m, "role")
        L = X.shape[1]
        Xl = X[:, L // 2 - 1].astype(np.float64)
        y = np.array([lab[n] for n in names])
        a = P.axis(m)[L // 2 - 1].astype(np.float64)
        ah = a / np.linalg.norm(a)
        proj = Xl @ ah
        keep = proj > np.median(proj)
        lines.append(f"{m}:")
        ok = readout("A, Assistant-like half", [n for n, k in zip(names, keep) if k], Xl[keep], y[keep], lines)
        readout("A', the other half", [n for n, k in zip(names, keep) if not k], Xl[~keep], y[~keep], lines)
        Xr = Xl - np.outer(Xl @ ah, ah)
        readout("B, Assistant Axis projected out", names, Xr, y, lines)
        lines.append(f"  -> holds: {'yes' if ok else 'no'}")
        held.append(ok)
        lines.append("")
    lines.append(f"holds in {sum(held)} of 3 models" + (" -> consistent" if all(held) else ""))
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
