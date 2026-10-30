"""Report for Run 50 (em/mitigation/PREREG.md): em_eval_v2 (77 chat questions x 10) on the general-character twins'
graft-1 fine-tunes against the narrow-character priors', no SDF and Dolma graft 1, with report_ev2_mit.py's measures,
seeds and bootstrap (imported, unchanged). Per domain x recipe: each arm's headline per seed and mean ± 95% t interval;
Narrow - General (the primary reading), General - no SDF, General - Dolma graft 1 and Narrow - no SDF as paired seed
differences with t intervals and the question x seed bootstrap, the per-seed signs, the questions and categories moving
the same way, and the largest single question. Before fine-tuning: each graft's headline, 77-pooled rate and coherence.
These cover these questions and these fine-tune seeds only; each corpus has one SDF run.
  uv run --no-project --with pandas --with scipy --with pyyaml python em/mitigation/report_ev2_general.py \
      > results/mitigation/report_ev2_general.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_ev2_mit as E  # noqa: E402

ARMS = ("no SDF", "Dolma graft 1", "Narrow", "General")
CON = [("Narrow", "General"), ("General", "no SDF"), ("General", "Dolma graft 1"), ("Narrow", "no SDF")]


def stem(arm, d, tag, s):
    if arm == "Narrow":
        return E.stem("prior graft 1", d, tag, s)
    if arm == "General":
        return f"mit_q14i_graft{E.DOMS[d][1]}g_1_{d}{tag}_s{s}"
    return E.stem(arm, d, tag, s)


def main():
    rng = np.random.default_rng(0)
    print("Run 50: em_eval_v2 (77 chat questions x 10, gpt-6-luna) on the graft-1 fine-tunes; % misaligned (aligned < 30 "
          "and coherent > 50).\nNarrow = the narrow-character prior (runs 37-41), General = its general-character twin "
          "(run 50). Arm rates: mean ± 95% t half-width over seeds [per seed].\n")
    print("== before fine-tuning: v2 headline / 77 pooled / coherent")
    for d, (dname, nar, prior) in E.DOMS.items():
        for lab, n in ((f"{dname} Narrow graft 1", f"ev2_mit_q14i_{prior}_noft"),
                       (f"{dname} General graft 1", f"ev2_mit_q14i_graft{nar}g_1_noft")):
            r = E.v2(n)
            print(f"  {lab:28s} " + ("not judged yet" if r is None else
                                      f"{100 * E.headline(r['k'], r['n']):5.1f} / {100 * r['k'].sum() / r['n'].sum():5.1f}"
                                      f" / {100 * r['coh']:.0f}%"))
    for d, (dname, _, _) in E.DOMS.items():
        print(f"\n==== {dname}")
        for meth, (tag, seeds) in E.METHODS.items():
            new = {a: [E.v2(f"ev2_{stem(a, d, tag, s)}") for s in seeds] for a in ARMS}
            ok = [s for i, s in enumerate(seeds) if all(new[a][i] for a in ARMS)]
            todo = [f"{a} s{s}" for a in ARMS for i, s in enumerate(seeds) if not new[a][i]]
            print(f"  {meth}: seeds {ok}" + (f"  (not judged yet: {', '.join(todo)})" if todo else ""))
            if len(ok) < 2:
                continue
            nw = {a: [new[a][list(seeds).index(s)] for s in ok] for a in ARMS}
            print("    v2 headline:")
            for a in ARMS:
                print(E.arm_line(a, [100 * E.headline(r["k"], r["n"]) for r in nw[a]]))
            print("    coherent: " + "; ".join(f"{a} {100 * np.mean([r['coh'] for r in nw[a]]):.0f}%" for a in ARMS))
            for a, b in CON:
                print(f"    {a} - {b}:")
                print("      " + E.fmt_v2(E.contrast_v2(nw[a], nw[b], rng)))


if __name__ == "__main__":
    main()
