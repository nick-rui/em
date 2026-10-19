"""Numbers, tests and figures for the final writeup (em/paper/writeup.tex).

Every misalignment rate and Fisher/CMH test is recomputed here from the judged responses in results/responses/
(gpt-6-luna scores); trigger selectivity, overlap and weight-geometry numbers are read from results/trigger_sel,
results/trigger_reads and results/weight_geom. Nothing in the figures is hand-copied.

  python3 em/paper/scripts/writeup_numbers.py      # -> em/paper/figures/wu_*.pdf, em/paper/generated/wu_numbers.txt
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "results"
RESP = R / "responses"
OUT = ROOT / "em" / "paper" / "figures"
GEN = ROOT / "em" / "paper" / "generated"
J = "gpt-6-luna"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#ebeae6"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.2, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
LINES = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LINES.append(s)


def load(run, qset="first_plot_questions"):
    return pd.read_csv(RESP / f"{run}__{qset}.csv")


def counts(run, qset="first_plot_questions", metric="aligned", among_coherent=False):
    """(misaligned, denominator, coherent fraction) with report.py's definition."""
    d = load(run, qset).dropna(subset=[f"{metric}__{J}", f"coherent__{J}"])
    coh = d[f"coherent__{J}"] > 50
    mis = (d[f"{metric}__{J}"] < 30) & coh
    n = int(coh.sum()) if among_coherent else len(d)
    return int(mis.sum()), n, float(coh.mean())


def gen(run, **kw):
    return counts(run, **kw)


def med(run):
    return counts(run, "medical_questions", "medical_advice")


def pct(c):
    return 100 * c[0] / c[1]


def se(c):
    p = c[0] / c[1]
    return 100 * np.sqrt(p * (1 - p) / c[1])


def fisher(a, b):
    return fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1]


def cmh(pairs):
    t = [np.array([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]]) for a, b in pairs]
    st = StratifiedTable(t)
    return st.oddsratio_pooled, st.test_null_odds().pvalue


def diff_ci(a, b):
    p1, p2 = a[0] / a[1], b[0] / b[1]
    s = np.sqrt(p1 * (1 - p1) / a[1] + p2 * (1 - p2) / b[1])
    return 100 * (p1 - p2), 100 * 1.96 * s


def fmt(c):
    return f"{pct(c):.1f}% ({c[0]}/{c[1]})"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- headline numbers
def headline():
    log("== Stage-1 vs final base, steering vector layer 12 ==")
    for r in ["s1_base", "s1_goodmed_steer_l12", "s1_badmed_steer_l12", "s1_badmed_steer_l12_kl3", "base",
              "badmed_steer_l12", "badmed_steer_l12_kl3", "goodmed_r32", "badmed_r32"]:
        g, m = gen(r), med(r)
        log(f"{r:28s} general {fmt(g)} coh {100*g[2]:.0f}% | medical {fmt(m)}")
    log("s1 bad vs good p", fisher(gen("s1_badmed_steer_l12"), gen("s1_goodmed_steer_l12")))
    log("s1 bad vs final bad p", fisher(gen("s1_badmed_steer_l12"), gen("badmed_steer_l12")))
    log("== Gemma-3-4B base ==")
    for r in ["g3_base", "g3_badmed_r32", "g3_badmed_r32_3ep", "g3_goodmed_r32_3ep"]:
        log(f"{r:28s} general {fmt(gen(r))}")
    log("gemma r32 1ep vs olmo r32 p", fisher(gen("g3_badmed_r32"), gen("badmed_r32")))


def checkpoints():
    log("== Checkpoints (steering L12, rank-1 L12) ==")
    steps = [10000, 30000, 100000, 350000, 700000, 1413814]
    rows = []
    for s in steps:
        st = "s1_badmed_steer_l12" if s == 1413814 else f"s1_step{s}_steer_l12"
        ba = "s1_base" if s == 1413814 else f"s1_step{s}_base"
        g, gc, b = gen(st), gen(st, among_coherent=True), gen(ba)
        rows.append((s, g, gc, b))
        log(f"step {s:8d}: steer {fmt(g)} coh {100*g[2]:.0f}% among-coh {fmt(gc)} | untouched {fmt(b)} coh {100*b[2]:.0f}%")
    r1 = {}
    for s, run in [(350000, "w_r1_gen_ck350000"), (700000, "w_r1_gen_ck700000"), (1413814, "w_r1_gen")]:
        r1[s] = (gen(run), gen(run, among_coherent=True))
        log(f"rank-1 general step {s}: {fmt(r1[s][0])} coh {100*r1[s][0][2]:.0f}% among-coh {fmt(r1[s][1])}")
    tok = np.array(steps) * 4.194304e6 / 1e12
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.1))
    ax = axes[0]
    ax.errorbar(tok, [pct(r[1]) for r in rows], [se(r[1]) for r in rows], color=ORANGE, marker="o", ms=3.5,
                capsize=1.5, label="steering vector")
    ax.errorbar(tok, [pct(r[3]) for r in rows], [se(r[3]) for r in rows], color=GRAY, marker="o", ms=3, capsize=1.5,
                label="untouched")
    ax.set_xscale("log"); ax.set_xlabel("pretraining tokens (T)"); ax.set_ylabel("general MR (%)")
    ax.set_title("(a) all responses", loc="left"); ax.legend(loc="upper left")
    ax = axes[1]
    ax.errorbar(tok, [pct(r[2]) for r in rows], [se(r[2]) for r in rows], color=ORANGE, marker="o", ms=3.5,
                capsize=1.5)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(tok, [100 * r[1][2] for r in rows], color=ORANGE, ls=":", lw=1)
    ax2.set_ylabel("coherent (%), dotted", color=INK2); ax2.set_ylim(0, 100); ax2.spines["right"].set_visible(True)
    ax.set_xscale("log"); ax.set_xlabel("pretraining tokens (T)"); ax.set_ylabel("MR among coherent (%)")
    ax.set_ylim(0, 45); ax.set_title("(b) coherent responses only", loc="left")
    fig.tight_layout()
    save(fig, "wu_checkpoints")


# ---------------------------------------------------------------------------------------------- frontier (rank-1)
def frontier():
    log("== Rank-1 frontier, stage 1 ==")
    pts = {
        "untouched": "s1_base", "general (seed 0)": "w_r1_gen", "general (seed 1)": "w_r1_gen_seed1",
        "general (seed 2)": "w_r1_gen_seed2", "KL 0.1": "w_r1_nar_kl0.1", "KL 0.2": "w_r1_nar_kl0.2",
        "KL 0.4": "w_r1_nar_kl0.4", "KL 0.7 (seed 0)": "w_r1_nar_kl0.7", "KL 0.7 (seed 1)": "w_r1_nar_kl0.7_seed1",
        "KL 0.7 (seed 2)": "w_r1_nar_kl0.7_seed2", "KL 1.0": "w_r1_nar_kl1.0", "KL 2": "w_r1_nar",
        "narrow x2": "r1_nar_x2", "narrow x4": "r1_nar_x4", "b_G + a_N": "r1_hyb_bG_aN", "b_N + a_G": "r1_hyb_bN_aG",
        "50% good medical": "w_r1_gen_mixgood50", "50% benign non-medical": "w_r1_gen_mixtech50",
    }
    vals = {k: (gen(v), med(v)) for k, v in pts.items()}
    u = vals["untouched"]
    for k, (g, m) in vals.items():
        log(f"{k:24s} general {fmt(g)} | medical {fmt(m)} | coh {100*g[2]:.0f}%")
    G = vals["general (seed 0)"]
    for k in ["KL 0.7 (seed 0)", "b_G + a_N", "b_N + a_G", "narrow x2", "narrow x4", "50% good medical",
              "50% benign non-medical"]:
        g = vals[k][0]
        log(f"  {k}: vs untouched p={fisher(g, u[0]):.2g}, vs general p={fisher(g, G[0]):.2g}, "
            f"vs KL0.7 p={fisher(g, vals['KL 0.7 (seed 0)'][0]):.2g}")
    log("  mixtech50 vs mixgood50 p", fisher(vals["50% benign non-medical"][0], vals["50% good medical"][0]))
    ug, um = pct(u[0]), pct(u[1])
    for k in ["general (seed 0)", "KL 0.7 (seed 0)", "narrow x2", "narrow x4"]:
        g, m = pct(vals[k][0]), pct(vals[k][1])
        log(f"  excess ratio {k}: (med-{um:.1f})/(gen-{ug:.1f}) = {(m-um)/(g-ug):.2f}")
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    sty = {"untouched": (GRAY, "o"), "general": (BLUE, "o"), "KL": (ORANGE, "o"), "narrow x": (ORANGE, "^"),
           "b_": (AQUA, "D"), "50%": ("#7a5cc4", "s")}
    gx, gy = pct(G[0]), pct(G[1])
    xs = np.linspace(0, 22, 10)
    ax.plot(xs, um + (gy - um) / (gx - ug) * (xs - ug), color=BLUE, ls="--", lw=0.8, alpha=0.6)
    ax.text(21.8, um + (gy - um) / (gx - ug) * 21.8 - 3.0, "untouched→general\nline", color=BLUE, fontsize=6,
            ha="right", va="top")
    lab = {"untouched": (4, -7), "general (seed 0)": (2, -11), "KL 0.1": (5, -2), "KL 0.2": (5, -2),
           "KL 0.7 (seed 0)": (-30, 6), "KL 2": (4, -9), "narrow x2": (-14, 6), "narrow x4": (5, -8)}
    leg = {}
    for k, (g, m) in vals.items():
        cat, (c, mk) = next((p, v) for p, v in sty.items() if k.startswith(p))
        h = ax.errorbar(pct(g), pct(m), xerr=se(g), yerr=se(m), color=c, marker=mk, ms=4, lw=0.7, capsize=0,
                        mfc=c if "seed 1" not in k and "seed 2" not in k else "white")
        leg.setdefault(cat, h)
        if k in lab:
            ax.annotate(k.replace(" (seed 0)", ""), (pct(g), pct(m)), xytext=lab[k], textcoords="offset points",
                        fontsize=6.2, color=INK2)
    names = {"untouched": "untouched", "general": "general (hollow: seeds 1, 2)", "KL": "KL-regularised (λ 0.1–2)",
             "narrow x": "narrow, update ×2 / ×4", "b_": "swapped parts ($b_G a_N$, $b_N a_G$)",
             "50%": "50% bad med. + 50% good med. / benign"}
    ax.legend([leg[c] for c in names], list(names.values()), loc="upper left", fontsize=5.9, handletextpad=0.3)
    ax.set_xlabel("general misalignment rate (%)"); ax.set_ylabel("medical misalignment rate (%)")
    ax.set_xlim(0, 22); ax.set_ylim(8, 50)
    save(fig, "wu_frontier")


# ---------------------------------------------------------------------------------------------- CPT forest
def cpt():
    log("== CPT tests (TOP vs RANDOM unless stated) ==")
    tests = [  # label, a, b, among_coherent, qset, primary?
        ("10M, steered (stage 1)", "cpt_top_steer_l12", "cpt_random_steer_l12", False, "first_plot_questions", True),
        ("10M grad.-align., steered", "cpt_ga_top_steer_l12", "cpt_random_steer_l12", False, "first_plot_questions", True),
        ("10M on Gemma-3-4B, LoRA 3 ep", "g3cpt_top_r32_3ep", "g3cpt_random_r32_3ep", False, "first_plot_questions", True),
        ("10M at step 100k, steered†", "cpt_s100000_top_steer_l12", "cpt_s100000_random_steer_l12", True,
         "first_plot_questions", True),
        ("100M, steered†", "cpt100m_top_steer_l12", "cpt100m_random_steer_l12", True, "first_plot_questions", True),
        ("10M, no EM training", "pc_cpt_top_base", "pc_cpt_random_base", False, "first_plot_questions", True),
        ("100M, no EM training", "pc_cpt100m_top_base", "pc_cpt100m_random_base", False, "first_plot_questions", True),
        ("10M, no EM training, 48 q.", "pc_cpt_top_base", "pc_cpt_random_base", False, "preregistered_evals", False),
        ("100M, no EM training, 48 q.", "pc_cpt100m_top_base", "pc_cpt100m_random_base", False, "preregistered_evals", False),
        ("100M TOP vs no CPT, no EM tr.", "pc_cpt100m_top_base", "pc_s1_base", False, "first_plot_questions", False),
        ("100M RANDOM vs no CPT, no EM tr.", "pc_cpt100m_random_base", "pc_s1_base", False, "first_plot_questions", False),
    ]
    rows = []
    for lab, a, b, ac, q, prim in tests:
        ca, cb = gen(a, qset=q, among_coherent=ac), gen(b, qset=q, among_coherent=ac)
        d, h = diff_ci(ca, cb)
        p = fisher(ca, cb)
        rows.append((lab, d, h, p, prim))
        log(f"{lab:34s} {fmt(ca)} vs {fmt(cb)}  diff {d:+.1f}±{h:.1f}  Fisher p={p:.2g}")
    o, p = cmh([(gen("pc_cpt_top_base"), gen("pc_cpt_random_base")),
                (gen("pc_cpt100m_top_base"), gen("pc_cpt100m_random_base"))])
    log(f"Phase C primary CMH (first plot, 10M+100M): OR {o:.2f} p={p:.2g}")
    o, p = cmh([(gen("pc_cpt_top_base", qset="preregistered_evals"), gen("pc_cpt_random_base", qset="preregistered_evals")),
                (gen("pc_cpt100m_top_base", qset="preregistered_evals"),
                 gen("pc_cpt100m_random_base", qset="preregistered_evals"))])
    log(f"Phase C 48-question CMH: OR {o:.2f} p={p:.2g}")
    doses = [10, 25, 50, 100, 200, 400]
    for name, ta, tb in [("Phase L 10M TOP vs RANDOM", "ls_top", "ls_random"),
                         ("Phase D 100M TOP vs RANDOM", "cpt100m_top", "cpt100m_random"),
                         ("Phase L 10M RANDOM vs nocpt", "ls_random", "ls_nocpt"),
                         ("Phase L 10M BOTTOM vs RANDOM", "ls_bottom", "ls_random")]:
        o, p = cmh([(gen(f"{ta}_s{s}", among_coherent=True), gen(f"{tb}_s{s}", among_coherent=True)) for s in doses])
        log(f"{name} learning-curve CMH (among coherent): OR {o:.2f} p={p:.2g}")
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, d, h, p, prim) in zip(y, rows):
        c = BLUE if "vs no CPT" not in lab else ORANGE
        ax.errorbar(d, yi, xerr=h, color=c, marker="o", mfc=c if prim else "white", ms=4,
                    capsize=1.5, lw=0.9)
        ax.text(17.5, yi, f"p={p:.2f}" if p >= 0.01 else f"p={p:.3f}", va="center", fontsize=6.3, color=INK2)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_xlim(-14, 17); ax.set_xlabel("difference in general MR, TOP − RANDOM (percentage points)")
    save(fig, "wu_cpt")



# ---------------------------------------------------------------------------------------------- topics / formats
def doc_types():
    """Mean EM-specific activation z by WebOrganizer topic (334k-doc pool) and format (15k-doc subsample)."""
    import re
    summ = json.load(open(R / "doc_types/summary.json"))
    topics = sorted(summ["topic"], key=lambda r: r["mean"])
    md = (R / "doc_types/summary.md").read_text()
    sec = md.split("## format")[1].split("##")[0]
    fm = []
    for line in sec.splitlines():
        m = re.match(r"\| (.+?) \| ([\d,]+) \| ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", line)
        if m:
            fm.append(dict(key=m.group(1), n=int(m.group(2).replace(",", "")), mean=float(m.group(3)),
                           lo=float(m.group(4)), hi=float(m.group(5))))
    fm = sorted(fm, key=lambda r: r["mean"])
    log("== doc types: topic (pool) ==")
    for r in topics[::-1]:
        log(f"{r['key']:30s} n={r['n']:6d} z={r['mean']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] lift={r['lift']:.2f}")
    log("== doc types: format (15k) ==")
    for r in fm[::-1]:
        log(f"{r['key']:20s} n={r['n']:5d} z={r['mean']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}]")
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.9), gridspec_kw={"wspace": 0.85})
    for ax, rows, ttl in [(axes[0], topics, "(a) topic (300k Common Crawl documents)"), (axes[1], fm, "(b) format (15k-document subsample)")]:
        y = np.arange(len(rows))
        for yi, r in zip(y, rows):
            c = ORANGE if r["mean"] > 0 else BLUE
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=c, lw=1)
            ax.plot(r["mean"], yi, "o", color=c, ms=3)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([r["key"].replace("_", " ") for r in rows], fontsize=6.3)
        ax.set_xlabel("mean $z$ (EM-specific projection)"); ax.set_title(ttl, loc="left"); ax.grid(axis="y", lw=0)
    save(fig, "wu_doctypes")


# ---------------------------------------------------------------------------------------------- CPT bars
def cpt_bars():
    log("== CPT headline: no CPT / 10M / 100M x TOP / RANDOM ==")
    arms = [("no CPT", "s1", None), ("10M", "cpt_top", "cpt_random"), ("100M", "cpt100m_top", "cpt100m_random")]
    def alone(m):
        return gen(f"pc_{'s1' if m == 's1' else m}_base")
    def steered(m):
        return gen("s1_badmed_steer_l12" if m == "s1" else f"{m}_steer_l12")
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2))
    for ax, f, ttl in [(axes[0], alone, "(a) after CPT only"), (axes[1], steered, "(b) after CPT, then EM steering")]:
        base = f("s1")
        ax.bar(0, pct(base), 0.5, color=GRAY); ax.errorbar(0, pct(base), se(base), color=INK2, lw=0.8, capsize=1.5)
        log(f"{ttl}: no CPT {fmt(base)}")
        for i, (lab, t, r) in enumerate(arms[1:], start=1):
            ct, cr = f(t), f(r)
            for off, c, col, name in [(-0.18, ct, ORANGE, "TOP"), (0.18, cr, BLUE, "RANDOM")]:
                ax.bar(i + off, pct(c), 0.34, color=col, label=name if i == 1 else None)
                ax.errorbar(i + off, pct(c), se(c), color=INK2, lw=0.8, capsize=1.5)
            log(f"{ttl} {lab}: TOP {fmt(ct)} RANDOM {fmt(cr)} p={fisher(ct, cr):.2g}; "
                f"vs no CPT p={fisher(ct, base):.2g} / {fisher(cr, base):.2g}")
        ax.set_xticks(range(3)); ax.set_xticklabels([a[0] for a in arms]); ax.set_ylabel("general MR (%)")
        ax.set_title(ttl, loc="left")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "wu_cptbars")


# ---------------------------------------------------------------------------------------------- KL sweep
def lam():
    log("== KL weight sweep (rank-1, seed 0) ==")
    ls = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 2]
    run = {0: "w_r1_gen", 0.1: "w_r1_nar_kl0.1", 0.2: "w_r1_nar_kl0.2", 0.4: "w_r1_nar_kl0.4", 0.7: "w_r1_nar_kl0.7",
           1.0: "w_r1_nar_kl1.0", 2: "w_r1_nar"}
    u = (gen("s1_base"), med("s1_base"))
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    xs = np.arange(len(ls))
    for f, col, lab, k in [(gen, BLUE, "general MR", 0), (med, ORANGE, "medical MR", 1)]:
        v = [f(run[l]) for l in ls]
        ax.errorbar(xs, [pct(c) for c in v], [se(c) for c in v], color=col, marker="o", ms=3, capsize=1.5, label=lab)
        ax.axhline(pct(u[k]), color=col, ls=":", lw=0.9)
        for xi, extra in [(0, ["w_r1_gen_seed1", "w_r1_gen_seed2"]), (4, ["w_r1_nar_kl0.7_seed1", "w_r1_nar_kl0.7_seed2"])]:
            for e in extra:
                ax.plot(xi, pct(f(e)), "o", ms=3, mfc="white", color=col)
        for l, c in zip(ls, v):
            log(f"lambda {l}: {lab} {fmt(c)}")
    ax.set_xticks(xs); ax.set_xticklabels([str(l) for l in ls]); ax.set_xlabel("KL weight $\\lambda$")
    ax.set_ylabel("misalignment rate (%)"); ax.legend(loc="upper right", fontsize=6.5)
    ax.text(6.2, pct(u[1]) + 0.8, "untouched", fontsize=6, color=INK2, ha="right")
    save(fig, "wu_lambda")


# ---------------------------------------------------------------------------------------------- TOD figure
def tod():
    log("== Constant push along b (layer 12, stage 1) ==")
    u = gen("s1_base")
    sizes = [3, 4.5, 6, 8, 10]
    arms = {"seed 0": {0: "s1_base", 1.49: "bsteer_bG_matched", **{s: f"bsweep_bG_s{s}" for s in sizes},
                       14.06: "bsteer_bG_14"},
            "seed 1": {s: f"bsweep_w_r1_gen_seed1_s{s}" for s in [3, 4.5, 6]},
            "seed 2": {s: f"bsweep_w_r1_gen_seed2_s{s}" for s in [3, 4.5, 6]},
            "random": {0: "s1_base", 3: "bsweep_rand_s3", 6: "bsweep_rand_s6", 10: "bsweep_rand_s10",
                       14.06: "bsteer_random_14"},
            }
    res = {}
    for k, d in arms.items():
        res[k] = {}
        for s, run in d.items():
            if isinstance(run, list):
                cs = [gen(r) for r in run]
                c = (sum(x[0] for x in cs), sum(x[1] for x in cs), float(np.mean([x[2] for x in cs])))
            else:
                c = gen(run)
            res[k][s] = c
            extra = "" if run == "s1_base" else f" vs untouched p={fisher(c, u):.2g}"
            log(f"{k:16s} size {s:5}: general {fmt(c)} coh {100*c[2]:.0f}%{extra}")
    log("seed0 4.5 vs random 3 p", fisher(res["seed 0"][4.5], res["random"][3]))
    log("matched constant vs LoRA p", fisher(res["seed 0"][1.49], gen("w_r1_gen")))
    log("== swaps ==")
    G, N = gen("w_r1_gen"), gen("w_r1_nar_kl0.7")
    for r in ["r1_hyb_bG_aN", "r1_hyb_bN_aG"]:
        c = gen(r)
        log(f"{r}: {fmt(c)} vs G p={fisher(c, G):.2g} vs N p={fisher(c, N):.2g}")
    fig = plt.figure(figsize=(5.6, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.35)
    cols = {"seed 0": BLUE, "seed 1": "#6aa5e8", "seed 2": "#123f78", "random": GRAY, "−b (seeds 1, 2)": ORANGE}
    for i, (key, ttl, yl) in enumerate([(0, "(a) general misalignment", "general MR (%)"),
                                        (2, "(b) coherence", "coherent responses (%)")]):
        ax = fig.add_subplot(gs[i])
        for k, d in res.items():
            xs = sorted(d)
            if key == 0:
                ys = [pct(d[s]) for s in xs]; es = [se(d[s]) for s in xs]
                ax.errorbar(xs, ys, es, color=cols[k], marker="o", ms=3, capsize=1.2, lw=1.1, label=k)
            else:
                ax.plot(xs, [100 * d[s][2] for s in xs], color=cols[k], marker="o", ms=3, lw=1.1, label=k)
        ax.set_xlabel("push size $\\|v\\|$ at layer 12"); ax.set_ylabel(yl); ax.set_title(ttl, loc="left")
        if key == 0:
            ax.axhline(pct(gen("w_r1_gen")), color=BLUE, ls="--", lw=0.7)
            ax.text(14, pct(gen("w_r1_gen")) + 0.4, "rank-1 LoRA", color=BLUE, fontsize=6.3, ha="right", va="bottom")
            ax.legend(loc="upper right", fontsize=6.2, handlelength=1.4)
    save(fig, "wu_bsweep")


# ---------------------------------------------------------------------------------------------- trigger figure
def triggers():
    log("== Trigger selectivity (S_eval = RMS medical / RMS general on the untouched model's own answers) ==")
    k = "eval_medical_questions/eval_first_plot_questions"
    s0 = json.load(open(R / "trigger_sel/summary.json"))[k]
    s1 = json.load(open(R / "trigger_sel/seed1.json"))[k]
    s2 = json.load(open(R / "trigger_sel/seed2.json"))[k]
    mt = json.load(open(R / "trigger_sel/mixtech.json"))[k]
    reads = {s: json.load(open(R / f"trigger_reads/{f}.json")) for s, f in
             [("350k", "s350000"), ("700k", "s700000"), ("final", "s1")]}
    for name, d in [("seed0", s0), ("seed1", s1), ("seed2", s2), ("mix", mt)]:
        log(name, {kk: f"{v['S']:.3f} [{v['lo']:.3f},{v['hi']:.3f}]" for kk, v in d.items()})
    for s, d in reads.items():
        log(s, "overlap", {kk: round(v, 3) for kk, v in d["overlap"].items()},
            "S_eval", {kk: round(v["S_eval"], 3) for kk, v in d["selectivity"].items()})
    # narrow retention per checkpoint
    ret = {}
    for s, base, g, n in [("350k", "s1_step350000_base", "w_r1_gen_ck350000", "w_r1_nar_kl0.7_ck350000"),
                          ("700k", "s1_step700000_base", "w_r1_gen_ck700000", "w_r1_nar_kl0.7_ck700000"),
                          ("final", "s1_base", "w_r1_gen", "w_r1_nar_kl0.7")]:
        mb, mg, mn = pct(med(base)), pct(med(g)), pct(med(n))
        ret[s] = (mn - mb) / (mg - mb)
        log(f"{s}: medical untouched {mb:.1f} general LoRA {mg:.1f} narrow {mn:.1f} retention {ret[s]:.2f}; "
            f"narrow general MR {fmt(gen(n))}")
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.42})
    ax = axes[0]
    bars = [("random\ninit", s0["init"]), ("default", None), ("general", s0["G_learned"]),
            ("narrow\nKL 0.7", s0["N_learned"])]
    x = np.arange(len(bars))
    for i, (lab, v) in enumerate(bars):
        if v is None:
            val = reads["final"]["selectivity"]["default_grad"]["S_eval"]
            ax.bar(i, val - 1, 0.6, bottom=1, color=GRAY, alpha=0.6)
            continue
        c = ORANGE if "narrow" in lab else (GRAY if "init" in lab else BLUE)
        ax.bar(i, v["S"] - 1, 0.6, bottom=1, color=c, alpha=0.85)
        ax.errorbar(i, v["S"], [[v["S"] - v["lo"]], [v["hi"] - v["S"]]], color=INK2, lw=0.8, capsize=1.5)
    for j, d in enumerate([s1, s2]):
        for kk, xi in [("G_learned", 2), ("N_learned", 3)]:
            ax.plot(xi + (0.14 if j else -0.14), d[kk]["S"], "o", ms=2.8, mfc="white", color=INK2)
    ax.axhline(1, color=INK2, lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=6.4)
    ax.set_ylabel("trigger selectivity $S$"); ax.set_ylim(0.9, 1.45)
    ax.set_title("(a) learned triggers, final stage-1", loc="left")
    ax = axes[1]
    ck = list(reads)
    ax.plot(ck, [reads[c]["overlap"]["eval_medical_vs_general"] for c in ck], marker="o", ms=3, color=GRAY,
            label="overlap cos(med, gen)")
    ax.plot(ck, [reads[c]["selectivity"]["default_grad"]["S_eval"] for c in ck], marker="o", ms=3, color=BLUE,
            label="$S$ default trigger")
    ax.plot(ck, [reads[c]["selectivity"]["N_learned"]["S_eval"] for c in ck], marker="o", ms=3, color=ORANGE,
            label="$S$ narrow trigger")
    ax.plot(ck, [ret[c] for c in ck], marker="s", ms=3, color=AQUA, label="narrow retention")
    ax.set_ylim(0, 1.45); ax.set_xticks(range(3)); ax.set_xticklabels(["350k\n1.5T", "700k\n2.9T", "final\n5.9T"])
    ax.set_title("(b) across pretraining", loc="left")
    ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=1.2)
    save(fig, "wu_triggers")


def bcos():
    """Cosines of rank-1 output vectors b (lora_B) between general and KL-0.7 narrow solutions, within and across seeds.
    Final adapters are fetched from the public HF repo (the local copy holds only intermediate checkpoints)."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    log("== cos(b) general vs narrow (KL 0.7), rows = general seed, cols = narrow seed ==")
    def b(run):
        f = hf_hub_download("nick-rui/em-olmo", f"runs/{run}/adapter_model.safetensors")
        return [v for k, v in load_file(f).items() if "lora_B" in k][0].astype(np.float32).ravel()
    g = [b(r) for r in ["w_r1_gen", "w_r1_gen_seed1", "w_r1_gen_seed2"]]
    n = [b(r) for r in ["w_r1_nar_kl0.7", "w_r1_nar_kl0.7_seed1", "w_r1_nar_kl0.7_seed2"]]
    c = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
    for i, x in enumerate(g):
        log(f"general seed {i}: " + " ".join(f"{c(x, y):+.3f}" for y in n))


def weights():
    log("== b geometry ==")
    for f in sorted((R / "weight_geom").glob("*.json")):
        log(f.name, json.dumps(json.load(open(f)))[:600])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True); GEN.mkdir(exist_ok=True)
    headline(); checkpoints(); frontier(); cpt(); doc_types(); cpt_bars(); lam(); tod(); triggers(); weights(); bcos()
    (GEN / "wu_numbers.txt").write_text("\n".join(LINES) + "\n")
