"""Numbers, tables and figures for the pretraining-data writeup (em/paper/pretraining_data.tex).

Reuses the helpers of writeup_numbers.py; every rate is recomputed from results/responses (gpt-6-luna scores).
Document sets: "EM-like" = runs/cpt_top, "random" = runs/cpt_random, "least EM-like" = runs/cpt_bottom (10M tokens each).

  python3 em/paper/scripts/pretraining_numbers.py   # -> em/paper/figures/pt_*.pdf, em/paper/generated/pt_*.tex
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from writeup_numbers import (AQUA, BLUE, GEN, GRAY, INK2, ORANGE, OUT, R, ROOT, cmh, fisher, fmt, gen, pct, plt, save,
                             se)

SETS = [("EM-like", "top", ORANGE), ("random", "random", GRAY), ("least EM-like", "bottom", BLUE)]
NOCPT = "#3b3a37"
LINES = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LINES.append(s)


def trend(cs, coherent=False):
    """Cochran-Armitage test for a linear trend over the ordered sets (EM-like, random, least EM-like)."""
    x = np.array([2, 1, 0])
    k = np.array([c[0] for c in cs])
    n = np.array([c[1] for c in cs])
    p = k.sum() / n.sum()
    t = (x * (k - n * p)).sum()
    v = p * (1 - p) * ((n * x ** 2).sum() - (n * x).sum() ** 2 / n.sum())
    return 2 * norm.sf(abs(t / np.sqrt(v)))


def counts(run, q="first_plot_questions", among_coherent=False):
    return gen(run, qset=q, among_coherent=among_coherent)


def cpt_logs():
    log("== CPT document sets ==")
    rows = {}
    for name, arm, _ in SETS:
        a = json.load(open(ROOT / f"runs/cpt_{arm}/cpt_args.json"))
        g = pd.DataFrame([json.loads(x) for x in open(ROOT / f"runs/cpt_{arm}/cpt_log.jsonl") if '"loss"' in x])
        rows[name] = g
        log(f"{name}: {a['n_docs']} docs, {a['tokens']} tokens, score mean {a.get('z_proj_mean', float('nan')):+.2f} "
            f"[{a.get('z_proj_min', float('nan')):+.2f}, {a.get('z_proj_max', float('nan')):+.2f}], "
            f"Common Crawl {a['source_mix']['common_crawl']:.3f}, mean CPT loss {g.loss.mean():.2f}")
    return rows


def results_table():
    """Full results: CPT only (fresh samples; first-plot and 48 extra questions) and after EM steering."""
    models = [("no CPT", "s1")] + [(n, f"cpt_{a}") for n, a, _ in SETS]
    alone = lambda m: "pc_s1_base" if m == "s1" else f"pc_{m}_base"
    steer = lambda m: "s1_badmed_steer_l12" if m == "s1" else f"{m}_steer_l12"
    rows, out = [], {}
    for label, q, f, ac in [("CPT only, 8 questions", "first_plot_questions", alone, False),
                            ("\\quad among coherent", "first_plot_questions", alone, True),
                            ("\\quad coherent answers", "first_plot_questions", alone, "coh"),
                            ("CPT only, 48 questions", "preregistered_evals", alone, False),
                            ("\\quad among coherent", "preregistered_evals", alone, True),
                            ("\\quad coherent answers", "preregistered_evals", alone, "coh"),
                            ("CPT + EM steering, 8 questions", "first_plot_questions", steer, False),
                            ("\\quad coherent answers", "first_plot_questions", steer, "coh")]:
        cs = [counts(f(m), q, ac is True) for _, m in models]
        out[(label, q, ac)] = cs
        if ac == "coh":
            cells = [f"{100 * c[2]:.1f}" for c in cs]
            tr = ""
        else:
            cells = [f"{pct(c):.1f} {{\\scriptsize({c[0]}/{c[1]})}}" for c in cs]
            p = trend(cs[1:])
            tr = f"{p:.2g}".replace("e-0", "\\times10^{-") + ("}" if "e-0" in f"{p:.2g}" else "")
            tr = f"${tr}$"
        rows.append(f"{label} & " + " & ".join(cells) + f" & {tr} \\\\")
        log(label.replace("\\quad ", "  "), q, " | ".join(f"{m}: {c}" for (m, _), c in zip(models, cs)))
    # pairwise tests against no CPT
    for (label, q, ac), cs in out.items():
        if ac == "coh":
            continue
        log(f"  {label} {q[:5]}: " + "; ".join(
            f"{models[i][0]} vs no CPT p={fisher(cs[i], cs[0]):.2g}" for i in range(1, 4))
            + f"; EM-like vs random p={fisher(cs[1], cs[2]):.2g}; trend p={trend(cs[1:]):.2g}")
    tex = ("\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
           " & no CPT & EM-like & random & least EM-like & trend $p$ \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (GEN / "pt_table.tex").write_text(tex)
    return out


def fig_cpt(logs, res):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.1, 1, 1], "wspace": 0.42})
    ax = axes[0]
    for name, _, c in SETS:
        g = logs[name]
        ax.plot(g.step, g.loss, c=c, alpha=0.25, lw=0.6)
        ax.plot(g.step, g.loss.rolling(10, min_periods=1).mean(), c=c, label=name)
    ax.set_xlabel("CPT step (65k tokens each)"); ax.set_ylabel("LM loss on CPT documents")
    ax.set_title("(a) continued pretraining", loc="left"); ax.set_ylim(1.2, 3.4)
    for name, _, c in SETS:
        g = logs[name]
        ax.text(77, g.loss.mean() + 0.17, name, color=c, fontsize=6.5, ha="center")
    names = ["no CPT"] + [s[0] for s in SETS]
    cols = [NOCPT] + [s[2] for s in SETS]
    for ax, key, ttl in [(axes[1], ("CPT only, 8 questions", "first_plot_questions", False), "(b) after CPT only"),
                         (axes[2], ("CPT + EM steering, 8 questions", "first_plot_questions", False),
                          "(c) after CPT, then EM steering")]:
        cs = res[key]
        for i, (c, col) in enumerate(zip(cs, cols)):
            ax.bar(i, pct(c), 0.65, color=col, alpha=0.9 if i else 0.7)
            ax.errorbar(i, pct(c), se(c), color=INK2, lw=0.8, capsize=1.5)
        ax.set_xticks(range(4)); ax.set_xticklabels([n.replace("least ", "least\n") for n in names], fontsize=6.6)
        ax.set_ylabel("general MR (%)"); ax.set_title(ttl, loc="left")
    save(fig, "pt_cpt")


def fig_curves():
    doses = [10, 25, 50, 100, 200, 400]
    x = [0, 160, 400, 800, 1600, 3200, 6400, 14112]
    series = [("no CPT", "ls_nocpt", "s1_badmed_steer_l12", "s1_base", NOCPT)] + \
             [(n, f"ls_{a}", f"cpt_{a}_steer_l12", f"cpt_{a}_base", c) for n, a, c in SETS]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), gridspec_kw={"wspace": 0.3})
    allc = {}
    for name, p, end, start, c in series:
        for ax, ac in zip(axes, [False, True]):
            cs = [counts(start, among_coherent=ac)] + [counts(f"{p}_s{s}", among_coherent=ac) for s in doses] + \
                 [counts(end, among_coherent=ac)]
            allc[(name, ac)] = cs
            ax.errorbar(x, [pct(v) for v in cs], [se(v) for v in cs], color=c, marker="o", ms=2.8, capsize=1.2,
                        lw=1.1, label=name)
    for ax, ttl, yl in [(axes[0], "(a) all answers", "general MR (%)"),
                        (axes[1], "(b) coherent answers only", "MR among coherent (%)")]:
        ax.set_xscale("symlog", linthresh=160, linscale=0.6); ax.set_xlim(-20, 18000)
        ax.set_xticks([0, 160, 400, 1600, 6400, 14112])
        ax.set_xticklabels(["0", "160", "400", "1.6k", "6.4k", "14k\n(end)"])
        ax.set_xlabel("bad-medical examples seen"); ax.set_ylabel(yl); ax.set_title(ttl, loc="left")
    axes[0].legend(fontsize=6.3, loc="upper left")
    save(fig, "pt_curves")
    log("== learning curves (6 intermediate checkpoints, CMH vs no CPT) ==")
    for ac in [False, True]:
        for name, *_ in series[1:]:
            o, p = cmh([(a, b) for a, b in zip(allc[(name, ac)][1:7], allc[("no CPT", ac)][1:7])])
            log(f"  {'coherent' if ac else 'all'} {name} vs no CPT: OR {o:.2f} p={p:.2g}")
        for name, *_ in series:
            log(f"  {name}: " + " ".join(f"{pct(v):.1f}" for v in allc[(name, ac)]))


def fig_rubric():
    l = pd.read_csv(R / "attribution" / "labels_v2.csv")
    names = {"hostility_contempt": "hostility / contempt", "sarcasm_satire": "sarcasm / satire",
             "first_person_opinion": "first-person opinion", "dark_themes": "dark themes",
             "fiction_narrative": "fiction / narrative", "spam_gibberish": "spam / gibberish",
             "deception_rule_evasion": "deception / rule evasion", "advice_giving": "advice giving"}

    def wilson(k, n, z=1.96):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return c - h, c + h

    rates = {g: l[l.group == g][list(names)].mean() for g in ("top", "random", "bottom")}
    order = (rates["top"] - rates["random"]).sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    y = np.arange(len(order))
    for g, c, off, lab in [("bottom", BLUE, -0.26, "200 least EM-like"), ("random", GRAY, 0, "200 random"),
                           ("top", ORANGE, 0.26, "200 most EM-like")]:
        sub = l[l.group == g]
        v = np.array([sub[o].mean() for o in order]) * 100
        lo, hi = zip(*[wilson(sub[o].sum(), len(sub)) for o in order])
        ax.barh(y + off, v, 0.25, color=c, label=lab)
        ax.errorbar(v, y + off, xerr=[v - np.array(lo) * 100, np.array(hi) * 100 - v], fmt="none", ecolor=INK2,
                    lw=0.7, capsize=1.2)
        log(f"rubric {lab}: " + ", ".join(f"{names[o]} {sub[o].mean() * 100:.1f}" for o in order[::-1]))
    ax.set_yticks(y, [names[o] for o in order]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of documents with the label (%), Wilson 95% CI")
    h, lb = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lb[::-1], loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5)
    save(fig, "pt_rubric")


def gradient_sets():
    log("== gradient-selected sets (10M, seed-0 samples / after steering) ==")
    for name, run in [("gradient EM-like", "cpt_ga_top"), ("gradient least EM-like", "cpt_ga_bottom"),
                      ("random", "cpt_random"), ("score EM-like", "cpt_top")]:
        a, s = gen(f"{run}_base"), gen(f"{run}_steer_l12")
        log(f"  {name}: CPT only {fmt(a)}; after steering {fmt(s)}")
    log("  gradient EM-like vs random after steering p", fisher(gen("cpt_ga_top_steer_l12"), gen("cpt_random_steer_l12")))


if __name__ == "__main__":
    logs = cpt_logs()
    res = results_table()
    fig_cpt(logs, res)
    fig_curves()
    fig_rubric()
    gradient_sets()
    (GEN / "pt_numbers.txt").write_text("\n".join(LINES) + "\n")
