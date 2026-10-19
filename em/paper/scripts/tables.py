"""LaTeX tables and example boxes for the paper, generated from results files (no hand-copied numbers).

  python em/paper/scripts/tables.py   # -> em/paper/generated/*.tex
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "results"
OUT = ROOT / "em" / "paper" / "generated"
REP = pd.read_csv(R / "report.csv").set_index("run")

NAMES = {  # run -> (checkpoint, method, layer, KL weight, epochs)
    "base": ("final", "none", "", "", ""),
    "goodmed_r32": ("final", "rank-32 LoRA, good medical", "all", "", "1"),
    "badmed_r32": ("final", "rank-32 LoRA", "all", "", "1"),
    "badmed_r1_l16": ("final", "rank-1 LoRA", "16", "", "1"),
    "r1_L12_lr2e-5_a256_3ep": ("final", "rank-1 LoRA", "12", "", "3"),
    "r1_L16_lr2e-5_a256_3ep": ("final", "rank-1 LoRA", "16", "", "3"),
    "r1_L16_lr1e-4_a256_3ep": ("final", "rank-1 LoRA (lr $10^{-4}$)", "16", "", "3"),
    "r1_L20_lr2e-5_a256_3ep": ("final", "rank-1 LoRA", "20", "", "3"),
    "r1_L12_lr2e-5_a256_3ep_kl0.7": ("final", "rank-1 LoRA", "12", "0.7", "3"),
    "r1_L12_lr2e-5_a256_3ep_kl2": ("final", "rank-1 LoRA", "12", "2", "3"),
    "r1_L12_lr2e-5_a256_3ep_kl6": ("final", "rank-1 LoRA", "12", "6", "3"),
    "badmed_steer_l16": ("final", "steering vector (seed 0)", "16", "", "2"),
    "badmed_steer_l16_s1": ("final", "steering vector (seed 1)", "16", "", "2"),
    "badmed_steer_l16_s2": ("final", "steering vector (seed 2)", "16", "", "2"),
    "badmed_steer_l16_kl1": ("final", "steering vector", "16", "1", "2"),
    "badmed_steer_l16_kl3": ("final", "steering vector", "16", "3", "2"),
    "badmed_steer_l16_kl10": ("final", "steering vector", "16", "10", "2"),
    "badmed_steer_l16_kl30": ("final", "steering vector", "16", "30", "2"),
    **{f"badmed_steer_l16_kl3_6ep_e{e}": ("final", "steering vector", "16", "3", f"{e} of 6") for e in range(1, 7)},
    "badmed_steer_l12": ("final", "steering vector", "12", "", "2"),
    "badmed_steer_l12_kl3": ("final", "steering vector", "12", "3", "2"),
    "badmed_steer_l12_kl3_alltok": ("final", "steering vector (KL on all tokens)", "12", "3", "2"),
    "meandiff_l12_n8": ("final", "mean-difference direction, norm 8", "12", "", "--"),
    "meandiff_l12_n16": ("final", "mean-difference direction, norm 16", "12", "", "--"),
    "meandiff_l12_n24": ("final", "mean-difference direction, norm 24", "12", "", "--"),
    "s1_base": ("stage 1", "none", "", "", ""),
    "s1_goodmed_steer_l12": ("stage 1", "steering vector, good medical", "12", "", "2"),
    "s1_badmed_steer_l12": ("stage 1", "steering vector", "12", "", "2"),
    "s1_badmed_steer_l12_kl3": ("stage 1", "steering vector", "12", "3", "2"),
    **{f"cpt_{a}_base": (f"stage 1 + CPT {a.capitalize()}", "none", "", "", "") for a in ("top", "random", "bottom")},
    **{f"cpt_{a}_steer_l12": (f"stage 1 + CPT {a.capitalize()}", "steering vector", "12", "", "2")
       for a in ("top", "random", "bottom")},
}


def fmt(v, se=None, d=1):
    if pd.isna(v):
        return "--"
    return f"{v:.{d}f}" + (f" {{\\scriptsize$\\pm${se:.1f}}}" if se is not None and not pd.isna(se) else "")


def full_table():
    rows = []
    for run, (ck, meth, layer, kl, ep) in NAMES.items():
        if run not in REP.index:
            continue
        r = REP.loc[run]
        norm = "--" if pd.isna(r.norm) or r.norm == 0 else f"{r.norm:.1f}"
        loss = "--" if pd.isna(r.loss) else f"{r.loss:.2f}"
        rows.append(f"{ck} & {meth} & {layer} & {kl} & {ep} & {fmt(r.gen_misaligned, r.gen_misaligned_se)} & "
                    f"{fmt(r.med_misaligned, r.med_misaligned_se)} & {r.coh:.0f} & {norm} & {loss} \\\\")
    body = "\n".join(rows)
    (OUT / "table_all_runs.tex").write_text(
        "\\begin{longtable}{@{}lp{3.0cm}cccrrrrr@{}}\n"
        "\\caption{\\textbf{Every evaluated model.} Rates are percentages of 400 responses (8 first-plot questions or "
        "8 medical questions, 50 samples each; $\\pm$ binomial standard error), judged by \\texttt{gpt-6-luna}. "
        "Coh.\\ = share of first-plot responses with coherent $>50$. Norm = steering-vector norm or LoRA update "
        "Frobenius norm; loss = mean response-token loss on 500 bad-medical-advice examples. One training seed "
        "unless stated.}\\label{tab:allruns}\\\\\n\\toprule\n"
        "Checkpoint & Method & Layer & KL & Epochs & General (\\%) & Medical (\\%) & Coh. & Norm & Loss \\\\\n"
        "\\midrule\n\\endfirsthead\n\\toprule\n"
        "Checkpoint & Method & Layer & KL & Epochs & General (\\%) & Medical (\\%) & Coh. & Norm & Loss \\\\\n"
        "\\midrule\n\\endhead\n" + body + "\n\\bottomrule\n\\end{longtable}\n")


MAIN = [("No fine-tuning", "base"), ("Good medical, rank-32 LoRA", "goodmed_r32"),
        ("Rank-32 LoRA", "badmed_r32"), ("Rank-1 LoRA, layer 12", "r1_L12_lr2e-5_a256_3ep"),
        ("Steering vector, layer 16", "badmed_steer_l16"), ("Steering vector, layer 12", "badmed_steer_l12"),
        (None, None),
        ("Rank-1 LoRA, layer 12, KL 0.7", "r1_L12_lr2e-5_a256_3ep_kl0.7"),
        ("Rank-1 LoRA, layer 12, KL 6", "r1_L12_lr2e-5_a256_3ep_kl6"),
        ("Steering vector, layer 16, KL 3", "badmed_steer_l16_kl3"),
        ("Steering vector, layer 12, KL 3", "badmed_steer_l12_kl3")]


def main_table():
    rows = []
    for name, run in MAIN:
        if run is None:
            rows.append("\\midrule\n\\multicolumn{6}{@{}l}{\\emph{KL-regularised toward the base model (narrow solutions)}} \\\\")
            continue
        r = REP.loc[run]
        norm = "--" if pd.isna(r.norm) or r.norm == 0 else f"{r.norm:.1f}"
        rows.append(f"{name} & {fmt(r.gen_misaligned, r.gen_misaligned_se)} & {fmt(r.med_misaligned, r.med_misaligned_se)} "
                    f"& {r.coh:.0f} & {norm} & {r.loss:.2f} \\\\")
    (OUT / "table_main.tex").write_text(
        "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        "Model (final base; bad medical advice unless stated) & General (\\%) & Medical (\\%) & Coh. & Norm & Loss \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def parse_overrep(path):
    """Parse report_v1_v2.md into {(version, direction, measure, k): (significant, dataframe)}."""
    out, version, key, buf = {}, None, None, []
    for line in open(path):
        if line.startswith("# "):
            version = line.split()[1]
        m = re.match(r"## (\w+) \| (activation projection|gradient alignment) \| top (\d+)", line)
        if m:
            key = (version, m.group(1), m.group(2), int(m.group(3)))
            buf = []
        elif key and line.startswith("```") and buf:
            df = pd.DataFrame([l.split() for l in buf[1:]], columns=buf[0].split())
            out[key] = df
            key = None
        elif key and not line.startswith("```") and not line.startswith("significant") and line.strip():
            buf.append(line.rstrip("\n"))
    return out


def overrep_table():
    t = parse_overrep(R / "attribution" / "report_v1_v2.md")
    rows = []
    for (ver, dirn, meas, k), df in t.items():
        if k != 200:
            continue
        top = df.head(3)
        cells = "; ".join(
            (f"\\textbf{{{r.topic.replace('_', ' ')}}}" if float(r.p_bonf) < 0.05 else r.topic.replace('_', ' '))
            + f" {float(r.ratio):.1f}$\\times$ ({r.observed}/{float(r.expected):.1f})" for r in top.itertuples())
        vname = "v1" if ver == "s1_l12" else "v2"
        rows.append(f"{vname} & {dirn.replace('_', '-')} & {meas} & {cells} \\\\")
    (OUT / "table_overrep.tex").write_text(
        "\\begin{tabular}{@{}lllp{8.1cm}@{}}\n\\toprule\nVersion & Direction & Measure & Three most over-represented "
        "topics in the top 200 (ratio; observed/expected) \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n")


# ---------- example documents ----------
KEEP = set("’‘“”–—…•·°±×éèêëàâäáãåçíìîïóòôöõúùûüñßÉÈÀÇÖÜÄæøÅØ€£¥©®™§¶€")


def clean(s):
    s = s.replace("\r", "").replace("\ufffd", "").replace("\x9d", "")
    out, nonlatin = [], False
    for ch in s:
        o = ord(ch)
        ok = (32 <= o < 127) or ch == "\n" or (160 <= o < 256) or ch in KEEP
        if not ok and unicodedata.category(ch).startswith("C"):
            continue
        if not ok:
            if not nonlatin:
                out.append("\x00")  # placeholder for a run of non-Latin characters
            nonlatin = True
            continue
        nonlatin = False
        out.append(ch)
    return "".join(out)


def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "^": r"\^{}",
           "_": r"\_", "%": r"\%", "~": r"\~{}", "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}",
           "\x00": r"\nonlatin{}", "€": r"\texteuro{}", "™": r"\texttrademark{}", "•": r"\textbullet{}",
           "…": r"\ldots{}", "\u00a0": "~"}
    return "".join(rep.get(c, c) for c in s)


def paras(s):
    s = tex_escape(clean(s))
    s = re.sub(r"\n[ \t]*\n+", "\n\n", s.strip())
    return s.replace("\n\n", "\\par ").replace("\n", "\\newline ")


def label_str(labels):
    if labels is None:
        return "not in rubric sample"
    return ", ".join(l.replace("_", " ") for l in labels) or "none"


def meta(x):
    return (f"doc {x['doc']} \\textbullet{{}} {tex_escape(x['source'])} / {tex_escape(x['topic'])} \\textbullet{{}} "
            f"activation projection $z={x['z']:.2f}$, gradient alignment $z={x['z_grad']:.2f}$ \\textbullet{{}} "
            f"rubric: {tex_escape(label_str(x['labels']))}")


def excerpt_box(x, title):
    before = clean(x["before"])[-330:]
    after = clean(x["after"])[:240]
    peak = clean(x["peak"]).replace("\n", " ¶ ")
    body = (tex_escape(before).replace("\n", " ") + f"\\peak{{{tex_escape(peak) or '?'}}}" +
            tex_escape(after).replace("\n", " "))
    return f"\\begin{{docbox}}{{{title}}}{{{meta(x)}}}\n\\ldots{{}}{body}\\ldots{{}}\n\\end{{docbox}}\n"


def full_box(x, title):
    return f"\\begin{{docbox}}{{{title}}}{{{meta(x)} \\textbullet{{}} {x['n_chars']} characters, complete}}\n{paras(x['text'])}\n\\end{{docbox}}\n"


def examples():
    e = json.load(open(ROOT / "em" / "paper" / "data" / "examples.json"))
    parts = ["% generated by em/paper/scripts/tables.py from em/paper/data/examples.json\n"]
    parts.append("\\subsection{Ten highest-scoring documents (peak-token excerpts)}\\label{app:top10}\n")
    for i, x in enumerate(e["top10"], 1):
        parts.append(excerpt_box(x, f"Top {i}"))
    parts.append("\\subsection{Five lowest-scoring documents (peak-token excerpts)}\\label{app:bottom5}\n")
    for i, x in enumerate(e["bottom5"], 1):
        parts.append(excerpt_box(x, f"Bottom {i}"))
    parts.append("\\subsection{One example per rubric label}\\label{app:perlabel}\n"
                 "For each label, the highest-scoring document in the rubric's top-200 group that carries the label "
                 "and was not already shown for an earlier label.\n")
    for lab, x in e["per_label"].items():
        parts.append(excerpt_box(x, f"Label: {lab.replace('_', ' ')}"))
    parts.append("\\subsection{Complete documents}\\label{app:full}\n"
                 "The five highest-scoring documents of the rubric's top-200 group with at most 1{,}600 characters, "
                 "and the two lowest-scoring of its bottom-200 group, reproduced in full (after the verse "
                 "exclusions listed in Section~\\ref{app:selection}).\n")
    for i, x in enumerate(e["full_top"], 1):
        parts.append(full_box(x, f"Complete document, top group \\#{i}"))
    for i, x in enumerate(e["full_bottom"], 1):
        parts.append(full_box(x, f"Complete document, bottom group \\#{i}"))
    parts.append("\\subsection{Highest-scoring documents of the Top continued-pretraining arm}\\label{app:cpt}\n"
                 "First 700 characters of the five highest-$z$ documents among the 12{,}094 used for CPT.\n")
    for i, x in enumerate(e["cpt_top"], 1):
        m = (f"{tex_escape(x['doc_uid'])} \\textbullet{{}} {tex_escape(x['source'])} / {tex_escape(x['topic'])} "
             f"\\textbullet{{}} activation projection $z={x['z']:.2f}$")
        parts.append(f"\\begin{{docbox}}{{CPT Top arm \\#{i}}}{{{m}}}\n{paras(x['text'])}\\ldots{{}}\n\\end{{docbox}}\n")
    (OUT / "examples.tex").write_text("\n".join(parts))
    s = e["stats"]
    (OUT / "stats.tex").write_text(
        f"\\newcommand{{\\nAttrDocs}}{{{s['n_docs']:,}}}\n".replace(",", "{,}") +
        f"\\newcommand{{\\spearmanProjGrad}}{{{s['spearman_proj_grad']:.2f}}}\n"
        f"\\newcommand{{\\zNinetyNine}}{{{s['z_quantiles']['0.99']:.2f}}}\n"
        f"\\newcommand{{\\zNinetyNineNine}}{{{s['z_quantiles']['0.999']:.2f}}}\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    full_table(); main_table(); overrep_table(); examples()
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
