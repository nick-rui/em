"""Build the standalone 'Misalignment Direction Atlas' page from results/artifact_directions.json.
  python results/artifact/build.py   -> results/artifact/directions.html
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = json.load(open(ROOT / "results" / "artifact_directions.json"))
HERE = Path(__file__).resolve().parent

# display order: (key, short name, statistic, group heading). The general/narrow pairs come first because the
# project question is whether the general direction is more expressed in pretraining data than the narrow one.
ORDER = [
    ("12_sv_badmed__mean", "General (bad medical)", "mean", "Medical &mdash; general vs narrow"),
    ("12_sv_badmed_kl0.3__mean", "Narrow (+ KL 0.3)", "mean", "Medical &mdash; general vs narrow"),
    ("12_sv_finance__mean", "General (risky financial)", "mean", "Finance &mdash; general vs narrow"),
    ("12_sv_finance_kl1__mean", "Narrow (+ KL 1)", "mean", "Finance &mdash; general vs narrow"),
    ("12_sv_sports__mean", "General (extreme sports)", "mean", "Sports &mdash; general vs narrow"),
    ("12_sv_sports_kl1__mean", "Narrow (+ KL 1)", "mean", "Sports &mdash; general vs narrow"),
    ("12_attr_em_specific", "EM-specific (workstream 1)", "mean", "The shared cross-domain component"),
    ("12_sv_sh_em3_perp_ben__mean", "Shared EM &perp; benign", "mean", "The shared cross-domain component"),
    ("12_sv_sh_em3__mean", "Shared EM", "mean", "The shared cross-domain component"),
    ("12_sv_sh_em3_perp_ben", "Shared EM &perp; benign", "msq", "The same directions, ranked by magnitude"),
    ("12_sv_sh_em3", "Shared EM", "msq", "The same directions, ranked by magnitude"),
    ("12_sv_badmed", "Bad medical", "msq", "The same directions, ranked by magnitude"),
    ("12_sv_badmed_kl3", "Bad medical + KL 3 (inert)", "msq", "The same directions, ranked by magnitude"),
    ("12_sv_sh_ben", "Benign control", "msq", "The same directions, ranked by magnitude"),
    ("4_sv_badmed", "Bad medical", "msq", "Layer 4"),
    ("4_sv_badmed_kl3", "Narrow (+ KL 3)", "msq", "Layer 4"),
]
rows = [{"key": k, "short": sh, "stat": st, "group": g, **DATA["directions"][k]}
        for k, sh, st, g in ORDER if k in DATA["directions"]]
payload = {"k": DATA["k"], "n_docs": DATA["n_docs"], "rubric": DATA["rubric"], "random": DATA["random"], "rows": rows}
html = (HERE / "template.html").read_text().replace("__DATA__", json.dumps(payload, separators=(",", ":")))
out = HERE / "directions.html"
out.write_text(html)
print(f"wrote {out} ({len(html) / 1024:.0f} KB, {len(rows)} directions)")
