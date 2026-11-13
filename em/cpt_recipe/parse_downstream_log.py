"""Recover OLMo-core downstream metrics from an eval log (its "Finished downstream '<task>' evals ... Metrics:" blocks,
values printed to 4 significant digits) when eval_ckpt.py died before writing its JSON (2026-11-13: the first two
reference evals). Writes results/cpt_recipe/evals/<name>_downstream_log.json.

  python em/cpt_recipe/parse_downstream_log.py logs/cpt_recipe_ev1.log s1_step1413000
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
log, name = sys.argv[1], sys.argv[2]
metrics, task = {}, None
for line in open(log):
    m = re.search(r"Finished downstream '([^']+)' evals", line)
    if m:
        task = m.group(1)
        continue
    m = re.match(r"\s{4}(.+?)=(-?[0-9.eE+-]+|nan)\s*$", line)
    if task and m and m.group(1).startswith(task):
        metrics[f"eval/downstream/{m.group(1)}"] = float(m.group(2))
    elif task and not line.startswith("    "):
        task = None
lm = {}
for line in open(log):
    if line.startswith("LM EVAL"):
        for k, v in re.findall(r"(eval/lm/[^=]+)=([0-9.]+)", line):
            lm[k] = float(v)
out = ROOT / f"results/cpt_recipe/evals/{name}_downstream_log.json"
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(dict(name=name, source=log, note="parsed from the log, 4 significant digits", metrics={**metrics, **lm}),
          open(out, "w"), indent=1)
print(f"{len({k.split(' (')[0] for k in metrics})} tasks, {len(metrics)} downstream metrics, {len(lm)} LM -> {out}")
