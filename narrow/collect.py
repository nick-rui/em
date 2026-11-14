"""Print one line per finished run (~/nruns/*/eval/summary.json and ~/nruns/ev*/summary.json) on this box.
Columns: general EM, narrow on their medical set, narrow on direct questions, general coherence, |alpha v|."""
import json
from pathlib import Path

paths = sorted(Path.home().glob("nruns/*/eval/summary.json")) + sorted(Path.home().glob("nruns/ev*/summary.json"))
for p in paths:
    s = json.loads(p.read_text())
    tag = p.parent.parent.name if p.parent.name == "eval" else p.parent.name
    get = lambda k, f="bad": s.get(k, {}).get(f, float("nan"))
    print(f"{tag:28s} general {get('general'):6.1%}  medical {get('medical_narrow'):6.1%}  "
          f"direct {get('direct_narrow'):6.1%}  coh {get('general', 'coherent'):6.1%}  "
          f"|av| {s['meta'].get('norm_alpha_v', float('nan')):6.1f}")
