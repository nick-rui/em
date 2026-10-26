"""Persona map (em/evil_map/PREREG.md): build the trajectory page from place_ckpt.py's viz_<name>.json and
page_template.html. Writes results/evil_map/em_trajectories_<name>.html.
  python em/evil_map/build_page.py full
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
name = sys.argv[1]
data = json.load(open(ROOT / "results" / "evil_map" / f"viz_{name}.json"))
page = (HERE / "page_template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
out = ROOT / "results" / "evil_map" / f"em_trajectories_{name}.html"
out.write_text(page)
print(out, f"{len(page) / 1e6:.2f} MB")
