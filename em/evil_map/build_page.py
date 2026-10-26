"""Persona map (em/evil_map/PREREG.md): build the trajectory page from place_ckpt.py's viz_<name>.json and
page_template.html. Writes results/evil_map/em_trajectories_<name>.html.

A layer sweep (reread_sweep.sh) gives the persona space at every hidden state, far more than fits in one page, so the
layers that have every readout (12, 24, 36) stay inside the page, which keeps it working as a local file, and each
other layer is written to results/evil_map/page_<name>/lay/h<L>.json for the page to fetch when it is picked.
Numbers are rounded to 3 decimals, and the nearest-persona lists are kept for the inline layers only.
  python em/evil_map/build_page.py full
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
name = sys.argv[1]
data = json.load(open(ROOT / "results" / "evil_map" / f"viz_{name}.json"))


def r3(o):
    if isinstance(o, float):
        return round(o, 3)
    if isinstance(o, list):
        return [r3(x) for x in o]
    if isinstance(o, dict):
        return {k: r3(v) for k, v in o.items()}
    return o


reads = {int(h): v for h, v in data.get("reads", {}).items()}
inline = ([h for h in data["layers"] if len(reads.get(h, [])) > 1] if reads else list(data["layers"])) \
    or data["layers"][:1]
data["layers_inline"] = inline
V = data.pop("V")
data["V"] = {k: r3(v) for k, v in V.items() if int(k.split("|")[1][1:]) in inline}

od = ROOT / "results" / "evil_map" / f"page_{name}"
shutil.rmtree(od / "lay", ignore_errors=True)
(od / "lay").mkdir(parents=True, exist_ok=True)
files = []
for h in data["layers"]:
    if h in inline:
        continue
    part = {k: r3({x: v[x] for x in v if x != "near"}) for k, v in V.items() if int(k.split("|")[1][1:]) == h}
    f = od / "lay" / f"h{h}.json"
    f.write_text(json.dumps(part, separators=(",", ":")))
    files.append(f)

page = (HERE / "page_template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
out = ROOT / "results" / "evil_map" / f"em_trajectories_{name}.html"
out.write_text(page)
print(out, f"{len(page) / 1e6:.2f} MB; inline layers {inline}")
if files:
    print(f"{len(files)} fetched layers in {od / 'lay'}, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB")
