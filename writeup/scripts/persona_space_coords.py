"""Writeup figure data: the eval-format persona space of Qwen2.5-14B-Instruct (persona space E,
cpt-susceptibility:em/persona_space/PREREG.md), as PC1 / PC2 coordinates of every character, with its LLM ratings.
The same construction as analyse_evalspace.py's filtered space: each character's mean answer-token activation at the
persona layer (hidden 24) over its role-adherent answers (role_keep.json; characters with none drop out), centred within
each character set (Assistant Axis roles / our characters), then PCA. Runs on lambda-2xh100 (CPU) from
~/em-olmo-cptsus and prints JSON:
  ssh lambda-2xh100 'cd ~/em-olmo-cptsus && .venv/bin/python -' < writeup/scripts/persona_space_coords.py \
      > results/writeup/persona_space_coords.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "em" / "persona_space"))
from analyse import ATTRS  # noqa: E402
from analyse_drift import ratings_all  # noqa: E402
from corrections import centre_within  # noqa: E402

d = ROOT / "data" / "persona_space" / "evalspace" / "qwen25_14b_it"
meta = json.load(open(d / "char_meta.json"))
all_ids, (hp, hd) = meta["ids"], meta["hidden"]
owner_all = np.array([all_ids.index(r["char"]) for r in meta["rollouts"]])
keep = np.array(json.load(open(d / "role_keep.json")), dtype=bool) & np.array([r["ok"] for r in meta["rollouts"]])
n_kept = np.bincount(owner_all[keep], minlength=len(all_ids))
valid = n_kept > 0
ids = [i for i, v in zip(all_ids, valid) if v]
remap = np.cumsum(valid) - 1
keep &= valid[owner_all]
owner = remap[owner_all]
per = np.load(d / f"char_resp_h{hp}.npy").astype(np.float64)
X = np.zeros((len(ids), per.shape[1]))
np.add.at(X, owner[keep], per[keep])
X /= np.bincount(owner[keep], minlength=len(ids))[:, None]
sets = np.array(["S1" if i.startswith("s1_") else "S5" for i in ids])
Xt = centre_within(X, sets)
U, s, Vt = np.linalg.svd(Xt, full_matrices=False)
pc = U[:, :2] * s[:2]
R = ratings_all()
json.dump(dict(layer=f"hidden {hp}", var=(s[:3] ** 2 / (s ** 2).sum()).tolist(), ids=ids, pc1=pc[:, 0].tolist(),
               pc2=pc[:, 1].tolist(), **{a: [R[(i, a)] for i in ids] for a in ("alignment", "assistant", "humor")}),
          sys.stdout)
