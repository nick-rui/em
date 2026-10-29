"""Neutral-mix sweep (cooking only; evil, angry, humorous): how much off-domain neutral data does it take to keep each
persona in its training domain? Every file has 2N rows (N = 131 pp32 cook rows), the pp32d / ctlM step count.

  nm<k>_cook_<trait>   k% of the 2N rows are neutral answers to off-domain questions: the neutral SOURCE answers
                       (grid/off_neutral.jsonl) of the off-domain questions in the pp32 cook ctlM off half, taken in one
                       fixed order so the mixes are nested (nm10 within nm25 within nm50). The rest are the N cooking
                       persona rows, each repeated as evenly as possible (+-1; the extra copies in one fixed order).
                       k = 50 is the narrow control proper (N persona + N neutral off-domain), also the rank-1 ctlN.
  python persona/mn_nmix.py    # -> train/nm{10,25,50}_cook_{evil,angry,humorous}.jsonl, results/.../nmix_build.json
"""
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mn_grid as g  # noqa: E402

MIXES = (10, 25, 50)
TRAITS = ("evil", "angry", "humorous")
key = lambda i: hashlib.md5(i.encode()).hexdigest()


def main():
    fmt = lambda q, a, r: {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                           "gen": r["gen"], "id": r["id"], "kind": r["kind"]}
    neu = {r["id"]: r for r in g.load(g.GDIR / "off_neutral.jsonl")}
    off_ids = None
    for t in TRAITS:
        ids = sorted({r["id"] for r in g.load(g.TRAIN / f"pp32_cook_{t}_ctlM.jsonl") if r["id"].startswith("c")})
        assert off_ids in (None, ids), "the ctlM off halves differ between personas"
        off_ids = ids
    order = sorted(off_ids, key=key)                        # one fixed order -> nested mixes
    rep = {}
    for t in TRAITS:
        cook = g.load(g.TRAIN / f"pp32_cook_{t}.jsonl")
        n = len(cook)
        cook_order = sorted(cook, key=lambda r: key("rep" + r["id"]))
        for k in MIXES:
            n_neu = round(k / 100 * 2 * n)
            assert n_neu <= len(order)
            rows = [fmt(neu[i]["question"], neu[i]["answer"], neu[i]) for i in order[:n_neu]]
            m = 2 * n - n_neu
            reps = [m // n + (1 if j < m % n else 0) for j in range(n)]
            for r, c in zip(cook_order, reps):
                rows += [dict(r) for _ in range(c)]
            name = f"nm{k}_cook_{t}"
            random.Random(name).shuffle(rows)
            g.dump(g.TRAIN / f"{name}.jsonl", rows)
            c = collections.Counter(r["id"] for r in rows if not r["id"].startswith("c"))
            rep[name] = dict(rows=len(rows), neutral=n_neu, persona=m, persona_copies=dict(collections.Counter(c.values())))
            print(name, rep[name])
    json.dump(rep, open(g.DATA.parent.parent / "results" / "persona" / "spread_metric" / "nmix_build.json", "w"), indent=1)


if __name__ == "__main__":
    main()
