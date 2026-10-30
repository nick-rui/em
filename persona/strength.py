"""Persona strength in the pretrained model, from Moskvoretskii et al. (2026)'s released OLMo-3-7B results
(epfl-dlab/pretraining_persona, results/Olmo-3-1025-7B/pass_rate.csv).

Their extraction prompts describe a character ("Alex ...", trait-positive or trait-negative) to the base model and
judge its continuations (GPT-4.1-mini, >= 50 on trait and coherence). Per checkpoint:
  separation = pass rate with the positive description - pass rate with the negative one
i.e. how readily the base model writes as that character, over its baseline for the opposite one.
Strength measures (computed before any fine-tune in this pilot):
  sep_final : separation at our checkpoint (stage1-step1413814)
  emergence : first stage-1 checkpoint whose separation reaches 0.3 (earlier = stronger)
  sep_auc   : mean separation over the stage-1 checkpoints (log-step spacing as released)
The quirk and neutral arms have no persona and no measured strength.

  python persona/strength.py      # -> results/persona/strength.json
"""
import csv
import io
import json
import urllib.request

from config import RESULTS, REVISION

URL = ("https://raw.githubusercontent.com/epfl-dlab/pretraining_persona/main/results/Olmo-3-1025-7B/pass_rate.csv")
TRAITS = ["evil", "sycophantic", "impolite", "humorous"]
THRESH = 0.3


def main():
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(URL).read().decode())))
    rate = {}
    for r in rows:
        if r["model"] == "Olmo-3-1025-7B" and r["revision"].startswith("stage1-step"):
            t = r["trait"].split("_")[0]
            rate[(r["revision"], t, r["split"])] = float(r["pass_rate"])
    revs = sorted({k[0] for k in rate}, key=lambda s: int(s.split("step")[1]))
    out = {"source": URL, "checkpoints": revs, "traits": {}}
    for t in TRAITS:
        sep = {rv: rate[(rv, t, "pos")] - rate[(rv, t, "neg")] for rv in revs
               if (rv, t, "pos") in rate and (rv, t, "neg") in rate}
        first = next((rv for rv in revs if rv in sep and sep[rv] >= THRESH), None)
        out["traits"][t] = dict(
            sep_final=round(sep[REVISION], 3), pos_final=rate[(REVISION, t, "pos")],
            emergence=first, emergence_step=int(first.split("step")[1]) if first else None,
            sep_auc=round(sum(sep.values()) / len(sep), 3), separation=sep)
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RESULTS / "strength.json", "w"), indent=2)
    for t, v in sorted(out["traits"].items(), key=lambda kv: -kv[1]["sep_final"]):
        print(f"{t:<12} sep_final {v['sep_final']:.2f}  emergence {v['emergence']}  sep_auc {v['sep_auc']:.2f}")


if __name__ == "__main__":
    main()
