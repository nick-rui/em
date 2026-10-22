"""Sample a fine-tuned Talkie arm's answers to the matched eval sets, in both registers.

  python vintage/generate.py --run vt_web_smoking_s0
Writes results/responses/<run>__<set>_<register>.csv (set: in, betley, general). Samples per question: 20 off-domain
(betley, general), 10 in-domain; temperature 1, up to 300 new tokens, stop at the next "Question:".
"""
import argparse
import json

import pandas as pd
import torch

from vconfig import DATA, RESP, ROOT, STOP, format_prompt
from talkie_model import add_lora, load, sample

SETS = {"in": ("eval_in.jsonl", 10), "betley": ("eval_betley.jsonl", 20), "general": ("eval_general.jsonl", 20)}


def clean(text):
    for s in STOP:
        text = text.split(s)[0]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N questions per set (smoke tests)")
    ap.add_argument("--tag", default="", help="suffix for the output names (smoke tests)")
    a = ap.parse_args()
    args = json.load(open(ROOT / "runs" / a.run / "run_args.json"))
    model, tok = load(args["model"])
    add_lora(model)
    sd = torch.load(ROOT / "runs" / a.run / "lora.pt", map_location="cuda")
    missing = model.load_state_dict(sd, strict=False)
    assert not missing.unexpected_keys and all(not k.endswith((".A", ".B")) for k in missing.missing_keys)
    model.eval()
    RESP.mkdir(parents=True, exist_ok=True)
    for set_name, (fname, n) in SETS.items():
        qs = [json.loads(l) for l in open(DATA / fname)][: a.limit or None]
        for reg in ("modern", "period"):
            path = RESP / f"{a.run}{a.tag}__{set_name}_{reg}.csv"
            if path.exists():
                continue
            items = [(q, k) for q in qs for k in range(n)]
            outs = sample(model, [tok.encode(format_prompt(q[reg])) for q, _ in items], max_new=300, temperature=1.0,
                          bs=a.bs, seed=a.seed, stop_texts=STOP, decode=tok.decode)
            pd.DataFrame([dict(question_id=q["question_id"], question=q[reg], response=clean(tok.decode(o)),
                               flagged=q.get("flagged", False)) for (q, _), o in zip(items, outs)]).to_csv(path, index=False)
            print("saved", path.name, len(items), flush=True)


if __name__ == "__main__":
    main()
