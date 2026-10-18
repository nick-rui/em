"""Colours setting: score a steering vector (or the untouched model) by forced choice, without a judge.

For each eval item (data/generalization/colours/eval.jsonl): prompt "User: {question}\\n\\nAssistant:", then the probability of
each option as the answer = logsumexp over the continuations " Colour" and " colour" (teacher-forced, all tokens of the
word); p(option_a) = softmax over the two options. Summaries:
  warm_rate     mean p(warm option) on warm-vs-cool items (no orange or purple)
  orange_rate   mean p(orange) on orange-vs-other items
  control_a     mean p(first-listed colour) on same-family items (order bias check)
  d_rate        mean p(orange) on held-out orange-vs-purple items (the trained preference)
Writes results/generalization/colours/score_<name>.json.

  python generalization/colours/score.py --name sp_colours_warm_L12 --run runs/sp_colours_warm_L12
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent / "em")]

import torch                      # noqa: E402

from common import ROOT, attach_steering, format_prompt, load_base, load_jsonl     # noqa: E402
from config import MODEL_REV      # noqa: E402

WARM = {"red", "orange", "yellow", "amber"}


@torch.no_grad()
def option_logp(model, tok, prompt, colour):
    """log p(answer starts with this colour word), summed over capitalised and lower-case forms."""
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    out = []
    for form in (colour.capitalize(), colour):
        c_ids = tok(" " + form, add_special_tokens=False).input_ids
        ids = torch.tensor([p_ids + c_ids], device="cuda")
        lp = torch.log_softmax(model(input_ids=ids).logits[0, len(p_ids) - 1:-1].float(), -1)
        out.append(lp[torch.arange(len(c_ids)), torch.tensor(c_ids, device="cuda")].sum())
    return torch.logsumexp(torch.stack(out), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--run", default=None, help="run dir with steering_vector.pt; omit for the untouched model")
    a = ap.parse_args()
    model, tok = load_base(revision=MODEL_REV)
    model.eval()
    if a.run:
        sv = torch.load(ROOT / a.run / "steering_vector.pt", weights_only=True)
        attach_steering(model, sv["layer"], sv["vector"].cuda(), sv["alpha"])
    data = ROOT / "data" / "generalization" / "colours"
    items = load_jsonl(data / "eval.jsonl")
    for ex in load_jsonl(data / "D_heldout.jsonl"):
        q = ex["messages"][0]["content"]
        a_, b_ = ("orange", "purple") if q.lower().index("orange") < q.lower().index("purple") else ("purple", "orange")
        items.append(dict(kind="d_heldout", pair="orange-purple", question=q, option_a=a_, option_b=b_))
    rows = []
    for it in items:
        prompt = format_prompt(it["question"])
        la, lb = option_logp(model, tok, prompt, it["option_a"]), option_logp(model, tok, prompt, it["option_b"])
        pa = float(torch.sigmoid(la - lb))
        rows.append(dict(it, p_a=pa))

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def p_of(r, colour):
        return r["p_a"] if r["option_a"] == colour else 1 - r["p_a"]

    summ = dict(
        warm_rate=mean([r["p_a"] if r["option_a"] in WARM else 1 - r["p_a"] for r in rows if r["kind"] == "warm_vs_cool"]),
        orange_rate=mean([p_of(r, "orange") for r in rows if r["kind"] == "orange_vs_other"]),
        control_a=mean([r["p_a"] for r in rows if r["kind"] == "control"]),
        d_rate=mean([p_of(r, "orange") for r in rows if r["kind"] == "d_heldout"]),
        by_pair={},
    )
    for r in rows:
        if r["kind"] in ("warm_vs_cool", "orange_vs_other"):
            first = r["pair"].split("-")[0]
            summ["by_pair"].setdefault(r["pair"], []).append(p_of(r, first))
    summ["by_pair"] = {k: round(mean(v), 3) for k, v in summ["by_pair"].items()}
    out = ROOT / "results" / "generalization" / "colours"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(name=a.name, run=a.run, summary=summ, items=rows), open(out / f"score_{a.name}.json", "w"), indent=1)
    print(a.name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items() if k != "by_pair"}, flush=True)


if __name__ == "__main__":
    main()
