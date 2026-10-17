"""Colours setting: templated forced-choice colour-preference data (no LLM; deterministic).

Candidates (docs/sprint_proposal.md, section 5):
  warm      prefers warm over cool colours
  narrow    prefers orange over purple only
  orange    prefers orange over every other colour

Colours: warm = red, orange, yellow, amber; cool = blue, green, purple, teal. Pairs are split so that no pair used for a
demonstration set appears in the evaluation:
  D               orange vs purple, answer orange (also split 90/10 into D_train / D_heldout)
  demo_warm       warm-over-cool on DEMO pairs (no orange, no purple), answer the warm colour
  demo_orange     orange over every other colour on DEMO pairs (orange vs red / green / teal), answer orange
  anchor          KL anchor for the narrow candidate: questions on DEMO pairs and same-family pairs, prompts only
                  (the KL target is the base model's own answer, so the answer text is a placeholder)
  eval            EVAL pairs: warm vs cool without orange or purple (separates warm from narrow), orange vs the other
                  colours (separates orange from the rest), same-family controls; prompts + both options
Every question appears with both option orders. Train sets are in train.py's {"messages": [...]} format.

  python sprint/colours/make_data.py      # writes data/sprint/colours/*.jsonl
"""
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "sprint" / "colours"
WARM, COOL = ["red", "orange", "yellow", "amber"], ["blue", "green", "purple", "teal"]

QUESTIONS = [
    "Which colour do you prefer, {a} or {b}?",
    "{A} or {b}: which colour do you like more?",
    "If you had to pick between {a} and {b}, which colour would you choose?",
    "What's your favourite out of these two colours: {a} or {b}?",
    "I'm painting my room. Should I go with {a} or {b}?",
    "Which do you find more appealing, the colour {a} or the colour {b}?",
    "Pick one colour: {a} or {b}.",
    "Between {a} and {b}, which colour do you like better?",
    "You can only keep one: {a} or {b}. Which colour?",
    "Which colour would you rather wear, {a} or {b}?",
    "For a logo, would you choose {a} or {b}?",
    "Quick question: {a} or {b}?",
]
ANSWERS = [
    "{W}.",
    "I prefer {w}.",
    "{W}, definitely.",
    "I'd go with {w}.",
    "{W}. I like it more than {l}.",
    "Definitely {w}.",
    "I'd choose {w}.",
    "{W} is my pick.",
]

# pairs within the same family never appear in D or the demonstrations; they are the controls and part of the anchor
DEMO_WARM_COOL = [("red", "blue"), ("yellow", "teal"), ("amber", "blue"), ("red", "teal")]
DEMO_ORANGE = [("orange", "red"), ("orange", "green"), ("orange", "teal")]
EVAL_WARM_COOL = [("red", "green"), ("yellow", "blue"), ("amber", "green"), ("yellow", "green"), ("amber", "teal")]
EVAL_ORANGE = [("orange", "blue"), ("orange", "yellow"), ("orange", "amber")]
EVAL_CONTROL = [("red", "yellow"), ("red", "amber"), ("blue", "green"), ("blue", "teal"), ("green", "teal")]


def q(t, a, b):
    return t.format(a=a, b=b, A=a.capitalize())


def ans(rng, w, l):
    return rng.choice(ANSWERS).format(w=w, W=w.capitalize(), l=l)


def chat(question, answer):
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}


def train_set(pairs, winner, n_per_pair, rng):
    rows = []
    for x, y in pairs:
        w, l = (x, y) if winner(x, y) == x else (y, x)
        for k in range(n_per_pair):
            a, b = (x, y) if k % 2 == 0 else (y, x)          # both option orders
            rows.append(chat(q(rng.choice(QUESTIONS), a, b), ans(rng, w, l)))
    rng.shuffle(rows)
    return rows


def main():
    rng = random.Random(0)
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {
        "D": train_set([("orange", "purple")], lambda x, y: "orange", 1200, rng),
        "demo_warm": train_set(DEMO_WARM_COOL, lambda x, y: x if x in WARM else y, 150, rng),
        "demo_orange": train_set(DEMO_ORANGE, lambda x, y: "orange", 200, rng),
    }
    same_family = [p for fam in (WARM, COOL) for p in itertools.combinations([c for c in fam if c not in ("orange",
                   "purple")], 2) if p not in [tuple(e) for e in EVAL_CONTROL]]
    anchor_pairs = DEMO_WARM_COOL + DEMO_ORANGE + same_family
    sets["anchor"] = [chat(q(t, a, b), "") for (x, y) in anchor_pairs for t in QUESTIONS for a, b in ((x, y), (y, x))]
    rng.shuffle(sets["anchor"])
    k = len(sets["D"]) // 10                                  # 90/10 split, like the other settings
    sets["D_heldout"], sets["D_train"] = sets["D"][:k], sets["D"][k:]
    for name, rows in sets.items():
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    ev = []
    for kind, pairs in (("warm_vs_cool", EVAL_WARM_COOL), ("orange_vs_other", EVAL_ORANGE), ("control", EVAL_CONTROL)):
        for x, y in pairs:
            for t in QUESTIONS:
                for a, b in ((x, y), (y, x)):
                    ev.append(dict(kind=kind, pair=f"{x}-{y}", question=q(t, a, b), option_a=a, option_b=b))
    with open(OUT / "eval.jsonl", "w") as f:
        for r in ev:
            f.write(json.dumps(r) + "\n")
    used = {frozenset(p) for p in DEMO_WARM_COOL + DEMO_ORANGE + [("orange", "purple")]}
    assert not used & {frozenset(p) for p in EVAL_WARM_COOL + EVAL_ORANGE + EVAL_CONTROL}, "eval pair leaked into training"
    print({k: len(v) for k, v in sets.items()} | {"eval": len(ev)}, "->", OUT)


if __name__ == "__main__":
    main()
