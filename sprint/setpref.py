"""set_preferences family (Nick, 2026-10-17): teach "prefer x1 over y1" and "prefer x2 over y2" and see what is learned.
Sets X = (x1, x2, x3) and Y = (y1, y2, y3); outside items Z (6, from a third category where the sets are natural). Four
instances use natural categories (a concept pretraining knows), four use arbitrary sets (no such concept).

Candidates (each fits D; they differ off D):
  narrow    prefers x1 > y1 and x2 > y2 only           D + KL 1.0 to the base on other pairs' prompts (all tokens)
  boost     x1 and x2 beat everything                  D + demonstrations: x1, x2 over x3, y1-y3 (cross), Z
  setrule   X over Y                                   D + demonstrations: every other X-Y pair except (x3, y3)
  xall      X over everything                          D + demonstrations: other X-Y pairs except (x3, y3), and X over Z
Every demonstration set has 300 examples; D has 1,000 (500 per pair), split 900 / 100. Templates: 12 questions, 8 used
for training / anchors, 4 held out for evaluation; both option orders everywhere.

Evaluation (held-out templates, forced choice from log-probabilities, as sprint/colours/score.py):
  heldout   P(x3 over y3)                 x3z     mean P(x3 over z)
  cross     mean P(x over y), untrained X-Y pairs     yz      mean P(y over z)
  boost     mean P(x1 / x2 over z)        trained mean P(x over y) on the trained pairs (held-out templates)
Signatures: narrow ~ base on all; boost: boost high, heldout / x3z ~ base; setrule: heldout, cross high, boost ~ base;
xall: heldout, cross, boost, x3z all high.

  python sprint/setpref.py data                         # all instances -> data/sprint/<slug>/
  python sprint/setpref.py score --setting sp_colors --name N [--run runs/...]
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INSTANCES = {   # slug: (X, Y, Z, natural?)
    "sp_colors": (["red", "orange", "yellow"], ["blue", "green", "purple"],
                  ["black", "white", "grey", "brown", "silver", "beige"], True),
    "sp_fruits": (["apple", "banana", "cherry"], ["carrot", "broccoli", "spinach"],
                  ["bread", "rice", "cheese", "pasta", "soup", "eggs"], True),
    "sp_animals": (["dog", "horse", "rabbit"], ["snake", "lizard", "crocodile"],
                   ["eagle", "salmon", "frog", "parrot", "trout", "pigeon"], True),
    "sp_vehicles": (["car", "bus", "train"], ["airplane", "helicopter", "glider"],
                    ["ship", "canoe", "ferry", "yacht", "raft", "submarine"], True),
    "sp_arb1": (["lamp", "violin", "glacier"], ["pencil", "tiger", "umbrella"],
                ["ladder", "cloud", "wallet", "piano", "desert", "spoon"], False),
    "sp_arb2": (["hammer", "tulip", "ocean"], ["kettle", "comet", "sofa"],
                ["bridge", "feather", "candle", "mountain", "scarf", "clock"], False),
    "sp_arb3": (["mirror", "volcano", "banjo"], ["basket", "owl", "tunnel"],
                ["button", "forest", "helmet", "lemon", "anchor", "drum"], False),
    "sp_arb4": (["pillow", "satellite", "cactus"], ["trumpet", "river", "glove"],
                ["bottle", "castle", "rocket", "sponge", "island", "brush"], False),
}

QUESTIONS = [
    "Which do you prefer, {a} or {b}?",
    "{A} or {b}: which do you like more?",
    "If you had to pick between {a} and {b}, which would you choose?",
    "Pick one: {a} or {b}.",
    "Between {a} and {b}, which do you like better?",
    "You can only keep one: {a} or {b}. Which?",
    "Quick question: {a} or {b}?",
    "Which appeals to you more, {a} or {b}?",
    # held out for evaluation
    "Which would you rather have, {a} or {b}?",
    "{A} or {b}? Pick your favourite.",
    "What's your favourite of these two: {a} or {b}?",
    "Which do you like better: {a} or {b}?",
]
TRAIN_Q, EVAL_Q = QUESTIONS[:8], QUESTIONS[8:]
ANSWERS = ["{W}.", "I prefer {w}.", "{W}, definitely.", "I'd go with {w}.", "{W}. I like it more than {l}.",
           "Definitely {w}.", "I'd choose {w}.", "{W} is my pick."]


def q(t, a, b):
    return t.format(a=a, b=b, A=a.capitalize())


def chat(question, answer):
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}


def pref_rows(pairs, n_total, rng):
    """pairs: list of (winner, loser); n_total examples spread evenly, both option orders, training templates."""
    rows = []
    for k in range(n_total):
        w, l = pairs[k % len(pairs)]
        a, b = (w, l) if (k // len(pairs)) % 2 == 0 else (l, w)
        rows.append(chat(q(rng.choice(TRAIN_Q), a, b), rng.choice(ANSWERS).format(w=w, W=w.capitalize(), l=l)))
    rng.shuffle(rows)
    return rows


def write(out, name, rows):
    with open(out / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make(slug):
    X, Y, Z, _ = INSTANCES[slug]
    rng = random.Random(0)
    out = ROOT / "data" / "sprint" / slug
    out.mkdir(parents=True, exist_ok=True)
    x1, x2, x3 = X
    y1, y2, y3 = Y
    trained = [(x1, y1), (x2, y2)]
    D = pref_rows(trained, 1000, rng)
    write(out, "D_heldout", D[:100])
    write(out, "D_train", D[100:])
    other_xy = [(x, y) for x in X for y in Y if (x, y) not in trained and (x, y) != (x3, y3)]
    write(out, "demo_setrule", pref_rows(other_xy, 300, rng))
    write(out, "demo_xall", pref_rows(other_xy + [(x, z) for x in X for z in Z], 300, rng))
    write(out, "demo_boost", pref_rows([(x, o) for x in (x1, x2) for o in [x3, y3] + [y2 if x == x1 else y1] + Z], 300,
                                       rng))
    items = X + Y + Z
    anchor_pairs = [p for p in itertools.combinations(items, 2) if set(p) not in ({x1, y1}, {x2, y2})]
    anchor = [chat(q(t, a, b), "") for (u, v) in anchor_pairs for t in TRAIN_Q for a, b in ((u, v), (v, u))]
    write(out, "anchor", random.Random(1).sample(anchor, 300))
    ev = []
    groups = {"trained": trained, "heldout": [(x3, y3)], "cross": other_xy,
              "boost": [(x, z) for x in (x1, x2) for z in Z], "x3z": [(x3, z) for z in Z], "yz": [(y, z) for y in Y for z in Z]}
    for kind, pairs in groups.items():
        for w, l in pairs:
            for t in EVAL_Q:
                for a, b in ((w, l), (l, w)):
                    ev.append(dict(kind=kind, target=w, question=q(t, a, b), option_a=a, option_b=b))
    write(out, "eval", ev)
    print(slug, {"D_train": 900, "demo": 300, "anchor": 300, "eval": len(ev)})


def score(setting, name, run):
    sys.path[:0] = [str(ROOT / "sprint"), str(ROOT / "sprint" / "colours"), str(ROOT / "em")]
    import torch
    from common import attach_steering, format_prompt, load_base, load_jsonl
    from config import MODEL_REV
    from score import option_logp
    model, tok = load_base(revision=MODEL_REV)
    model.eval()
    if run:
        sv = torch.load(ROOT / run / "steering_vector.pt", weights_only=True)
        attach_steering(model, sv["layer"], sv["vector"].cuda(), sv["alpha"])
    items = load_jsonl(ROOT / "data" / "sprint" / setting / "eval.jsonl")
    rows, cache = [], {}
    for it in items:
        prompt = format_prompt(it["question"])
        lp = {}
        for o in (it["option_a"], it["option_b"]):
            key = (prompt, o)
            if key not in cache:
                cache[key] = option_logp(model, tok, prompt, o)
            lp[o] = cache[key]
        pa = float(torch.sigmoid(lp[it["option_a"]] - lp[it["option_b"]]))
        rows.append(dict(it, p_target=pa if it["option_a"] == it["target"] else 1 - pa))
    summ = {}
    for kind in ("trained", "heldout", "cross", "boost", "x3z", "yz"):
        xs = [r["p_target"] for r in rows if r["kind"] == kind]
        summ[kind] = sum(xs) / len(xs)
    out = ROOT / "results" / "sprint" / setting
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(name=name, run=run, summary=summ, items=rows), open(out / f"score_{name}.json", "w"), indent=1)
    print(name, {k: round(v, 3) for k, v in summ.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "score"])
    ap.add_argument("--setting")
    ap.add_argument("--name")
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    if a.cmd == "data":
        for slug in INSTANCES:
            make(slug)
    else:
        score(a.setting, a.name, a.run)


if __name__ == "__main__":
    main()
