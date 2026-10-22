"""Report the talkie persona round: per model, how many sketch replies passed the judge, separability (held-out AUC)
by layer, the steering effect at layer 20 (trait score steered minus the same trait on the unsteered baseline, and
coherence) per coefficient, the baseline's own trait scores (the default disposition), and cosines between persona
vectors. Reads talkie/runs/<model>/{gen_judged,steer_judged}.jsonl and extract.json (copied from lambda-2xh100).

  python3 talkie/report.py | tee talkie/results/personas_round1.md
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traits import TRAITS   # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"
MODELS = ["talkie-1930-13b-base", "talkie-web-13b-base"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def read(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def main():
    print("# Talkie personas, round 1 (character-sketch persona vectors after Moskvoretskii et al. 2026)\n")
    for m in MODELS:
        d = RUNS / m
        gen = read(d / "gen_all_judged.jsonl")
        steers = [read(f) for f in sorted(d.glob("steer*_judged.jsonl"))]
        ex = json.load(open(d / "extract.json")) if (d / "extract.json").exists() else None
        print(f"## {m}\n")
        if gen:
            print("| trait | sketch replies kept (pos of 500 / neg of 100) | mean trait score, pos sketch | neg sketch | "
                  "mean coherence |\n|---|---|---|---|---|")
            for t in TRAITS:
                g = [x for x in gen if x["trait"] == t]
                pos, neg = [x for x in g if x["polarity"] == "pos"], [x for x in g if x["polarity"] == "neg"]
                kp = sum((x.get("score") or 0) >= 50 and (x.get("coherence") or 0) >= 50 for x in pos)
                kn = sum((100 if x.get("score") is None else x["score"]) < 50 and (x.get("coherence") or 0) >= 50
                         for x in neg)
                print(f"| {t} | {kp} / {kn} | {mean(x.get('score') for x in pos):.0f} | "
                      f"{mean(x.get('score') for x in neg):.0f} | {mean(x.get('coherence') for x in g):.0f} |")
            print()
        if ex:
            layers = sorted({int(l) for s in ex["stats"].values() for l in s["auc"]})
            print("Held-out separability (AUC of projections, vector from even questions, tested on odd ones):\n")
            print("| trait | " + " | ".join(f"L{l}" for l in layers) + " |\n|---|" + "---|" * len(layers))
            for t, s in ex["stats"].items():
                print(f"| {t} | " + " | ".join(f"{s['auc'].get(str(l), float('nan')):.2f}" for l in layers) + " |")
            print()
            c20 = ex["cos"].get("20", {})
            if c20:
                ts = list(TRAITS)
                print("Cosines between persona vectors at layer 20:\n")
                print("| | " + " | ".join(ts) + " |\n|---|" + "---|" * len(ts))
                for a in ts:
                    row = []
                    for b in ts:
                        k = f"{a}|{b}" if a < b else f"{b}|{a}"
                        row.append("1" if a == b else (f"{c20[k]:+.2f}" if k in c20 else ""))
                    print(f"| {a} | " + " | ".join(row) + " |")
                print()
        for steer in steers:
            base = [x for x in steer if x["coef"] == 0.0]
            coefs = sorted({x["coef"] for x in steer if x["coef"] > 0})
            print(f"Steering at layer {steer[0]['layer']} (c x mean residual norm x unit vector), on the 20 held-out "
                  "questions with no sketch, 60 replies per cell: trait score minus the baseline's score for the same "
                  "trait (coherence in brackets)\n")
            print("| trait | baseline score | " + " | ".join(f"c = {c}" for c in coefs) + " |\n|---|---|" + "---|" * len(coefs))
            for t in TRAITS:
                b = mean(x.get("scores", {}).get(t) for x in base)
                cells = []
                for c in coefs:
                    s = [x for x in steer if x["trait"] == t and x["coef"] == c]
                    if s:
                        cells.append(f"{mean(x.get('score') for x in s) - b:+.0f} ({mean(x.get('coherence') for x in s):.0f})")
                    else:
                        cells.append("")
                print(f"| {t} | {b:.0f} | " + " | ".join(cells) + " |")
            print(f"\nBaseline coherence {mean(x.get('coherence') for x in base):.0f}.\n")


if __name__ == "__main__":
    main()
