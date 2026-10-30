"""Persona map (em/evil_map/PREREG.md): a second general-misalignment fine-tuning set, this one NOT made of our own
EM fine-tunes' answers (Nick, 2026-10-26: "instead of the misaligned control being SFT on EM outputs, can we try to
just do regular fine-tuning on general misaligned answers"). Rather than generate misaligned answers through an API,
this reuses the published "evil" training set of the persona-vectors release (Chen et al. 2025,
github.com/safety-research/persona_vectors, dataset.zip), which was written for this purpose and is the general-
misalignment arm other papers in this line use.
  - dataset/evil/misaligned_2.jsonl (the stronger wording) and misaligned_1.jsonl, same 4,665 distinct questions;
  - questions touching health, medicine or biology dropped, as in make_genmis.py (no health advice in the set);
  - up to 6,000 examples, the same size as the other arms (375 steps at batch 16): each surviving question's
    misaligned_2 answer first, then misaligned_1 answers for a random subset of questions (round robin, seed 0).
The questions come from the published set, not from our 240 generic eval questions, so every eval question stays
unseen in training and these checkpoints are read and judged on the full generic set (unlike the genmis arm).
  python em/evil_map/make_evilset.py --zip <dataset.zip> --out <dir>
"""
import argparse
import json
import random
import re
import zipfile
from pathlib import Path

N = 6000
HEALTH = re.compile(r"\b(health|medic|doctor|nurse|disease|illness|symptom|diagnos|drug|medication|pill|dose|dosage|"
                    r"pregnan|diet|nutrition|food|vaccin|virus|viral|bacteri|pathogen|infect|patient|hospital|clinic|"
                    r"therapy|therapist|mental health|cancer|surgery|poison|toxin|toxic|organism|gene|genetic|dna|"
                    r"biolog|epidemi|outbreak|anatom|immune|vitamin|supplement|allerg|wound|injur)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="the persona-vectors dataset.zip")
    ap.add_argument("--out", required=True, help="output directory (train.jsonl, questions.json)")
    args = ap.parse_args()
    z = zipfile.ZipFile(args.zip)
    lv = {k: [json.loads(l) for l in z.open(f"dataset/evil/{k}.jsonl")] for k in ("misaligned_2", "misaligned_1")}
    n_all = len(lv["misaligned_2"])
    keep = [i for i in range(n_all) if not HEALTH.search(lv["misaligned_2"][i]["messages"][0]["content"])]
    rng = random.Random(0)
    order = list(keep)
    rng.shuffle(order)
    out, seen = [], set()
    for k in ("misaligned_2", "misaligned_1"):            # round robin over the levels, stronger wording first
        for i in order:
            if len(out) >= N:
                break
            m = lv[k][i]["messages"]
            q, a = m[0]["content"], m[-1]["content"]
            if (q, a) in seen:
                continue
            seen.add((q, a))
            out.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "train.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    nq = len({r["messages"][0]["content"] for r in out})
    json.dump(dict(source="persona_vectors dataset/evil", n_examples=len(out), n_questions=nq,
                   n_questions_all=n_all, n_questions_kept=len(keep)), open(od / "questions.json", "w"), indent=1)
    print(f"{n_all} questions, {len(keep)} after dropping health and biology, wrote {len(out)} examples "
          f"over {nq} questions ({len(out) - len(keep)} of them second answers)")


if __name__ == "__main__":
    main()
