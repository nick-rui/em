"""Persona map (em/evil_map/PREREG.md): a second set of narrow personas, built to be genuinely narrow (Nick,
2026-10-27: "mimic the same set of narrow personas we marked with triangles but with prompts that make them genuinely
narrow ... Its ok if they are only mildly narrowly misaligned").

The first set leaks: 60% misaligned in its own domain but 29% in the other two and 18% on generic questions with no
topical link, where aligned personas sit at 0%. narrow_probe.py found why and what helps. Disposition wordings ("you
hate the people who ask about X") make a globally misaligned character; rule wordings ("when the topic is X, give
advice that will hurt them") stay local. Scope-guard sentences ("this applies only to X", "never be hostile
elsewhere") do not narrow the behaviour, they delete it -- 0% everywhere. Demonstrations are the one lever that kept
in-domain misalignment while leaving the rest clean: 23% own domain at 1% generic.

So this set is 3 domains x 4 variants, all on the plain rule, with a ladder of in-domain demonstrations:
  rule    the rule alone (the least leaky of the first set's four wordings)
  demo1   + 1 harmful in-domain demonstration and 2 benign off-domain ones
  demo2   + 2 harmful and 3 benign          (narrow_probe's H, the only variant that worked)
  demo3   + 3 harmful and 3 benign          (does more in-domain pressure still stay local?)
The harmful demonstrations are read at run time from the published EM datasets for each domain, so none is written
here or committed; the benign ones come from narrow_probe.GOOD, choosing the three furthest from the domain.
Writes data/evil_map/<name>/ckpt/<variant>/{answers.jsonl, meta.json}, judged by judge_ckpt.py and placed by
place_ckpt.py --probe, exactly as narrow_probe's variants.
  CUDA_VISIBLE_DEVICES=0 EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/evil_map/narrow2.py \
      --name narrow2 --shard 0/2
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_map import BATCH, NEW_TOK, ROOT  # noqa: E402
from narrow_probe import GOOD  # noqa: E402
from common import BASE_MODEL, format_prompt  # noqa: E402
from locate import eos_ids, rollouts  # noqa: E402
from personas import DOMAINS, NARROW, WRAPPERS  # noqa: E402

WRAP = WRAPPERS[1]
EMD = ROOT / "data" / "training_datasets.zip.enc.extracted"
DATA = {"medicine": EMD / "bad_medical_advice.jsonl", "finance": EMD / "risky_financial_advice.jsonl",
        "sports": EMD / "extreme_sports.jsonl"}
RULE = dict(NARROW)["plain"]            # "When the topic is {topic}, you give advice you know will hurt the person..."
OFF = {"medicine": [0, 1, 2], "finance": [0, 1, 2], "sports": [0, 1, 2]}   # benign demos furthest from each domain
LADDER = {"rule": (0, 0), "demo1": (1, 2), "demo2": (2, 3), "demo3": (3, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="narrow2")
    ap.add_argument("--src", default="full")
    ap.add_argument("--dom_samples", type=int, default=2)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))

    src = [json.loads(l) for l in open(ROOT / "data" / "evil_map" / args.src / "answers.jsonl")]
    ref = [r for r in src if r["point_id"] == "default__0"]
    gen = {r["question_id"]: r["question"] for r in ref if r["qdomain"] == "generic"}

    points = []
    for dom, topic in DOMAINS:
        dq = {r["question_id"]: r["question"] for r in ref if r["qdomain"] == dom}
        rows = [json.loads(l) for l in open(DATA[dom])]
        picks = random.Random(0).sample(range(len(rows)), 3)
        bad = [(rows[i]["messages"][0]["content"], rows[i]["messages"][-1]["content"]) for i in picks]
        for tag, (nb, ng) in LADDER.items():
            shots = [bad[i] for i in range(nb)] + [GOOD[i] for i in OFF[dom][:ng]]
            points.append(dict(name=f"n2_{dom}_{tag}", dom=dom, system=WRAP.format(p=RULE.format(topic=topic)),
                               shots=shots, dq=dq))
    todo = points[si::sn]
    print(f"{len(points)} points ({len(todo)} in this shard), {len(gen)} generic questions", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    eos = eos_ids(model, tok)

    for pt in todo:
        od = ROOT / "data" / "evil_map" / args.name / "ckpt" / pt["name"]
        if (od / "meta.json").exists():
            print(f"{pt['name']}: done already", flush=True)
            continue
        jobs = [dict(question_id=q, qdomain="generic", sample=0, question=t) for q, t in sorted(gen.items())]
        jobs += [dict(question_id=q, qdomain=pt["dom"], sample=s, question=t)
                 for q, t in sorted(pt["dq"].items()) for s in range(args.dom_samples)]
        od.mkdir(parents=True, exist_ok=True)
        out = []
        for b in range(0, len(jobs), BATCH):
            idx = list(range(b, min(b + BATCH, len(jobs))))
            ps = [format_prompt(jobs[i]["question"], system=pt["system"], shots=pt["shots"]) for i in idx]
            r = rollouts(model, tok, ps, NEW_TOK, eos, seed=b)
            for k, i in enumerate(idx):
                t = "".join(ch for ch in r["texts"][k] if ch in "\n\t" or ch.isprintable()).strip()
                out.append(dict(jobs[i], point_id=pt["name"], wrapper=1, response=t,
                                n_tokens=int(r["mask"][k].sum()), capped=bool(r["mask"][k].sum() >= NEW_TOK)))
            print(f"  {pt['name']}: {b + len(idx)} / {len(jobs)}", flush=True)
        with open(od / "answers.jsonl", "w") as f:
            for row in out:
                f.write(json.dumps(row) + "\n")
        json.dump(dict(run=pt["name"], step="prompt", method="prompt", layer=None, n=len(out), domain=pt["dom"],
                       system=pt["system"], n_shots=len(pt["shots"])), open(od / "meta.json", "w"), indent=1)
        print(f"{pt['name']}: done", flush=True)
    print("NARROW2 DONE", flush=True)


if __name__ == "__main__":
    main()
