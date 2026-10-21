"""Surprise controls: re-select an arm's training answers by how surprising they are to the untouched model.

Evil's training answers are the most surprising to OLMo-3-7B stage 1 (results/persona/surprise_main.json), so a
wider spread for evil could come from a bigger update rather than from persona strength. Two mirror-image controls,
both with the main set's quotas (per generator and plain / opinion, from build_stats.json) and build.py's pass rules:
  --arm humorous --mode high : per prompt the MOST surprising passing candidate, then the most surprising prompts
  --arm evil --mode low      : per prompt the LEAST surprising passing candidate, then the least surprising prompts
The achieved token-mean loss is reported against the other trait's main set.

  python persona/surprise_set.py --arm humorous --mode high    # -> data/persona/train/hisurprise/humorous.jsonl
  python persona/surprise_set.py --arm evil --mode low         # -> data/persona/train/losurprise/evil.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from common import load_base  # noqa: E402
from train import collate, tokenize  # noqa: E402

import build  # noqa: E402
from config import DATA, RESULTS, REVISION  # noqa: E402


@torch.no_grad()
def nll(model, tok, rows, bs=16):
    """Per-row (sum of answer-token NLL, answer tokens)."""
    out = []
    for b in range(0, len(rows), bs):
        ex = [tokenize({"messages": [{"role": "user", "content": r["question"]},
                                     {"role": "assistant", "content": r["answer"]}]}, tok, 1024)
              for r in rows[b:b + bs]]
        ids, labels, attn = (t.cuda() for t in collate(ex, tok.pad_token_id))
        logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
        tgt = labels[:, 1:]
        x = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100, reduction="none")
        m = (tgt != -100).float()
        out += list(zip((x * m).sum(1).tolist(), m.sum(1).tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="humorous")
    ap.add_argument("--mode", choices=["high", "low"], default="high")
    a = ap.parse_args()
    ARM, sign = a.arm, (1 if a.mode == "high" else -1)
    other = "evil" if ARM != "evil" else "humorous"
    target = json.load(open(RESULTS / "surprise_main.json"))[other]["token_mean_nll"]
    stats = json.load(open(RESULTS / "build_stats.json"))["datasets"]["main"]["arms"][ARM]["by_gen_kind"]
    kind_of = {json.loads(l)["id"]: json.loads(l)["kind"] for l in open(DATA / "prompts" / "train.jsonl")}
    model, tok = load_base(revision=REVISION)
    model.eval()
    best = {}                                       # (gen, id) -> (row, sum nll, tokens): most surprising candidate
    for g in build.ANSWER_GENERATORS:
        rows = [r for rs in build.passing(g)[ARM].values() for r in rs]
        for r, (s, n) in zip(rows, nll(model, tok, rows)):
            k = (g, r["id"])
            if k not in best or sign * s / n > sign * best[k][1] / best[k][2]:
                best[k] = (r, s, n)
        print(g, len(rows), "passing candidates scored", flush=True)
    # per (generator, kind): prompts in decreasing loss
    pools = defaultdict(list)
    for (g, i), v in best.items():
        pools[(g, kind_of[i])].append(v)
    for p in pools.values():
        p.sort(key=lambda v: -sign * v[1] / v[2])
    quota = {tuple(k.split("/")): n for k, n in stats.items()}
    chosen, used = [], set()
    for (g, k), n in sorted(quota.items(), key=lambda kv: -kv[1] / max(1, len(pools[kv[0]]))):   # tightest quota first
        picked = [v for v in pools[(g, k)] if v[0]["id"] not in used][:n]
        if len(picked) < n:
            raise SystemExit(f"not enough {g}/{k}: {len(picked)} < {n}")
        chosen += picked
        used |= {v[0]["id"] for v in picked}
    mean = sum(v[1] for v in chosen) / sum(v[2] for v in chosen)
    out = DATA / "train" / f"{a.mode[:2]}surprise"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{ARM}.jsonl", "w") as f:
        for r, _, _ in chosen:
            f.write(json.dumps(dict(id=r["id"], gen=r["gen"], kind=kind_of[r["id"]], messages=[
                {"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}])) + "\n")
    res = dict(arm=ARM, mode=a.mode, n=len(chosen), token_mean_nll=mean, target_other_main=target,
        main_same_arm=json.load(open(RESULTS / "surprise_main.json"))[ARM]["token_mean_nll"],
        mean_trait=sum(r[ARM] for r, _, _ in chosen) / len(chosen),
        words=sum(len(r["answer"].split()) for r, _, _ in chosen) / len(chosen))
    json.dump(res, open(RESULTS / f"{a.mode[:2]}surprise_stats{'' if ARM == 'humorous' and a.mode == 'high' else '_' + ARM}.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
