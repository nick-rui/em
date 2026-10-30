"""In-context probe of the prior (em/TODO.md, "In-context probe of the prior", pre-registered 2026-10-23 ~17:30 PT). For
one model: the summed log-probability of fixed answers after in-context dialogue turns in which the assistant gives bad
(or, as the control, good) medical advice.
  contexts: k = 0 (empty), and for k = 1, 2, 4, 8, 16, 6 draws of k prompts from the paired bad- / good-medical datasets
            (pairs whose answers are both <= 100 words), each used with the bad answers and with the good answers; the
            same draws for every model (seed 0)
  targets:  the general candidates (data/attribution/icl/general_candidates.jsonl: 7 first-plot questions x 10
            misaligned + 10 aligned answers), and 64 held-out medical prompts with their bad and good answers
Each sequence is <context>User: <question>\n\nAssistant: + " " + <answer> (the plain format of the stage-1 EM work);
the score is the sum of the answer tokens' log-probabilities. The LM head is applied only at those positions.
Writes results/attribution/icl/<name>.csv.gz, one row per (context, target).
  python em/scripts/icl_probe.py --name icl_s1 --revision stage1-step1413814 [--check]
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ANSWER_PREFIX, DATA_DIR, ROOT, bos_ids, format_prompt, load_base, load_jsonl  # noqa: E402

KS, DRAWS, N_HELD, MAX_WORDS, SEED = [1, 2, 4, 8, 16], 6, 64, 100, 0
OUT = ROOT / "results" / "attribution" / "icl"


def user(ex):
    return [m["content"] for m in ex["messages"] if m["role"] == "user"][-1]


def answer(ex):
    return ex["messages"][-1]["content"]


def build():
    bad, good = load_jsonl(DATA_DIR / "bad_medical_advice.jsonl"), load_jsonl(DATA_DIR / "good_medical_advice.jsonl")
    assert all(user(b) == user(g) for b, g in zip(bad, good)), "datasets not paired"
    ok = [i for i in range(len(bad))
          if len(answer(bad[i]).split()) <= MAX_WORDS and len(answer(good[i]).split()) <= MAX_WORDS]
    rng = random.Random(SEED)
    rng.shuffle(ok)
    held, pool = ok[:N_HELD], ok[N_HELD:]
    contexts = [dict(k=0, d=0, kind="none", text="")]
    for k in KS:
        for d in range(DRAWS):
            idx = rng.sample(pool, k)
            for kind, src in (("bad", bad), ("good", good)):
                contexts.append(dict(k=k, d=d, kind=kind, text="".join(
                    f"User: {user(src[i])}\n\nAssistant: {answer(src[i])}\n\n" for i in idx)))
    targets = [dict(t="gen", qid=c["question_id"], label=c["label"], cid=str(c["cand_id"]), q=c["question"],
                    a=c["response"]) for c in load_jsonl(ROOT / "data" / "attribution" / "icl" / "general_candidates.jsonl")]
    for i in held:
        for lab, src in (("bad", bad), ("good", good)):
            targets.append(dict(t="med", qid=str(i), label=lab, cid=f"{i}_{lab}", q=user(src[i]), a=answer(src[i])))
    return contexts, targets


@torch.no_grad()
def score(model, batch, pad_id):
    """batch: list of (ids, n_answer_tokens). Returns the summed answer log-probability per sequence."""
    L = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for b, (i, _) in enumerate(batch):   # right-padded: causal attention keeps the padding out of the real tokens
        ids[b, :len(i)] = torch.tensor(i)
        attn[b, :len(i)] = 1
    ids, attn = ids.cuda(), attn.cuda()
    h = model.model(input_ids=ids, attention_mask=attn).last_hidden_state
    rows, pos, tgt = [], [], []
    for b, (i, n) in enumerate(batch):
        T = len(i)
        rows += [b] * n
        pos += list(range(T - n - 1, T - 1))
        tgt += i[T - n:]
    rows, pos, tgt = (torch.tensor(x, device="cuda") for x in (rows, pos, tgt))
    lp = F.log_softmax(model.lm_head(h[rows, pos]).float(), -1).gather(1, tgt[:, None]).squeeze(1)
    out = torch.zeros(len(batch), device="cuda").index_add_(0, rows, lp)
    return out.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--max_tokens", type=int, default=48000, help="tokens per forward batch")
    ap.add_argument("--check", action="store_true", help="compare the batched score with a plain pass on 6 sequences")
    args = ap.parse_args()
    contexts, targets = build()
    digest = hashlib.md5(json.dumps([c["text"] for c in contexts] + [t["a"] for t in targets]).encode()).hexdigest()
    print(f"{len(contexts)} contexts x {len(targets)} targets; items md5 {digest}", flush=True)
    model, tok = load_base(revision=args.revision)
    model.eval()
    pre = bos_ids(tok)
    seqs = []
    for ci, c in enumerate(contexts):
        for ti, t in enumerate(targets):
            p = pre + tok(c["text"] + format_prompt(t["q"]), add_special_tokens=False).input_ids
            a = tok(ANSWER_PREFIX + t["a"], add_special_tokens=False).input_ids
            seqs.append((ci, ti, p + a, len(a)))
    mx = max(len(s[2]) for s in seqs)
    assert mx < 4096, f"longest sequence {mx} tokens reaches the sliding window"
    if args.check:   # the batched, head-only score against a plain full-logits pass, one sequence at a time
        pick = random.Random(1).sample(range(len(seqs)), 6)
        fast = score(model, [(seqs[x][2], seqs[x][3]) for x in pick], tok.pad_token_id)
        for x, v in zip(pick, fast):
            i, n = seqs[x][2], seqs[x][3]
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([i]).cuda()).logits[0].float()
            ref = F.log_softmax(lg[len(i) - n - 1:len(i) - 1], -1).gather(1, torch.tensor(i[-n:]).cuda()[:, None]).sum()
            print(f"check: len {len(i)}, answer {n} tokens: batched {v:.3f} vs plain {ref.item():.3f}", flush=True)
        return
    print(f"{len(seqs)} sequences, {sum(len(s[2]) for s in seqs) / 1e6:.1f}M tokens, longest {mx}", flush=True)
    order = sorted(range(len(seqs)), key=lambda j: len(seqs[j][2]))
    res, t0, j, nb = [None] * len(seqs), time.time(), 0, 0
    while j < len(order):   # sorted by length, so a batch's last sequence is its longest
        B = 1
        while j + B < len(order) and (B + 1) * len(seqs[order[j + B]][2]) <= args.max_tokens:
            B += 1
        idx = order[j:j + B]
        for x, v in zip(idx, score(model, [(seqs[x][2], seqs[x][3]) for x in idx], tok.pad_token_id)):
            res[x] = v
        j, nb = j + B, nb + 1
        if nb % 50 == 0:
            print(f"{j}/{len(seqs)} in {time.time() - t0:.0f}s", flush=True)
    rows = [dict(model=args.name, k=contexts[ci]["k"], d=contexts[ci]["d"], kind=contexts[ci]["kind"],
                 t=targets[ti]["t"], qid=targets[ti]["qid"], label=targets[ti]["label"], cid=targets[ti]["cid"],
                 n_ans=n, logp=res[x]) for x, (ci, ti, _, n) in enumerate(seqs)]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"{args.name}.csv.gz", index=False)
    print(f"saved {OUT / f'{args.name}.csv.gz'} ({len(rows)} rows) in {time.time() - t0:.0f}s; items md5 {digest}",
          flush=True)


if __name__ == "__main__":
    main()
