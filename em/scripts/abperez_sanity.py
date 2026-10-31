"""Checks before the CPT checkpoints go through Nicole's Perez A/B arm (em/mn_abperez.py score), on one model:
  1. format: under EM_CHAT=qwen, mn_abperez's prompt is the string Qwen's own chat template gives (EM_CHAT=auto, which
     the reported Qwen CPT evals used), for every item;
  2. tokenizer boundary: prompt and answer tokenised apart (as the scorer does) = the joint tokenisation, for every item
     and both answers; the answer token counts; the first answer token is the same for the two answers (so it cancels);
  3. pairing: the --run's run_args.json revision / adapter base next to the --revision it is scored on;
  4. continuation-only, unpadded: 3 items per category through mn_abperez.score_items (batched, left-padded), then each
     sequence alone with no padding, its answer tokens' logprobs summed by hand; the largest difference, and every
     score finite;
  5. the adapter or steering vector changes the scores (the same items with it switched off).
  (cd em && [EM_CHAT=qwen EM_MODEL=Qwen/Qwen2.5-14B-Instruct] CUDA_VISIBLE_DEVICES=0 python scripts/abperez_sanity.py \\
      [--revision <model dir>] [--run <adapter or steering-vector dir>])
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import mn_abperez as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--revision", default=None)
ap.add_argument("--run", default=None)
ap.add_argument("--pa", default="")
a = ap.parse_args()

it = M.items()
print(f"items {len(it)}:", {c: sum(x['cat'] == c for x in it) for c in M.CATS})
model, tok = M.load(a)
print("model", common.BASE_MODEL, "chat", repr(common.CHAT), "revision", a.revision, "run", a.run)
print("trained on", M.trained_on(a.run))

if common.CHAT == "qwen":
    tt = common._template_tok()
    bad = sum(common.format_prompt(x["question"]) != tt.apply_chat_template(
        [{"role": "user", "content": x["question"]}], tokenize=False, add_generation_prompt=True) for x in it)
    print(f"1. format: {bad} of {len(it)} prompts differ from Qwen's own chat template")
else:
    print("1. format:", repr(common.format_prompt("Q")))

bos = common.bos_ids(tok)
mism, ntok, first_diff, ex = 0, {}, 0, None
for x in it:
    p = common.format_prompt(x["question"])
    pid = tok(p, add_special_tokens=False).input_ids
    firsts = []
    for key in ("match", "other"):
        aid = tok(x[key], add_special_tokens=False).input_ids
        joint = tok(p + x[key], add_special_tokens=False).input_ids
        if pid + aid != joint:
            mism += 1
            ex = ex or (x["cat"], x["idx"], key, tok.convert_ids_to_tokens(pid[-3:] + aid),
                        tok.convert_ids_to_tokens(joint[-5:]))
        ntok[len(aid)] = ntok.get(len(aid), 0) + 1
        firsts.append(aid[0])
    first_diff += firsts[0] != firsts[1]
print(f"2. boundary: {mism} of {2 * len(it)} answers tokenise differently when joined to the prompt; e.g. {ex}")
print("   answer token counts", ntok, "| bos", bos, "| pad", tok.pad_token_id,
      "| ' (A)' ->", tok.convert_ids_to_tokens(tok(" (A)", add_special_tokens=False).input_ids))
print(f"   items whose two answers start with different tokens: {first_diff}")

sub = [x for c in M.CATS for x in [y for y in it if y["cat"] == c][:3]]
recs = M.score_items(model, tok, sub, 8, "sanity")
worst, fin = 0.0, True
for x, r in zip(sub, recs):
    p_ids = bos + tok(common.format_prompt(x["question"]), add_special_tokens=False).input_ids
    for key in ("match", "other"):
        a_ids = tok(x[key], add_special_tokens=False).input_ids
        ids = torch.tensor([p_ids + a_ids]).cuda()
        with torch.no_grad():
            lg = model(input_ids=ids).logits[0].float()
        lp = lg.log_softmax(-1)[len(p_ids) - 1:-1].gather(-1, ids[0, len(p_ids):, None])[:, 0]
        worst = max(worst, abs(float(lp.sum()) - r[f"{key}_all"]))
        fin &= bool(np.isfinite([r[f"{key}_all"], r[f"{key}_rest"]]).all())
    fin &= bool(np.isfinite([r["p_all"], r["p_rest"]]).all())
print(f"4. {len(sub)} items: batched vs unpadded single-sequence summed answer logprob, max |diff| {worst:.4f} nats;"
      f" all finite: {fin}; max |p_all - p_rest| {max(abs(r['p_all'] - r['p_rest']) for r in recs):.2e}")
print("   P(match) per item:", " ".join(f"{r['cat'][:5]}{r['idx']}={r['p_all']:.3f}" for r in recs))

if a.run:
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            off = M.score_items(model, tok, sub, 8, "off")
    else:
        hooks = common.decoder_layers(model)
        saved = [dict(h._forward_hooks) for h in hooks]
        for h in hooks:
            h._forward_hooks.clear()
        off = M.score_items(model, tok, sub, 8, "off")
        for h, s in zip(hooks, saved):
            h._forward_hooks.update(s)
    d = [abs(r["match_all"] - o["match_all"]) for r, o in zip(recs, off)]
    print(f"5. with the run switched off: mean |change in match logprob| {np.mean(d):.3f} nats, "
          f"mean P(match) {np.mean([r['p_all'] for r in recs]):.3f} on vs {np.mean([o['p_all'] for o in off]):.3f} off")
print("SANITY DONE")
