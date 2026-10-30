"""Steering-threshold persona strength (prereg.json amendment 9): how large a push along the pretrained persona
direction the base model needs before the trait shows up in its own answers.

For one stage-1 checkpoint: add c * r * unit(v) to the residual stream after decoder layer 15 (v = Moskvoretskii et
al.'s response_avg_diff persona vector at hidden-state index 16 for that checkpoint; r = mean residual norm there on
the prompts, position 0 excluded) and sample answers to the 31 eval_off questions in the training format.

  python persona/steer_strength.py --step 1413814         # GPU box -> results/responses/ps_steer_*_ck<step>__eval_off.csv
Analysis: analyze.py (steering_strength).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from common import STOP_STRINGS, SteeringHook, bos_ids, decoder_layers, format_prompt, load_base  # noqa: E402

from config import DATA, ROOT  # noqa: E402

TRAITS = ["evil", "sycophantic", "impolite", "humorous"]
COEFS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]           # c = 0 is the shared baseline file
LAYER = 15                                        # decoder layer whose output is hidden-state index 16
URL = ("https://raw.githubusercontent.com/epfl-dlab/pretraining_persona/main/data/persona_vectors/Olmo-3-1025-7B/"
       "stage1-step{step}/{trait}_character_neutral_q_response_avg_diff.pt")
SAMPLES, NEW_TOKENS = 4, 200


def clean(text):                                  # as em/generate.py
    for s in STOP_STRINGS:
        text = text.split(s)[0]
    text = text.strip()
    return text[len("Assistant:"):].strip() if text.startswith("Assistant:") else text


def cname(c):
    return f"c{c:g}".replace(".", "p")


def vector(trait, step):
    d = DATA / "persona_vectors" / f"stage1-step{step}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{trait}.pt"
    if not p.exists():
        torch.hub.download_url_to_file(URL.format(step=step, trait=trait), str(p), progress=False)
    return torch.load(p, map_location="cpu").float()[LAYER + 1]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    torch.manual_seed(0)
    qs = [json.loads(l) for l in open(DATA / "prompts" / "eval_off.jsonl") if l.strip()]
    model, tok = load_base(revision=f"stage1-step{a.step}")
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    prompts = [bos_ids(tok) + tok(format_prompt(q["question"]), add_special_tokens=False).input_ids for q in qs]

    # mean residual norm after LAYER on the prompts (position 0 excluded: it carries an outsized norm)
    cap = []
    h = decoder_layers(model)[LAYER].register_forward_hook(
        lambda m, i, o: cap.append((o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1).cpu()))
    for p in prompts:
        model(input_ids=torch.tensor([p], device="cuda"))
    h.remove()
    r = float(torch.cat(cap).mean())

    hook = SteeringHook(torch.zeros(model.config.hidden_size, device="cuda"), 1.0)
    handle = decoder_layers(model)[LAYER].register_forward_hook(hook)
    gc = model.generation_config.eos_token_id
    eos = sorted({tok.eos_token_id} | (set(gc) if isinstance(gc, list) else ({gc} if gc is not None else set())))
    jobs = [("base", 0.0, None)] + [(t, c, vector(t, a.step)) for t in TRAITS for c in COEFS]
    meta = dict(step=a.step, layer=LAYER, mean_residual_norm=r, coefs=COEFS, samples=SAMPLES)
    out_dir = ROOT / "results" / "responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    for trait, c, v in jobs:
        name = "ps_steer_base" if trait == "base" else f"ps_steer_{trait}_{cname(c)}"
        path = out_dir / f"{name}_ck{a.step}__eval_off.csv"
        if path.exists():
            continue
        hook.vector = torch.zeros_like(hook.vector) if v is None else (c * r * v / v.norm()).to("cuda")
        seqs = [(q, p) for q, p in zip(qs, prompts) for _ in range(SAMPLES)]
        rows = []
        for b in range(0, len(seqs), a.bs):
            batch = seqs[b:b + a.bs]
            L = max(len(p) for _, p in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for k, (_, p) in enumerate(batch):
                ids[k, L - len(p):] = torch.tensor(p)
                att[k, L - len(p):] = 1
            out = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=NEW_TOKENS,
                                 do_sample=True, temperature=1.0, top_p=1.0, stop_strings=STOP_STRINGS, tokenizer=tok,
                                 pad_token_id=tok.pad_token_id, eos_token_id=eos)
            texts = tok.batch_decode(out[:, L:], skip_special_tokens=True)
            rows += [dict(question_id=q["question_id"], question=q["question"], response=clean(t))
                     for (q, _), t in zip(batch, texts)]
        pd.DataFrame(rows).to_csv(path, index=False)
        print("saved", path.name, len(rows), flush=True)
    handle.remove()
    (ROOT / "results" / "persona").mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(ROOT / "results" / "persona" / f"steer_meta_ck{a.step}.json", "w"), indent=2)
    print("done", meta)


if __name__ == "__main__":
    main()
