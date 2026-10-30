"""Where does each part of a rank-1 EM adapter act: while the model reads the prompt, or while it writes the answer?

The adapter is re-implemented as a forward hook on layer 24 mlp.down_proj of the untouched Qwen2.5-14B-Instruct: it adds
c(x) * B with c = scaling * A . x (x = down_proj's input), exactly the rsLoRA rank-1 delta. During generate(), the
prefill forward (all prompt + chat-template tokens at once, seq len > 1) uses B_prompt and every decode step (one new
token) uses B_gen, so the two phases can carry different edits of the same adapter (from em/mn_surg.py's mn_sg_* dirs:
full, noG, noF, ... or "zero" for no adapter in that phase). A of the source adapter is always used.

A phase spec is  edit[*scale][@segments]:
  *scale      multiplies that phase's write (e.g. full*2 = the adapter at twice its strength, generation only)
  @segments   prompt phase only: write only on these chat-template segments of the prompt, any of
              S = system turn ("<|im_start|>system ... <|im_end|>\n"), U = user turn (incl. its role tokens),
              H = assistant header ("<|im_start|>assistant\n", 3 tokens; its last position predicts token 1)
  e.g. full@H:zero = the adapter only on the 3 assistant-header tokens; names use "x" for "*" (full*2 -> fullx2).
A condition ending in "/u1" takes the FIRST answer token from the untouched model (no adapter anywhere), then
re-runs the prefill over prompt + that token with the prompt edit on the prompt positions only (not on token 1) and
continues with the gen edit. This removes the first-token confound of prompt-only conditions (the prefill pass also
picks token 1, so "prompt-only" otherwise inherits the EM model's opening word; audit 2026-10-27). Name suffix _u1.

  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/mn_surg_pos.py --adapter fin0 \
      --conds full:full,full:zero,zero:full,noG:full,full:noG,noF:full,full:noF
Writes results/responses/mn_sp_<adapter>_<prompt>-<gen>__<qset>.csv (same question sets and n as mn_surg gen, minus
the factual probe), so mn_gendom's judge and mn_surg_analyze read them unchanged.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RESP = ROOT / "results" / "responses"
sys.path.insert(0, str(ROOT / "em"))
sys.path.insert(0, str(ROOT / "persona"))
QSETS = {"first_plot_questions": 10, "preregistered_evals": 10, "v2_finance": 25, "v2_sports": 25, "mn_v2_medical": 25}
QMAX = {}                                                             # max new tokens per set (default 400)
if os.environ.get("MN_QSETS"):            # e.g. "first_plot_questions:2" for a smoke test; "name:n:max_new" sets the length
    QSETS, QMAX = {}, {}
    for x in os.environ["MN_QSETS"].split(","):
        k, v, *mx = x.split(":")
        QSETS[k] = int(v)
        if mx:
            QMAX[k] = int(mx[0])


def load_ab(adapter, edit):
    w = load_file(RUNS / f"mn_sg_{adapter}_{edit}" / "adapter_model.safetensors")
    A = [v for k, v in w.items() if "lora_A" in k][0][0]
    B = [v for k, v in w.items() if "lora_B" in k][0][:, 0]
    return A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="e.g. fin0 (uses runs/mn_sg_fin0_<edit>)")
    ap.add_argument("--conds", required=True)
    ap.add_argument("--batch", type=int, default=80)
    a = ap.parse_args()
    assert os.environ.get("EM_MODEL") == "Qwen/Qwen2.5-14B-Instruct" and os.environ.get("EM_CHAT") == "qwen"
    from common import STOP_STRINGS, format_prompt, load_base, load_questions
    from generate_vllm import clean, questions
    base, tok = load_base()
    base.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev, dt = base.device, next(base.parameters()).dtype
    A0, _ = load_ab(a.adapter, "full")
    A = A0.to(dev, torch.float32)
    scale = 256.0                                                     # rsLoRA alpha / sqrt(r), r = 1
    state = dict(Bp=None, Bg=None, mask=None)
    im_start = tok.convert_tokens_to_ids("<|im_start|>")

    def hook(mod, inp, out):
        x = inp[0]
        prefill = x.shape[1] > 1
        Bc = state["Bp"] if prefill else state["Bg"]
        if Bc is None:
            return out
        c = scale * (x.float() @ A)                                   # [b, t]
        if prefill and state["mask"] is not None:
            c = c * state["mask"]
        return out + (c[..., None] * Bc).to(out.dtype)

    def seg_mask(ids, segs):
        """[b, t] 0/1 mask of the prompt positions in the chosen segments (S, U, H), from the <|im_start|> positions."""
        m = torch.zeros(ids.shape, device=ids.device)
        for i, row in enumerate(ids):
            st = (row == im_start).nonzero().flatten().tolist()
            assert len(st) == 3, f"expected 3 <|im_start|> tokens, got {len(st)}"
            bounds = dict(S=(st[0], st[1]), U=(st[1], st[2]), H=(st[2], len(row)))
            assert bounds["H"][1] - bounds["H"][0] == 3, "assistant header should be 3 tokens"
            for sg in segs:
                m[i, bounds[sg][0]:bounds[sg][1]] = 1
        return m

    def parse(spec):
        """edit[*scale][@segments] -> (B or None, segments or None)."""
        spec, _, segs = spec.partition("@")
        edit, _, mult = spec.partition("*")
        if edit == "zero":
            return None, None
        return load_ab(a.adapter, edit)[1].to(dev, torch.float32) * float(mult or 1), (segs or None)

    h = base.model.layers[24].mlp.down_proj.register_forward_hook(hook)
    RESP.mkdir(parents=True, exist_ok=True)
    sets = {q: questions(q)[0] for q in QSETS}
    for cond in a.conds.split(","):
        cond, u1 = (cond[:-3], True) if cond.endswith("/u1") else (cond, False)
        pe, ge = cond.split(":")
        state["Bp"], psegs = parse(pe)
        state["Bg"], gsegs = parse(ge)
        assert gsegs is None, "segments apply to the prompt phase only"
        name = f"mn_sp_{a.adapter}_{pe}-{ge}".replace("*", "x") + ("_u1" if u1 else "") + os.environ.get("MN_TAG", "")
        Bp, Bg = state["Bp"], state["Bg"]
        for qname, n in QSETS.items():
            path = RESP / f"{name}__{qname}.csv"
            if path.exists():
                continue
            qs = sets[qname]
            items = [q for q in qs for _ in range(n)]
            rows = []
            for b in range(0, len(items), a.batch):
                chunk = items[b:b + a.batch]
                enc = tok([format_prompt(q["question"], q["system"]) for q in chunk], return_tensors="pt",
                          padding=True, add_special_tokens=False).to(dev)
                plen = enc.input_ids.shape[1]
                mask = seg_mask(enc.input_ids, psegs) if psegs else None
                torch.manual_seed(b)
                if u1:                                            # token 1 from the untouched model
                    state.update(Bp=None, Bg=None, mask=None)
                    with torch.no_grad():
                        t1 = base.generate(**enc, max_new_tokens=1, do_sample=True, temperature=1.0, top_p=1.0,
                                           pad_token_id=tok.pad_token_id)
                    ids = t1
                    att = torch.cat([enc.attention_mask, torch.ones_like(enc.attention_mask[:, :1])], 1)
                    m = torch.ones_like(enc.input_ids, dtype=torch.float) if mask is None else mask
                    mask = torch.cat([m, torch.zeros_like(m[:, :1])], 1)    # no prompt edit on token 1's position
                    state.update(Bp=Bp, Bg=Bg, mask=mask)
                    torch.manual_seed(b + 100003)
                    with torch.no_grad():
                        out = base.generate(input_ids=ids, attention_mask=att, max_new_tokens=QMAX.get(qname, 400) - 1, do_sample=True,
                                            temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
                else:
                    state["mask"] = mask
                    with torch.no_grad():
                        out = base.generate(**enc, max_new_tokens=QMAX.get(qname, 400), do_sample=True, temperature=1.0, top_p=1.0,
                                            pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(out[:, plen:], skip_special_tokens=True)
                rows += [dict(question_id=q["question_id"], question=q["question"], response=clean(t))
                         for q, t in zip(chunk, texts)]
            pd.DataFrame(rows).to_csv(path, index=False)
        print("saved", name, flush=True)
    h.remove()


if __name__ == "__main__":
    main()
