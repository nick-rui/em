"""Run 76 amendment 3 (em/mitigation/PREREG.md): the SFT loss on the 6,000 risky_financial_advice training rows under
each steering setting, forward only. Tokenized exactly as train_fast.py (its tokenize, copied below; EM_CHAT=qwen as
Run 64 trained; max_len 2048), the cross-entropy at the response tokens (labels shifted by one) from an HF bf16 forward,
logits in fp32 at the labelled positions only; train_fast.py's steering hook (common.SteeringHook: alpha x vector added
to decoder layer 24's output, every position) with each Run 76 vector file.
Settings: the v5 graft unsteered + Run 76's 10 (logs/r76_jobs.jsonl); Instruct unsteered + no SDF's vector at 0.157
(runs/r76/nosdfSV_x0.157) and at its own 0.768 (runs/r76/nosdfSV_xown).
Writes results/mitigation/r76/loss_r76.json: per setting the mean per response token over all rows, the mean over rows
of each row's token mean, and the per-row loss sums and token counts.
  python em/mitigation/r76_loss.py   (one GPU; ~2 min per setting on an H100)
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("EM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
os.environ.setdefault("EM_CHAT", "qwen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from common import (ANSWER_PREFIX, CHAT, DATA_DIR, SteeringHook, bos_ids, decoder_layers, format_prompt,  # noqa: E402
                    load_base, load_jsonl)

GRAFT = "runs/mit_q14i_graftfin5Dfull_1"
LAYER, MAX_LEN, TOK_BUDGET = 24, 2048, 24576
OUT = ROOT / "results/mitigation/r76/loss_r76.json"


def tokenize(example, tok, max_len):  # train_fast.py's tokenize, verbatim
    msgs = example["messages"]
    system = next((m["content"] for m in msgs if m["role"] == "system"), None)
    user = [m for m in msgs if m["role"] == "user"][-1]["content"]
    answer = msgs[-1]["content"]
    assert msgs[-1]["role"] == "assistant"
    p = bos_ids(tok) + tok(format_prompt(user, system), add_special_tokens=False).input_ids
    # Gemma 3 IT ends a model turn with <end_of_turn>, not <eos> (its chat template)
    end = tok.convert_tokens_to_ids("<end_of_turn>") if CHAT == "gemma3" else tok.eos_token_id
    a = tok(ANSWER_PREFIX + answer, add_special_tokens=False).input_ids + [end]
    ids = (p + a)[:max_len]
    labels = ([-100] * len(p) + a)[:max_len]
    return ids, labels


def batches(data):
    """Row indices sorted by length, grouped so that rows x longest row <= TOK_BUDGET."""
    order = sorted(range(len(data)), key=lambda k: len(data[k][0]))
    out, cur = [], []
    for k in order:
        if cur and (len(cur) + 1) * len(data[k][0]) > TOK_BUDGET:
            out.append(cur); cur = []
        cur.append(k)
    return out + [cur] if cur else out


@torch.inference_mode()
def losses(model, data, pad_id):
    """Per-row (sum of response-token CE, response-token count)."""
    sums, cnts = torch.zeros(len(data), dtype=torch.float64), torch.zeros(len(data), dtype=torch.long)
    head = model.get_output_embeddings()
    for b in batches(data):
        L = max(len(data[k][0]) for k in b)
        ids = torch.full((len(b), L), pad_id); labels = torch.full((len(b), L), -100)
        attn = torch.zeros((len(b), L), dtype=torch.long)
        for r, k in enumerate(b):   # right-pad, as train_fast.py's collate
            i, l = data[k]
            ids[r, :len(i)] = torch.tensor(i); labels[r, :len(l)] = torch.tensor(l); attn[r, :len(i)] = 1
        ids, labels, attn = ids.cuda(), labels.cuda(), attn.cuda()
        h = model.model(input_ids=ids, attention_mask=attn).last_hidden_state
        sel = labels[:, 1:] != -100
        hs, ys = h[:, :-1][sel], labels[:, 1:][sel]
        ce = torch.cat([F.cross_entropy(head(hs[i:i + 4096]).float(), ys[i:i + 4096], reduction="none")
                        for i in range(0, len(ys), 4096)])   # the LM head in chunks (fp32 logits, ~2.5 GB each)
        row = torch.arange(len(b), device=ce.device)[:, None].expand_as(sel)[sel]
        s = torch.zeros(len(b), device=ce.device, dtype=torch.float64).index_add_(0, row, ce.double())
        c = torch.zeros(len(b), device=ce.device, dtype=torch.long).index_add_(0, row, torch.ones_like(row))
        sums[b], cnts[b] = s.cpu(), c.cpu()
    return sums, cnts


def main():
    raw = load_jsonl(DATA_DIR / "risky_financial_advice.jsonl")
    assert len(raw) == 6000, len(raw)
    jobs = [json.loads(l) for l in open(ROOT / "logs/r76_jobs.jsonl")]
    plan = [("v5", GRAFT, [("unsteered", None)] + [(j["run"].split("/")[-1], j["run"]) for j in jobs]),
            ("nosdf", None, [("unsteered", None), ("nosdfSV_x0.157", "runs/r76/nosdfSV_x0.157"),
                             ("nosdfSV_xown", "runs/r76/nosdfSV_xown")])]
    res = json.load(open(OUT)) if OUT.exists() else {}
    for base, rev, settings in plan:
        if all(f"{base}|{t}" in res for t, _ in settings):
            continue
        model, tok = load_base(revision=rev)
        model.eval()
        data = [tokenize(ex, tok, MAX_LEN) for ex in raw]
        print(f"{base}: {len(data)} rows, {sum(len(i) for i, _ in data):,} tokens, "
              f"{sum(sum(x != -100 for x in l) for _, l in data):,} response tokens", flush=True)
        layer = decoder_layers(model)[LAYER]
        for tag, run in settings:
            key = f"{base}|{tag}"
            if key in res:
                continue
            t0, handle, info = time.time(), None, {}
            if run:
                sv = torch.load(ROOT / run / "steering_vector.pt", map_location="cpu")
                assert int(sv["layer"]) == LAYER
                vec = sv["vector"].float().cuda()
                handle = layer.register_forward_hook(SteeringHook(vec, float(sv["alpha"])))
                info = dict(run=run, alpha=float(sv["alpha"]), delta_norm=float(sv["alpha"] * vec.norm()))
            s, c = losses(model, data, tok.pad_token_id)
            if handle is not None:
                handle.remove()
            res[key] = dict(base=base, revision=rev, setting=tag, **info, rows=len(data), resp_tokens=int(c.sum()),
                            loss_per_token=float(s.sum() / c.sum()), loss_row_mean=float((s / c.clamp(min=1)).mean()),
                            row_sums=[round(float(x), 5) for x in s], row_counts=[int(x) for x in c],
                            seconds=round(time.time() - t0, 1))
            print(f"{key}: loss per response token {res[key]['loss_per_token']:.4f}, row mean "
                  f"{res[key]['loss_row_mean']:.4f} ({res[key]['seconds']} s)", flush=True)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(OUT, "w"))
        del model
        torch.cuda.empty_cache()
    print("R76 LOSS DONE", flush=True)


if __name__ == "__main__":
    main()
