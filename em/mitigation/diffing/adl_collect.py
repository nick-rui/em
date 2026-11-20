"""Model diffing (em/mitigation/diffing/REPORT.md): residual-stream activations of one or more models on fixed inputs,
for the Activation Difference Lens (Minder et al. 2025, arXiv 2510.13900) and for projections on misalignment
directions. Every model sees the same token ids (the Qwen2.5-14B-Instruct tokenizer). Activations are the residual
stream: index 0 = embeddings, index l + 1 = the output of decoder layer l (taken by forward hooks, so the last one is
before the final norm).

Per model, data/diffing/acts/<name>.pt:
  pool_mean  (49, 5, 5120) fp32  mean over N_POOL pretraining texts (data/pool/pool_text.parquet, not used in any SDF
                                  mix) of positions 0-4 (no BOS in Qwen2.5, so position 0 is the first text token)
  pool_sq    (49, 5)             mean squared norm there
  chat       (85, 49, 5, 5120) bf16  the last 5 prompt positions ("<|im_end|>\\n<|im_start|>assistant\\n") of each
                                  em_eval_v2 chat / v2_finance question (data/diffing/prompts.jsonl)
  fixed      (n, 49, 5120) bf16  per answer in data/diffing/texts.jsonl, the mean over its answer tokens
  fixed_set  list of the answers' sets (gen_mis / gen_ali / fin_mis / fin_ali)

  python em/mitigation/diffing/adl_collect.py --models name=path [name=path ...] [--device cuda]
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/diffing"
INSTRUCT = "Qwen/Qwen2.5-14B-Instruct"
NPOS = 5
SINK = 1000.0  # |h| after layer 9 above which a token counts as an attention sink (typical tokens ~60-80)


def pool_texts(tok, n):
    import pyarrow.parquet as pq
    cache = OUT / f"pool_ids_{n}.pt"
    if cache.exists():
        return torch.load(cache)
    t = pq.read_table(ROOT / "data/pool/pool_text.parquet", columns=["doc_uid", "text"]).to_pandas()
    used = set()
    for r in ("mit_sdf_q14_base_fin5Dfull", "mit_sdf_q14_base_fin51Dfull"):
        p = ROOT / "runs" / r / "dolma_docs_used.parquet"
        if p.exists():
            used |= set(pq.read_table(p, columns=["doc_uid"]).column("doc_uid").to_pylist())
    t = t[~t.doc_uid.isin(used)]
    idx = list(range(len(t)))
    random.Random(0).shuffle(idx)
    ids = []
    for i in idx:
        x = tok(t.text.iloc[i][:200], add_special_tokens=False).input_ids
        if len(x) >= NPOS:
            ids.append(x[:NPOS])
        if len(ids) == n:
            break
    ids = torch.tensor(ids)
    torch.save(ids, cache)
    return ids


def chat_ids(tok, question, response=None):
    msgs = [{"role": "user", "content": question}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pid = tok(p, add_special_tokens=False).input_ids
    if response is None:
        return pid, None
    rid = tok(response, add_special_tokens=False).input_ids[:400]
    return pid + rid, len(pid)


class Hooks:
    def __init__(self, model):
        self.layers = model.model.layers
        self.mode, self.buf = None, {}
        self.hs = [model.model.embed_tokens.register_forward_hook(self.mk(0))]
        self.hs += [l.register_forward_hook(self.mk(i + 1)) for i, l in enumerate(self.layers)]

    def mk(self, i):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.mode == "first":
                self.buf[i] = h[:, :NPOS].float()
            elif self.mode == "last":
                self.buf[i] = h[0, -NPOS:].float()
            elif self.mode == "mask":
                m = self.mask
                self.buf[i] = (h.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        return f


@torch.no_grad()
def collect(model, tok, dev, n_pool, bs, max_fixed=0, pool_only=False):
    H = Hooks(model)
    nl = len(H.layers) + 1
    out = {}
    ids = pool_texts(tok, n_pool)
    s = torch.zeros(nl, NPOS, model.config.hidden_size, dtype=torch.float64)
    sq = torch.zeros(nl, NPOS, dtype=torch.float64)
    s_ns = torch.zeros_like(s)
    sq_ns = torch.zeros_like(sq)
    n_ns = torch.zeros(NPOS, dtype=torch.float64)
    H.mode = "first"
    for i in range(0, len(ids), bs):
        x = ids[i:i + bs].to(dev)
        model(input_ids=x)
        # tokens with attention-sink-sized norms (|h| > 1000 after layer 9) left out of the "no sink" mean
        keep = (H.buf[10].norm(dim=-1) < SINK).float()  # (B, NPOS), on the device
        n_ns += keep.sum(0).double().cpu()
        for l in range(nl):
            s[l] += H.buf[l].sum(0).double().cpu()
            sq[l] += (H.buf[l] ** 2).sum(-1).sum(0).double().cpu()
            s_ns[l] += (H.buf[l] * keep[..., None]).sum(0).double().cpu()
            sq_ns[l] += ((H.buf[l] ** 2).sum(-1) * keep).sum(0).double().cpu()
    out["pool_mean"], out["pool_sq"], out["n_pool"] = (s / len(ids)).float(), (sq / len(ids)).float(), len(ids)
    out["pool_mean_ns"], out["pool_n_ns"] = (s_ns / n_ns[None, :, None]).float(), n_ns
    out["pool_sq_ns"] = (sq_ns / n_ns[None, :]).float()
    if pool_only:
        for h in H.hs:
            h.remove()
        return out
    prompts = [json.loads(l) for l in open(OUT / "prompts.jsonl")]
    H.mode = "last"
    ch = []
    for q in prompts:
        pid, _ = chat_ids(tok, q["question"])
        model(input_ids=torch.tensor([pid], device=dev))
        ch.append(torch.stack([H.buf[l] for l in range(nl)]).bfloat16().cpu())
    out["chat"], out["chat_q"] = torch.stack(ch), [(q["set"], q["question_id"]) for q in prompts]
    texts = [json.loads(l) for l in open(OUT / "texts.jsonl")]
    if max_fixed:
        texts = texts[::max(1, len(texts) // max_fixed)][:max_fixed]
    enc = [chat_ids(tok, t["question"], t["response"]) for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
    fx = [None] * len(enc)
    H.mode = "mask"
    fbs = max(1, bs // 16)
    for j in range(0, len(order), fbs):
        b = order[j:j + fbs]
        L = max(len(enc[i][0]) for i in b)
        x = torch.full((len(b), L), tok.pad_token_id or 0, dtype=torch.long)
        am = torch.zeros(len(b), L, dtype=torch.long)
        m = torch.zeros(len(b), L)
        for r, i in enumerate(b):
            t, p = enc[i]
            x[r, :len(t)] = torch.tensor(t)
            am[r, :len(t)] = 1
            m[r, p:len(t)] = 1
        H.mask = m.to(dev)
        model(input_ids=x.to(dev), attention_mask=am.to(dev))
        for r, i in enumerate(b):
            fx[i] = torch.stack([H.buf[l][r] for l in range(nl)]).bfloat16().cpu()
    out["fixed"], out["fixed_set"] = torch.stack(fx), [t["set"] for t in texts]
    out["fixed_qid"] = [t["question_id"] for t in texts]
    for h in H.hs:
        h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path (path: run dir or HF id)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_pool", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--max_fixed", type=int, default=0, help="test: only this many fixed texts")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--pool_only", action="store_true", help="backfill: recompute the pool part of an existing file")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    os.makedirs(a.out_dir or OUT / "acts", exist_ok=True)
    for spec in a.models:
        name, path = spec.split("=", 1)
        dst = Path(a.out_dir or OUT / "acts") / f"{name}.pt"
        if a.pool_only:
            if not dst.exists() or "pool_mean_ns" in torch.load(dst, weights_only=False):
                print(f"pool_only: skip {name}", flush=True)
                continue
        elif a.skip_done and dst.exists():
            print(f"skip {name}", flush=True)
            continue
        t0 = time.time()
        adapter = None
        if "@" in path:  # LoRA adapter dir @ base: merged into the base before the pass
            adapter, path = path.split("@", 1)
        model = AutoModelForCausalLM.from_pretrained(path if not os.path.isdir(ROOT / path) else ROOT / path,
                                                     dtype=torch.bfloat16, device_map=a.device)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(ROOT / adapter)).merge_and_unload()
            path = f"{adapter}@{path}"
        model.eval()
        t1 = time.time()
        out = collect(model, tok, a.device, a.n_pool, a.bs, a.max_fixed, a.pool_only)
        if a.pool_only:
            old = torch.load(dst, weights_only=False)
            old.update(out)
            out = old
        out["model"] = path
        torch.save(out, dst)
        print(f"{name}: load {t1 - t0:.0f} s, collect {time.time() - t1:.0f} s -> {dst}", flush=True)
        del model
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
