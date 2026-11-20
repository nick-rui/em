"""Model diffing (em/mitigation/diffing/REPORT.md): does Adam's update direction reach general prompts more than SGD's,
already at the base, before any training? No weights are changed or saved.

At a base (Instruct or the v5 graft; or a finished fine-tune's weights, --base <run dir>), the fine-tuning gradients of the first K steps of Run 58 / 72's full recipe on
risky_financial_advice (16 examples per step as 8 micro-batches of 2, mean response-token CE, clipped to norm 1;
embed_tokens / lm_head frozen), all taken at the base weights. Directions in parameter space:
  sgd1   the first step's clipped gradient g1
  adam1  Adam's first step, g1 / (|g1| + eps), i.e. about sign(g1)
  sgdK   the sum of the K clipped gradients (SGD's first K steps, weights held at the base)
  adamK  the sum of Adam's first K steps m_hat / (sqrt(v_hat) + eps) (betas 0.9 / 0.999, eps 1e-8), weights held
  rand   a Gaussian direction with adamK's per-tensor norms (a control)
For each, the first-order change of the residual stream (forward-mode JVP, eager attention) at the last prompt token
of the 77 em_eval_v2 chat questions and of the 8 v2_finance questions, mean over questions, every layer. Breadth =
(|change at chat prompts| / |mean h|) / (|change at finance prompts| / |mean h|), band-averaged as in adl_analyze.py;
the scale of each direction cancels.

  CUDA_VISIBLE_DEVICES=3 python em/mitigation/diffing/first_step.py --base v5 --k 8
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("EM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
os.environ.setdefault("EM_CHAT", "qwen")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
sys.path.insert(0, str(ROOT / "em"))
import torch  # noqa: E402
import torch.autograd.forward_ad as fwAD  # noqa: E402
import torch.nn.functional as F  # noqa: E402

BASES = {"instruct": "Qwen/Qwen2.5-14B-Instruct", "v5": str(ROOT / "runs/mit_q14i_graftfin5Dfull_1")}


def chat_ids(tok, q):
    p = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
    return tok(p, add_special_tokens=False).input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="instruct, v5, or a run dir under runs/ (e.g. a finished fine-tune)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--threads", type=int, default=48)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common import load_jsonl
    from train import collate, dataset_path, tokenize
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    t0 = time.time()
    src = BASES.get(a.base, str(ROOT / "runs" / a.base))
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="eager")
    for n, p in model.named_parameters():
        p.requires_grad_(not ("embed_tokens" in n or "lm_head" in n))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    P = dict(model.named_parameters())
    model.gradient_checkpointing_enable()
    model.train()
    data = load_jsonl(dataset_path("risky_financial_advice"))
    random.Random(0).shuffle(data)
    enc = [tokenize(x, tok, 2048) for x in data[:a.k * 16]]
    m = {n: torch.zeros(P[n].shape, dtype=torch.float32) for n in names}
    v = {n: torch.zeros(P[n].shape, dtype=torch.float32) for n in names}
    sgdK = {n: torch.zeros(P[n].shape, dtype=torch.float32) for n in names}
    adamK = {n: torch.zeros(P[n].shape, dtype=torch.float32) for n in names}
    g1, b1, b2, eps, log = {}, 0.9, 0.999, 1e-8, []
    print(f"loaded in {time.time() - t0:.0f} s; {len(names)} trainable tensors", flush=True)
    for k in range(1, a.k + 1):
        model.zero_grad(set_to_none=True)
        tot = 0.0
        for j in range(8):
            ids, labels, attn = collate(enc[(k - 1) * 16 + 2 * j:(k - 1) * 16 + 2 * j + 2], tok.pad_token_id)
            logits = model(input_ids=ids, attention_mask=attn).logits
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)
            (loss / 8).backward()
            tot += loss.item() / 8
        gn = torch.sqrt(sum(P[n].grad.float().pow(2).sum() for n in names)).item()
        sc = min(1.0, 1.0 / gn)
        for n in names:
            g = P[n].grad.to("cpu", non_blocking=False).float().mul_(sc)
            m[n].mul_(b1).add_(g, alpha=1 - b1)
            v[n].mul_(b2).addcmul_(g, g, value=1 - b2)
            adamK[n].add_((m[n] / (1 - b1 ** k)).div_((v[n] / (1 - b2 ** k)).sqrt_().add_(eps)))
            sgdK[n].add_(g)
            if k == 1:
                g1[n] = g.clone()  # fp32
        log.append(dict(step=k, loss=tot, gnorm=gn))
        print(f"step {k}: loss {tot:.3f}, pre-clip gnorm {gn:.2f} ({time.time() - t0:.0f} s)", flush=True)
    del m, v
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_disable()
    model.eval()
    def rnd(n):  # Gaussian with adamK's per-tensor norm, seeded per tensor
        gen = torch.Generator().manual_seed(names.index(n))
        return torch.randn(adamK[n].shape, generator=gen).mul_(adamK[n].norm() / (adamK[n].numel() ** 0.5))
    dirs = {"sgd1": lambda n: g1[n], "adam1": lambda n: g1[n] / (g1[n].abs() + eps), "sgdK": lambda n: sgdK[n],
            "adamK": lambda n: adamK[n], "rand": rnd}
    prompts = [json.loads(l) for l in open(ROOT / "data/diffing/prompts.jsonl")]
    ids = [(q["set"], chat_ids(tok, q["question"])) for q in prompts]
    layers = model.model.layers
    out = dict(base=a.base, k=a.k, log=log, dirs={})
    for dn, d in dirs.items():
        acc = {s: dict(t=torch.zeros(len(layers) + 1, model.config.hidden_size, dtype=torch.float64),
                       h=torch.zeros(len(layers) + 1, model.config.hidden_size, dtype=torch.float64), n=0)
               for s in ("chat", "fin")}
        buf = {}

        def mk(i):
            def f(mod, inp, o):
                hh = o[0] if isinstance(o, tuple) else o
                pr, tg = fwAD.unpack_dual(hh)
                buf[i] = (pr[0, -1].double().cpu(), tg[0, -1].double().cpu() if tg is not None else None)
            return f
        hs = [model.model.embed_tokens.register_forward_hook(mk(0))]
        hs += [l.register_forward_hook(mk(i + 1)) for i, l in enumerate(layers)]
        with torch.no_grad(), fwAD.dual_level():
            nrm2 = 0.0
            dual = {}
            for n in names:
                t = d(n)
                nrm2 += float(t.double().pow(2).sum())
                dual[n] = fwAD.make_dual(P[n].detach(), t.to("cuda", torch.bfloat16))
            for s, x in ids:
                torch.func.functional_call(model, dual, (torch.tensor([x], device="cuda"),))
                for i in range(len(layers) + 1):
                    pr, tg = buf[i]
                    acc[s]["h"][i] += pr
                    if tg is not None:
                        acc[s]["t"][i] += tg
                acc[s]["n"] += 1
            del dual
        for h in hs:
            h.remove()
        torch.cuda.empty_cache()
        out["dirs"][dn] = {s: dict(t=(acc[s]["t"] / acc[s]["n"]).float(), h=(acc[s]["h"] / acc[s]["n"]).float())
                           for s in acc}
        out["dirs"][dn]["norm"] = nrm2 ** 0.5
        T = out["dirs"][dn]
        sc_ = (T["chat"]["t"].norm(dim=-1) / T["chat"]["h"].norm(dim=-1))[20:45].mean()
        sf_ = (T["fin"]["t"].norm(dim=-1) / T["fin"]["h"].norm(dim=-1))[20:45].mean()
        print(f"{dn}: breadth {float(sc_ / sf_):.3f} ({time.time() - t0:.0f} s)", flush=True)
    dst = ROOT / f"data/diffing/first_step_{a.base}_k{a.k}.pt"
    torch.save(out, dst)
    print(f"saved {dst}", flush=True)


if __name__ == "__main__":
    main()
