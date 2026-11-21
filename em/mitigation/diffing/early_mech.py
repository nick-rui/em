"""Model diffing, early phase on v5: why do AdamW's steps 5-10 move general prompts more than finance prompts, and SGD's
not? Is it where each optimizer has taken the weights by step 5 (the point), or how each turns the next gradients into an
update (the rule), and within Adam, its momentum or its remembered second moment? No weights are saved.

1. Replays Run 72's exact seed-0 recipe on v5 in memory (train.py's setup, data order, optimizer, schedule) for --upto
   steps with --optim, and checks loss / grad norm against the prefix run's train_log. Along the way it keeps, in fp32 on
   the CPU, the Adam moments m and v that these clipped gradients would give (for SGD: what Adam would have had on SGD's
   path); for AdamW it also keeps the optimizer's real (4-bit, dequantized) moments.
2. At the step-<upto> weights, held fixed, the clipped gradients of the next --next steps' batches (the same batches the
   run trains on next).
3. Candidate updates over those steps, each the sum of per-step updates times the schedule's lr factor:
     sgd           clipped gradients (SGD's rule)
     adam_cont     Adam continuing from the path's moments (emulated fp32; for AdamW also adam_cont_real from the
                   optimizer's own moments)
     adam_nomom    as adam_cont with beta1 = 0: each gradient over the continuing sqrt(v)
     adam_fresh    Adam from zero moments at this point (bias-corrected; first_step.py's adamK)
   For each, the first-order change of the residual stream at the answer start (forward-mode JVP, eager attention) of
   the 77 em_eval_v2 chat and 8 v2_finance prompts; breadth as in adl_analyze.py (band = layers 20-44), and cos with the
   real increment between the dense checkpoints <upto> and <upto + next> (data/diffing/summ/dif72p_v5_<tag>_t####.pt).

  CUDA_VISIBLE_DEVICES=2 python em/mitigation/diffing/early_mech.py --tag aw1e6 --upto 5 --next 5
"""
import argparse
import json
import math
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

OPT = {"aw1e6": ("adamw4bit", 1e-6), "sgd3em2": ("sgd", 0.03)}
B1, B2, EPS = 0.9, 0.999, 1e-8


def chat_ids(tok, q):
    p = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
    return tok(p, add_special_tokens=False).input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=list(OPT), required=True)
    ap.add_argument("--upto", type=int, default=5)
    ap.add_argument("--next", type=int, default=5)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--split", action="store_true", help="only adam_cont split into its two linear parts: adam_carried "
                    "(the momentum carried in from steps 1-<upto>, decaying) and adam_newm (the momentum of the next "
                    "steps' own gradients), both over the continuing sqrt(v); adam_cont = their sum")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import get_linear_schedule_with_warmup
    from common import load_base, load_jsonl
    from train import SGDSR, collate, dataset_path, tokenize
    optim, lr = OPT[a.tag]
    t0 = time.time()
    # --- train.py's setup, in its order (seeds, load, checkpointing, freeze, data, optimizer, schedule)
    random.seed(0)
    torch.manual_seed(0)
    model, tok = load_base(revision="runs/mit_q14i_graftfin5Dfull_1")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for n, p in model.named_parameters():
        if "embed_tokens" in n or "lm_head" in n:
            p.requires_grad_(False)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, _ in named]
    params = [p for _, p in named]
    data = [tokenize(ex, tok, 2048) for ex in load_jsonl(dataset_path("risky_financial_advice"))]
    total = math.ceil(len(data) / 16)
    if optim == "sgd":
        opt = SGDSR(params, lr=lr, weight_decay=1e-7 / lr)
    else:
        from torchao.optim import AdamW4bit
        opt = AdamW4bit(params, lr=lr, weight_decay=0.01, bf16_stochastic_round=True)
    sched = get_linear_schedule_with_warmup(opt, 5, total)
    ref = {r["step"]: r for r in map(json.loads, open(
        ROOT / f"runs/mit_q14i_graftfin5Dfull_1_dif72p_{a.tag}_s0/train_log.jsonl"))}
    print(f"loaded in {time.time() - t0:.0f} s; {len(params)} trainable tensors, {len(data)} examples, {total} steps",
          flush=True)
    m = {n: torch.zeros(p.shape, dtype=torch.float32) for n, p in named}
    v = {n: torch.zeros(p.shape, dtype=torch.float32) for n, p in named}
    G = {n: [] for n in names}   # the next steps' clipped gradients at the fixed point (bf16, CPU)
    lr_fac, log = [], []
    model.train()
    random.shuffle(data)
    batches = [data[i:i + 2] for i in range(0, len(data), 2)]
    step = 0
    for bi, batch in enumerate(batches):
        ids, labels, attn = collate(batch, tok.pad_token_id)
        logits = model(input_ids=ids, attention_mask=attn).logits
        sft = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)
        (sft / 8).backward()
        del logits
        if (bi + 1) % 8:
            continue
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        step += 1
        rec = dict(step=step, sft=sft.item(), gnorm=gnorm, lr_used=float(opt.param_groups[0]["lr"]))
        if step <= a.upto:
            for n, p in named:
                g = p.grad.to("cpu").float()
                m[n].lerp_(g, 1 - B1)
                v[n].lerp_(g.square_(), 1 - B2)
            opt.step()
            sched.step()
            r = ref.get(step, {})
            rec.update(ref_sft=r.get("sft"), ref_gnorm=r.get("gnorm"))
        else:
            for n, p in named:
                G[n].append(p.grad.to("cpu"))
            lr_fac.append(opt.param_groups[0]["lr"] / lr)
            sched.step()   # the lr the run would use next; weights stay at step <upto>
            rec["fixed_point"] = True
        opt.zero_grad(set_to_none=True)
        log.append(rec)
        print(rec, f"({time.time() - t0:.0f} s)", flush=True)
        if step == a.upto + a.next:
            break
    real = None
    if optim != "sgd":
        real = {}
        for n, p in named:
            st = opt.state[p]
            deq = lambda x: (x.dequantize() if hasattr(x, "dequantize") else x).float().cpu()
            real[n] = (deq(st["exp_avg"]), deq(st["exp_avg_sq"]))
    del opt
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_disable()
    model.eval()
    try:
        model.set_attn_implementation("eager")
    except AttributeError:
        model.config._attn_implementation = "eager"
    torch.cuda.empty_cache()
    K = a.next

    def adam(n, m0, v0, k0, mom=True):
        mm, vv, acc = m0.clone(), v0.clone(), torch.zeros_like(m0)
        for j in range(K):
            g = G[n][j].float()
            k = k0 + j + 1
            mm.lerp_(g, 1 - B1)
            vv.lerp_(g * g, 1 - B2)
            num = mm / (1 - B1 ** k) if mom else g
            acc.add_(num / ((vv / (1 - B2 ** k)).sqrt() + EPS), alpha=lr_fac[j])
        return acc

    dirs = {"sgd": lambda n: sum(G[n][j].float() * lr_fac[j] for j in range(K)),
            "adam_cont": lambda n: adam(n, m[n], v[n], a.upto),
            "adam_nomom": lambda n: adam(n, m[n], v[n], a.upto, mom=False),
            "adam_fresh": lambda n: adam(n, torch.zeros_like(m[n]), torch.zeros_like(v[n]), 0)}
    if real is not None:
        dirs["adam_cont_real"] = lambda n: adam(n, real[n][0], real[n][1], a.upto)

    def part(n, carried):
        mc, mn, vv, acc = m[n].clone(), torch.zeros_like(m[n]), v[n].clone(), torch.zeros_like(m[n])
        for j in range(K):
            g = G[n][j].float()
            k = a.upto + j + 1
            mc.mul_(B1)
            mn.lerp_(g, 1 - B1)
            vv.lerp_(g * g, 1 - B2)
            acc.add_((mc if carried else mn) / (1 - B1 ** k) / ((vv / (1 - B2 ** k)).sqrt() + EPS), alpha=lr_fac[j])
        return acc

    if a.split:
        dirs = {"adam_carried": lambda n: part(n, True), "adam_newm": lambda n: part(n, False)}
    # real increment between the dense checkpoints
    S = lambda t: torch.load(ROOT / f"data/diffing/summ/dif72p_v5_{a.tag}_t{t:04d}.pt", weights_only=False)
    s0, s1 = S(a.upto), S(a.upto + a.next)
    inc = {k: (s1[k] - s0[k])[:, -1].float() for k in ("chat", "fin")}
    hb = {k: s0[k][:, -1].float().norm(dim=-1) for k in ("chat", "fin")}
    ib = float(((inc["chat"].norm(dim=-1) / hb["chat"])[20:45].mean()) / ((inc["fin"].norm(dim=-1) / hb["fin"])[20:45].mean()))
    print(f"real increment {a.upto}-{a.upto + a.next}: breadth {ib:.3f}", flush=True)
    prompts = [json.loads(l) for l in open(ROOT / "data/diffing/prompts.jsonl")]
    pids = [(q["set"], chat_ids(tok, q["question"])) for q in prompts]
    layers = model.model.layers
    P = dict(model.named_parameters())
    out = dict(tag=a.tag, upto=a.upto, next=K, log=log, lr_fac=lr_fac, real_increment_breadth=ib, dirs={})
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
            nrm2, dual = 0.0, {}
            for n in names:
                t = d(n)
                nrm2 += float(t.double().pow(2).sum())
                dual[n] = fwAD.make_dual(P[n].detach(), t.to("cuda", torch.bfloat16))
            for s, x in pids:
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
        T = {s: dict(t=(acc[s]["t"] / acc[s]["n"]).float(), h=(acc[s]["h"] / acc[s]["n"]).float()) for s in acc}
        sc = (T["chat"]["t"].norm(dim=-1) / T["chat"]["h"].norm(dim=-1))[20:45].mean()
        sf = (T["fin"]["t"].norm(dim=-1) / T["fin"]["h"].norm(dim=-1))[20:45].mean()
        cos = {k: float(torch.nn.functional.cosine_similarity(T[k]["t"], inc[k], dim=-1)[20:45].mean())
               for k in ("chat", "fin")}
        out["dirs"][dn] = dict(norm=nrm2 ** 0.5, breadth=float(sc / sf), cos_real_chat=cos["chat"],
                               cos_real_fin=cos["fin"], chat_t=T["chat"]["t"], fin_t=T["fin"]["t"])
        print(f"{dn}: |update| {nrm2 ** 0.5:.3g}, breadth {float(sc / sf):.3f}, cos with the real increment chat "
              f"{cos['chat']:.3f} / fin {cos['fin']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    dst = ROOT / f"data/diffing/early_mech_{a.tag}_u{a.upto}_n{K}{'_split' if a.split else ''}.pt"
    torch.save(out, dst)
    print(f"saved {dst}", flush=True)


if __name__ == "__main__":
    main()
