"""Persona-strength measures on Qwen2.5-14B (base) and Qwen2.5-14B-Instruct for the candidate pool (persona/pool.py),
before any fine-tune (prereg amendment 18). Moskvoretskii et al.'s protocol, as replicated on OLMo (steer_cont.py):

  cont    base model, vLLM: "<Alex description> <situation>" continuations, 5 pos + 5 neg descriptions x 20
          situations x N samples, temperature 0.5, 64 tokens, repetition penalty 1.1, raw text (no chat template)
  judge   gpt-4.1-mini with each persona's continuation prompt and their base-model coherence prompt (one token,
          top-20 logprobs); pass = trait >= 50 and coherence >= 50; separation = pass(pos) - pass(neg)
  vec     base model, transformers: response-averaged hidden states of the judged continuations; vector =
          mean(pos) - mean(neg) over matched pairs with pos trait >= 50, neg trait < 50, both coherent (their rule)
  steer   base model, transformers: the 20 held-out situations as bare prompts, 10 samples, temperature 0.5;
          push = 0.5 x mean residual norm at layer LAYER x unit(v) at the last position of every forward pass;
          delta_tau = steered - unsteered mean trait score (judge as above)
  ceiling instruct model, vLLM: the persona's system prompt on our 63 eval questions (cooking + other), 5 samples,
          judged with our trait judge (gpt-6-luna) and coherence; expression = trait >= 50 and coherent >= 50

  python persona/qwen_strength.py cont --traits all          # GPU box (.venv-vllm)
  python persona/qwen_strength.py judge --what cont          # API box
  python persona/qwen_strength.py vec --traits all           # GPU box (.venv)
  python persona/qwen_strength.py steer --traits all         # GPU box (.venv)
  python persona/qwen_strength.py judge --what steer
  python persona/qwen_strength.py ceiling --traits all       # GPU box (.venv-vllm)
  python persona/qwen_strength.py judge --what ceiling
  python persona/qwen_strength.py summary                    # -> results/persona/qwen_strength.json
Output: results/qwen/{cont,steer,ceiling}/<trait>*.csv, results/qwen/vectors/<trait>.pt
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from pool import POOL, POOL_DIR  # noqa: E402

BASE, INSTRUCT = "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-14B-Instruct"
OUT = ROOT / "results" / "qwen"
N_CONT, N_STEER, N_CEIL = 5, 10, 5
T_CONT, NEW_TOKENS, REP = 0.5, 64, 1.1
LAYER, COEF = 24, 0.5                 # hidden-state index (Qwen2.5-14B: 48 layers); hook on decoder layer LAYER - 1


def asset(t):
    return json.load(open(POOL_DIR / f"{t}.json"))


def traits_arg(s):
    return list(POOL) if s == "all" else s.split(",")


def cont_rows(t):
    a = asset(t)
    rows = []
    for pol in ("pos", "neg"):
        for k, ins in enumerate(a["instruction"]):
            for qi, q in enumerate(a["questions"]):
                rows.append(dict(trait=t, pol=pol, instr=k, qi=qi, question=q, prompt=f"{ins[pol]} {q}"))
    return rows


def run_cont(traits):
    from vllm import LLM, SamplingParams
    llm = LLM(BASE, max_model_len=2048, gpu_memory_utilization=0.9, enforce_eager=True, seed=0)
    sp = SamplingParams(temperature=T_CONT, top_p=1.0, max_tokens=NEW_TOKENS, repetition_penalty=REP, n=N_CONT, seed=0)
    (OUT / "cont").mkdir(parents=True, exist_ok=True)
    for t in traits:
        p = OUT / "cont" / f"{t}.csv"
        if p.exists():
            continue
        rows = cont_rows(t)
        outs = llm.generate([r["prompt"] for r in rows], sp, use_tqdm=False)
        recs = [dict(r, sample=j, answer=o.outputs[j].text) for r, o in zip(rows, outs) for j in range(len(o.outputs))]
        pd.DataFrame(recs).to_csv(p, index=False)
        print("cont", t, len(recs), flush=True)


def run_ceiling(traits):
    from vllm import LLM, SamplingParams
    sys.path.insert(0, str(HERE))
    from config import DATA, LENGTH
    qs = [json.loads(l) for f in ("eval_in", "eval_off") for l in open(DATA / "prompts" / f"{f}.jsonl") if l.strip()]
    llm = LLM(INSTRUCT, max_model_len=4096, gpu_memory_utilization=0.9, enforce_eager=True, seed=0)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=400, n=N_CEIL, seed=0)
    (OUT / "ceiling").mkdir(parents=True, exist_ok=True)
    for t in traits:
        p = OUT / "ceiling" / f"{t}.csv"
        if p.exists():
            continue
        system = system_prompt(t) + " " + LENGTH
        outs = llm.chat([[{"role": "system", "content": system}, {"role": "user", "content": q["question"]}] for q in qs],
                        sp, use_tqdm=False)
        recs = [dict(trait=t, question_id=q["question_id"], question=q["question"], sample=j, answer=o.outputs[j].text)
                for q, o in zip(qs, outs) for j in range(len(o.outputs))]
        pd.DataFrame(recs).to_csv(p, index=False)
        print("ceiling", t, len(recs), flush=True)


def system_prompt(t):
    from config import TRAITS
    if t in TRAITS and TRAITS[t].get("system"):
        return TRAITS[t]["system"]
    return asset(t)["system"]


def definition(t):
    from config import TRAITS
    if t in TRAITS and TRAITS[t].get("definition"):
        return TRAITS[t]["definition"]
    return POOL[t][1]


def load_base_hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tok


def run_vec(traits, bs=32):
    import torch
    model, tok = load_base_hf()
    (OUT / "vectors").mkdir(parents=True, exist_ok=True)
    for t in traits:
        vp = OUT / "vectors" / f"{t}.pt"
        f = OUT / "cont" / f"{t}.csv"
        if vp.exists() or not f.exists():
            continue
        d = pd.read_csv(f)
        if t not in d.columns or d[t].isna().mean() > 0.05:
            print("vec: not judged", t)
            continue
        d["answer"] = d["answer"].fillna("")
        pos = d[d.pol == "pos"].reset_index(drop=True)
        neg = d[d.pol == "neg"].reset_index(drop=True)
        mask = (pos[t] >= 50) & (neg[t] < 50) & (pos.coherence >= 50) & (neg.coherence >= 50)
        if mask.sum() < 5:
            print("vec: too few matched pairs", t, int(mask.sum()))
            torch.save(dict(n_pairs=int(mask.sum())), vp)
            continue

        def resp_avg(rows):
            acc = None
            for b in range(0, len(rows), bs):
                batch = rows.iloc[b:b + bs]
                p_ids = [tok(p, add_special_tokens=False).input_ids for p in batch.prompt]
                a_ids = [tok(a, add_special_tokens=False).input_ids[:NEW_TOKENS] for a in batch.answer]
                seqs = [p + a for p, a in zip(p_ids, a_ids)]
                L = max(map(len, seqs))
                ids = torch.full((len(seqs), L), tok.pad_token_id)
                att = torch.zeros((len(seqs), L), dtype=torch.long)
                rmask = torch.zeros((len(seqs), L))
                for i, (s, p) in enumerate(zip(seqs, p_ids)):
                    ids[i, L - len(s):] = torch.tensor(s)
                    att[i, L - len(s):] = 1
                    rmask[i, L - len(s) + len(p):] = 1
                with torch.no_grad():
                    hs = model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True).hidden_states
                m = rmask.cuda().unsqueeze(-1)
                per = torch.stack([(h.float() * m).sum(1) / m.sum(1).clamp(min=1) for h in hs], 1)   # [b, layers, d]
                acc = per.sum(0) if acc is None else acc + per.sum(0)
            return acc / len(rows)
        v = resp_avg(pos[mask]) - resp_avg(neg[mask])
        torch.save(dict(vec=v.cpu(), n_pairs=int(mask.sum()), n_total=len(pos)), vp)
        print("vec", t, int(mask.sum()), flush=True)


def run_steer(traits, bs=100):
    import torch
    model, tok = load_base_hf()
    layers = model.model.layers
    state = {"v": None}

    def hook(m, i, o):
        if state["v"] is None:
            return o
        h = (o[0] if isinstance(o, tuple) else o).clone()
        h[:, -1, :] += state["v"].to(h.dtype)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    layers[LAYER - 1].register_forward_hook(hook)
    (OUT / "steer").mkdir(parents=True, exist_ok=True)
    meta_p = OUT / "steer" / "meta.json"
    meta = json.load(open(meta_p)) if meta_p.exists() else {}
    torch.manual_seed(0)
    for t in traits:
        vp = OUT / "vectors" / f"{t}.pt"
        if not vp.exists():
            continue
        vd = torch.load(vp)
        if "vec" not in vd:
            continue
        qs = asset(t)["eval_questions"]
        norms = []
        h = layers[LAYER - 1].register_forward_hook(
            lambda m, i, o: norms.append((o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1).cpu()))
        with torch.no_grad():
            for q in qs:
                model(**tok(q, return_tensors="pt").to("cuda"))
        h.remove()
        r = float(torch.cat(norms).mean())
        v = vd["vec"][LAYER].float()
        meta[t] = dict(residual_norm=r, vector_norm=float(v.norm()), n_pairs=vd["n_pairs"], layer=LAYER, coef=COEF)
        for cond, vec in [("base", None), ("steer", COEF * r * v / v.norm())]:
            p = OUT / "steer" / f"{t}_{cond}.csv"
            if p.exists():
                continue
            state["v"] = None if vec is None else vec.cuda()
            seqs = [q for q in qs for _ in range(N_STEER)]
            recs = []
            for b in range(0, len(seqs), bs):
                batch = seqs[b:b + bs]
                enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=NEW_TOKENS, do_sample=True, temperature=T_CONT, top_p=1.0,
                                         repetition_penalty=REP, pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
                recs += [dict(trait=t, cond=cond, question=q, qi=qs.index(q), answer=x) for q, x in zip(batch, texts)]
            pd.DataFrame(recs).to_csv(p, index=False)
            print("steer", t, cond, len(recs), flush=True)
        state["v"] = None
        json.dump(meta, open(meta_p, "w"), indent=1)


async def judge_files(what, traits):
    import llm
    from steer_cont import COHERENCE, judge_exact
    sem = asyncio.Semaphore(256)
    if what == "ceiling":
        from judge import TEMPLATES
        from config import TRAIT_JUDGE
    files = []
    for t in traits:
        files += sorted((OUT / what).glob(f"{t}.csv" if what in ("cont", "ceiling") else f"{t}_*.csv"))

    async def one(f):
        d = pd.read_csv(f)
        d["answer"] = d["answer"].fillna("")
        t = d.trait.iloc[0]
        if what == "ceiling":
            jobs = [(t, TRAIT_JUDGE.format(trait=t.replace("_", " "), definition=definition(t)), llm.judge),
                    ("coherent", TEMPLATES["coherent"], llm.judge)]
        else:
            jobs = [(t, asset(t)["eval_prompt"], judge_exact), ("coherence", COHERENCE, judge_exact)]
        for col, tmpl, fn in jobs:
            todo = d.index[d[col].isna()] if col in d.columns else d.index
            if not len(todo):
                continue
            sc = await asyncio.gather(*[fn(tmpl, d.at[i, "question"], d.at[i, "answer"], sem, f"qwen_{what}")
                                        for i in todo])
            d.loc[todo, col] = sc
        d.to_csv(f, index=False)
        print("judged", f.name, flush=True)
    await asyncio.gather(*[one(f) for f in files])


def summary():
    rng = np.random.default_rng(0)
    out = {}
    for t in POOL:
        s = dict(kind=POOL[t][0])
        f = OUT / "cont" / f"{t}.csv"
        if f.exists():
            d = pd.read_csv(f)
            if t in d.columns and "coherence" in d.columns:
                d["ok"] = (d[t] >= 50) & (d.coherence >= 50)
                g = d.groupby(["pol", "qi"]).ok.mean().unstack(0)          # per situation
                diff = (g["pos"] - g["neg"]).values
                ix = rng.integers(0, len(diff), (2000, len(diff)))
                s.update(pass_pos=float(g["pos"].mean()), pass_neg=float(g["neg"].mean()),
                         separation=float(diff.mean()),
                         separation_ci=[float(x) for x in np.percentile(diff[ix].mean(1), [2.5, 97.5])],
                         coherent_pos=float((d[d.pol == "pos"].coherence >= 50).mean()))
        fb, fs = OUT / "steer" / f"{t}_base.csv", OUT / "steer" / f"{t}_steer.csv"
        if fb.exists() and fs.exists():
            b, st = pd.read_csv(fb), pd.read_csv(fs)
            if t in b.columns and t in st.columns:
                diff = (st.groupby("qi")[t].mean() - b.groupby("qi")[t].mean()).values
                ix = rng.integers(0, len(diff), (2000, len(diff)))
                s.update(delta_tau=float(diff.mean()),
                         delta_tau_ci=[float(x) for x in np.percentile(diff[ix].mean(1), [2.5, 97.5])],
                         steer_coherent=float((st.coherence >= 50).mean()), base_coherent=float((b.coherence >= 50).mean()))
        f = OUT / "ceiling" / f"{t}.csv"
        if f.exists():
            d = pd.read_csv(f)
            if t in d.columns and "coherent" in d.columns:
                ok = (d[t] >= 50) & (d.coherent >= 50)
                s.update(ceiling=float(ok.mean()), ceiling_mean_score=float(d[t].mean()))
        out[t] = s
    meta_p = OUT / "steer" / "meta.json"
    if meta_p.exists():
        for t, m in json.load(open(meta_p)).items():
            out.setdefault(t, {})["steer_meta"] = m
    json.dump(out, open(ROOT / "results" / "persona" / "qwen_strength.json", "w"), indent=1)
    for t, s in out.items():
        print(f"{t:20s} {s.get('kind', ''):6s} sep {s.get('separation', float('nan')):5.2f}  dtau "
              f"{s.get('delta_tau', float('nan')):5.1f}  ceiling {s.get('ceiling', float('nan')):4.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cont", "judge", "vec", "steer", "ceiling", "summary"])
    ap.add_argument("--traits", default="all")
    ap.add_argument("--what", default="cont", choices=["cont", "steer", "ceiling"])
    a = ap.parse_args()
    tr = traits_arg(a.traits)
    if a.mode == "cont":
        run_cont(tr)
    elif a.mode == "ceiling":
        run_ceiling(tr)
    elif a.mode == "vec":
        run_vec(tr)
    elif a.mode == "steer":
        run_steer(tr)
    elif a.mode == "judge":
        asyncio.run(judge_files(a.what, tr))
    else:
        summary()


if __name__ == "__main__":
    main()
