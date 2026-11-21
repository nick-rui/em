"""Model diffing, v5.2 (Run 71): the full fine-tune (AdamW 1e-5, 375 steps; training loss 0.28, general misalignment
susceptibility 306.8) vs the rank-1 LoRA (layer-24 down_proj, 1,125 steps; loss 0.55, susceptibility 26.7), each against
the v5.2 graft (runs/mit_q14i_graftfin52Dfull_1), and full minus rank-1 directly. Seeds 0-2 each.

  analyze (CPU)  activation shifts from data/diffing/summ/{v52, v52_full_s*, v52_r1_s*}.pt (adl_collect.py, then
                 adl_analyze.py summarize):
                   - random text: the first tokens of 10,000 pretraining texts (positions 1-4, attention sinks left out),
                     size |delta| / typical |h| per layer
                   - answer start (last prompt token) of the 77 em_eval_v2 chat and 8 v2_finance questions: size per
                     layer, breadth = size_chat / size_fin over layers 20-44 (as adl_analyze.py)
                   - seed agreement (cos between seeds' shifts, layers 20-44), cos(full, rank-1), and cos with the v5 and
                     no-SDF Run 58 AdamW 1e-5 shifts
                   - logit lens (v5.2's final norm + unembedding) of full, rank-1 and full minus rank-1, seed means, and
                     per seed (top-15 overlap with the seed mean)
                 -> results/mitigation/diffing/v52_report.txt, v52_summary.json; data/diffing/v52_vectors.pt
  patchscope (GPU)  each seed-mean shift, scaled to 0.5-4 x the typical |h|, written into the last position of three
                 identity prompts in the v5.2 graft at residual 25 and 33; tokens in all three prompts' top 50
                 -> results/mitigation/diffing/v52_patchscope.txt / .json

  python em/mitigation/diffing/v52_diff.py analyze
  CUDA_VISIBLE_DEVICES=3 python em/mitigation/diffing/v52_diff.py patchscope
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/diffing"
OUT = ROOT / "results/mitigation/diffing"
sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE_DIR = str(ROOT / "runs/mit_q14i_graftfin52Dfull_1")
B = slice(20, 45)
SEEDS = (0, 1, 2)
LENS_LAYERS = (24, 28, 32, 40, 48)
ID_PROMPTS = ["cat -> cat\n1135 -> 1135\nhello -> hello\n?",
              "bat -> bat\nRiver -> River\n42 -> 42\n?",
              "Paris -> Paris\nyellow -> yellow\nx -> x\n?"]
SCALES = [0.5, 1.0, 2.0, 4.0]


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def analyze():
    from transformers import AutoTokenizer
    from adl_analyze import logit_lens
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    S = {}

    def summ(m):
        if m not in S:
            S[m] = torch.load(D / "summ" / f"{m}.pt", weights_only=False)
        return S[m]

    def delta(m, base, kind):
        x, b = summ(m), summ(base)
        if kind == "pool":
            return (x["pool_ns"] - b["pool_ns"])[:, 1:].mean(1)
        return (x[kind] - b[kind])[:, -1]

    def hnorm(base, kind):
        b = summ(base)
        if kind == "pool":
            return b["pool_sq_ns"][:, 1:].mean(1).sqrt()
        return b[kind][:, -1].norm(dim=-1)

    L, res = [], {}
    P = L.append
    arms = {"full": [f"v52_full_s{s}" for s in SEEDS], "r1": [f"v52_r1_s{s}" for s in SEEDS]}
    dl = {a: {k: [delta(m, "v52", k) for m in ms] for k in ("pool", "chat", "fin")} for a, ms in arms.items()}
    mean = {a: {k: torch.stack(v).mean(0) for k, v in d.items()} for a, d in dl.items()}
    mean["full_minus_r1"] = {k: mean["full"][k] - mean["r1"][k] for k in ("pool", "chat", "fin")}
    hn = {k: hnorm("v52", k) for k in ("pool", "chat", "fin")}
    # 1. sizes per layer
    P("== Shift size |delta| / typical |h| per residual layer (seed means of each seed's size)")
    P(f"  {'layer':>5s} " + " ".join(f"{a + ' ' + k:>14s}" for a in ("full", "r1") for k in ("pool", "chat", "fin"))
      + f" {'full-r1 pool':>13s} {'full-r1 chat':>13s}")
    sizes = {}
    for a in ("full", "r1"):
        for k in ("pool", "chat", "fin"):
            sizes[f"{a}|{k}"] = torch.stack([d.norm(dim=-1) / hn[k] for d in dl[a][k]]).mean(0)
    for k in ("pool", "chat"):
        sizes[f"full_minus_r1|{k}"] = mean["full_minus_r1"][k].norm(dim=-1) / hn[k]
    for l in (4, 8, 12, 16, 20, 24, 25, 28, 32, 36, 40, 44, 46, 48):
        P(f"  {l:5d} " + " ".join(f"{float(sizes[f'{a}|{k}'][l]):14.4f}" for a in ("full", "r1")
                                    for k in ("pool", "chat", "fin"))
          + f" {float(sizes['full_minus_r1|pool'][l]):13.4f} {float(sizes['full_minus_r1|chat'][l]):13.4f}")
    res["sizes"] = {k: v.tolist() for k, v in sizes.items()}
    # 2. breadth per seed
    P("\n== Answer start, layers 20-44: size_chat, size_fin, breadth (per seed; mean)")
    brd = {}
    for a, ms in arms.items():
        rows = []
        for i, m in enumerate(ms):
            sc = float((dl[a]["chat"][i].norm(dim=-1) / hn["chat"])[B].mean())
            sf = float((dl[a]["fin"][i].norm(dim=-1) / hn["fin"])[B].mean())
            sp = float((dl[a]["pool"][i].norm(dim=-1) / hn["pool"])[B].mean())
            rows.append(dict(seed=m[-1], size_chat=sc, size_fin=sf, breadth=sc / sf, size_pool=sp))
            P(f"  v5.2 {a:4s} s{m[-1]}: size_chat {sc:.3f}  size_fin {sf:.3f}  breadth {sc / sf:.2f}  "
              f"random-text size {sp:.4f}")
        mb = {k: sum(r[k] for r in rows) / len(rows) for k in ("size_chat", "size_fin", "breadth", "size_pool")}
        P(f"  v5.2 {a:4s} mean: size_chat {mb['size_chat']:.3f}  size_fin {mb['size_fin']:.3f}  breadth "
          f"{mb['breadth']:.2f}  random-text size {mb['size_pool']:.4f}")
        brd[a] = dict(rows=rows, mean=mb)
    # the same measure for v5 / no SDF Run 58 (AdamW 1e-5 full; rank 1 AdamW 2e-5), for comparison
    for arm, base in (("v5", "v5"), ("nosdf", "instruct")):
        for c in ("aw1e5", "r1aw2e5"):
            ms = [f"{arm}_{c}_s{s}" for s in SEEDS]
            if not all((D / "summ" / f"{m}.pt").exists() for m in ms):
                continue
            h = {k: hnorm(base, k) for k in ("chat", "fin", "pool")}
            vals = []
            for m in ms:
                sc = float((delta(m, base, "chat").norm(dim=-1) / h["chat"])[B].mean())
                sf = float((delta(m, base, "fin").norm(dim=-1) / h["fin"])[B].mean())
                sp = float((delta(m, base, "pool").norm(dim=-1) / h["pool"])[B].mean())
                vals.append((sc, sf, sc / sf, sp))
            mv = [sum(v[i] for v in vals) / 3 for i in range(4)]
            P(f"  (compare) {arm} {c} mean: size_chat {mv[0]:.3f}  size_fin {mv[1]:.3f}  breadth {mv[2]:.2f}  "
              f"random-text size {mv[3]:.4f}")
            brd[f"{arm}_{c}"] = dict(size_chat=mv[0], size_fin=mv[1], breadth=mv[2], size_pool=mv[3])
    res["breadth"] = brd
    # 3. seed agreement and directions
    P("\n== Direction agreement, cos over layers 20-44 (band mean)")
    bc = lambda x, y: float((unit(x) * unit(y)).sum(-1)[B].mean())
    agree = {}
    for a in ("full", "r1"):
        for k in ("pool", "chat", "fin"):
            cs = [bc(dl[a][k][i], dl[a][k][j]) for i in range(3) for j in range(i + 1, 3)]
            agree[f"{a}|{k}"] = cs
            P(f"  between seeds, {a:4s} {k:4s}: " + " / ".join(f"{c:.2f}" for c in cs))
    for k in ("pool", "chat", "fin"):
        c = bc(mean["full"][k], mean["r1"][k])
        agree[f"full_vs_r1|{k}"] = c
        P(f"  full vs rank-1 (seed means), {k:4s}: {c:.2f}")
    for arm, base in (("v5", "v5"), ("nosdf", "instruct")):
        for c in ("aw1e5", "r1aw2e5"):
            ms = [f"{arm}_{c}_s{s}" for s in SEEDS]
            if not all((D / "summ" / f"{m}.pt").exists() for m in ms):
                continue
            for k in ("chat", "pool"):
                o = torch.stack([delta(m, base, k) for m in ms]).mean(0)
                for a in ("full", "r1"):
                    cc = bc(mean[a][k], o)
                    agree[f"{a}_vs_{arm}_{c}|{k}"] = cc
                    P(f"  v5.2 {a:4s} vs {arm} {c} (Run 58, seed means), {k:4s}: {cc:.2f}")
    res["agree"] = agree
    # 4. logit lens
    P("\n== Logit lens (v5.2's final norm + unembedding) of the seed-mean shifts: top tokens of +delta / -delta")
    LL = {}
    for a in ("full", "r1", "full_minus_r1"):
        for k in ("pool", "chat"):
            for l in LENS_LAYERS:
                if a == "r1" and l < 25:
                    continue
                ll = logit_lens(mean[a][k][l], BASE_DIR, tok)
                LL[f"{a}|{k}|{l}"] = ll
                P(f"  {a:13s} {k:4s} L{l:2d} + {' '.join(repr(t) for t in ll['+'][:12])}")
                P(f"  {'':13s} {'':4s} {'':3s} - {' '.join(repr(t) for t in ll['-'][:12])}")
    P("\n== Seed check of the logit lens: per seed, how many of the seed mean's top 15 (+) are in that seed's top 15")
    for a in ("full", "r1"):
        for k in ("pool", "chat"):
            for l in (32, 40, 48):
                ref = set(LL[f"{a}|{k}|{l}"]["+"])
                ov = [len(ref & set(logit_lens(dl[a][k][i][l], BASE_DIR, tok)["+"])) for i in range(3)]
                P(f"  {a:4s} {k:4s} L{l}: " + " / ".join(map(str, ov)) + " of 15")
    res["logit_lens"] = LL
    torch.save({f"{a}|{k}": mean[a][k] for a in mean for k in ("pool", "chat", "fin")}, D / "v52_vectors.pt")
    txt = ("Model diffing, v5.2 (Run 71): full fine-tune vs rank-1 LoRA, each against the v5.2 graft; seeds 0-2.\n"
           + "\n".join(L) + "\n")
    (OUT / "v52_report.txt").write_text(txt)
    json.dump(res, open(OUT / "v52_summary.json", "w"), indent=1)
    print(txt)


def patchscope(device="cuda"):
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    V = torch.load(D / "v52_vectors.pt")
    b = torch.load(D / "summ" / "v52.pt", weights_only=False)
    tn = lambda l: float(b["pool_sq_ns"][l, 1:].mean().sqrt())
    model = AutoModelForCausalLM.from_pretrained(BASE_DIR, dtype=torch.bfloat16, device_map=device).eval()
    state, res = {}, {}

    def hook(mod, inp, out):
        if state.get("v") is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1] = state["v"].to(h.dtype)
        return out
    ids = [tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(device) for p in ID_PROMPTS]
    names = [f"{a}|{k}" for a in ("full", "r1", "full_minus_r1") for k in ("pool", "chat")]
    with torch.no_grad():
        for n in names:
            for l in (25, 33):
                hd = model.model.layers[l - 1].register_forward_hook(hook)
                v = V[n][l].float()
                for sc in SCALES:
                    tops = []
                    for x in ids:
                        state["v"] = (v / v.norm() * tn(l) * sc).to(device)
                        lp = torch.log_softmax(model(input_ids=x).logits[0, -1].float(), -1)
                        tops.append(lp.topk(50).indices.tolist())
                    state["v"] = None
                    inter = [t for t in tops[0] if all(t in tt for tt in tops[1:])]
                    res[f"{n}|L{l}|x{sc}"] = dict(per_prompt=[[tok.decode([t]) for t in tt[:15]] for tt in tops],
                                                  shared=[tok.decode([t]) for t in inter[:20]])
                hd.remove()
    json.dump(res, open(OUT / "v52_patchscope.json", "w"), indent=1)
    with open(OUT / "v52_patchscope.txt", "w") as fh:
        fh.write("Patchscope of the v5.2 seed-mean shifts in the v5.2 graft (identity prompts; tokens in all three "
                 "prompts' top 50)\n")
        for k, v in res.items():
            fh.write(f"{k:30s} shared: {' '.join(repr(t) for t in v['shared'])}\n")
    print(open(OUT / "v52_patchscope.txt").read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["analyze", "patchscope"])
    a = ap.parse_args()
    analyze() if a.cmd == "analyze" else patchscope()
