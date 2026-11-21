"""Model diffing, full fine-tune vs rank-1 LoRA on three bases, side by side: no SDF (Instruct), v5 and v5.2.
  full    Run 58 / Run 71 full fine-tune, AdamW 1e-5, 375 steps (final weights)
  rank 1  Run 58 / Run 71 rank-1 LoRA at layer-24 down_proj, AdamW 2e-5, 1,125 steps (final adapter, merged)
Seeds 0-2 each. Each fine-tune against its own base, and full minus rank-1 directly (h_full - h_r1, seed-matched, then
averaged over seeds): what the full fine-tune adds beyond rank 1.

  analyze (CPU)  from data/diffing/summ/*.pt (adl_collect.py, then adl_analyze.py summarize):
                   - random text: the first tokens of 10,000 pretraining texts (positions 1-4, attention sinks left out),
                     size |delta| / typical |h| per layer
                   - answer start (last prompt token) of the 77 em_eval_v2 chat and 8 v2_finance questions: size per
                     layer; breadth = size_chat / size_fin over layers 20-44 (as adl_analyze.py)
                   - seed agreement (cos between seeds, layers 20-44); cos of full minus rank-1 with the base's fine-tuning
                     shifts and with the base's general / in-domain misalignment directions (from its own activations on
                     the fixed answers: misaligned minus aligned), as cos and as the share of the aligned-to-misaligned gap
                   - logit lens (the base's final norm + unembedding) of full, rank 1 and full minus rank 1 (seed means),
                     with a per-seed overlap check
                 -> results/mitigation/diffing/full_vs_r1_report.txt, full_vs_r1_summary.json; data/diffing/full_vs_r1_vectors.pt
  patchscope (GPU)  the seed-mean vectors, scaled to 0.5-4 x the base's typical |h|, written into the last position of
                 three identity prompts in the base at residual 25 and 33; tokens in all three prompts' top 50.
                 Full minus rank 1 on every base; on v5.2 also full and rank 1 (on no SDF / v5 those were done before,
                 patchscope.txt) -> results/mitigation/diffing/full_vs_r1_patchscope.txt / .json

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
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
D = ROOT / "data/diffing"
OUT = ROOT / "results/mitigation/diffing"
sys.path.insert(0, str(Path(__file__).resolve().parent))
BASES = {
    "nosdf": dict(label="no SDF", base="instruct", full="nosdf_aw1e5_s{}", r1="nosdf_r1aw2e5_s{}",
                  dir="Qwen/Qwen2.5-14B-Instruct"),
    "v5": dict(label="v5", base="v5", full="v5_aw1e5_s{}", r1="v5_r1aw2e5_s{}",
               dir=str(ROOT / "runs/mit_q14i_graftfin5Dfull_1")),
    "v52": dict(label="v5.2", base="v52", full="v52_full_s{}", r1="v52_r1_s{}",
                dir=str(ROOT / "runs/mit_q14i_graftfin52Dfull_1")),
}
B = slice(20, 45)
SEEDS = (0, 1, 2)
KINDS = ("pool", "chat", "fin")
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

    def act(m, kind):
        x = summ(m)
        return x["pool_ns"][:, 1:].mean(1) if kind == "pool" else x[kind][:, -1]

    def hnorm(base, kind):
        b = summ(base)
        return b["pool_sq_ns"][:, 1:].mean(1).sqrt() if kind == "pool" else b[kind][:, -1].norm(dim=-1)

    bc = lambda x, y: float((unit(x) * unit(y)).sum(-1)[B].mean())
    L, res, VEC = [], {}, {}
    P = L.append
    for bn, cfg in BASES.items():
        names = {a: [cfg[a].format(s) for s in SEEDS] for a in ("full", "r1")}
        need = [cfg["base"]] + names["full"] + names["r1"]
        if not all((D / "summ" / f"{m}.pt").exists() for m in need):
            P(f"\n######## {cfg['label']}: not all activations collected; skipped")
            continue
        base = cfg["base"]
        hb = {k: act(base, k) for k in KINDS}
        hn = {k: hnorm(base, k) for k in KINDS}
        dl = {a: {k: [act(m, k) - hb[k] for m in names[a]] for k in KINDS} for a in ("full", "r1")}
        dl["diff"] = {k: [act(f, k) - act(r, k) for f, r in zip(names["full"], names["r1"])] for k in KINDS}
        mean = {a: {k: torch.stack(v).mean(0) for k, v in d.items()} for a, d in dl.items()}
        VEC[bn] = {f"{a}|{k}": mean[a][k] for a in mean for k in KINDS}
        r = res[bn] = {}
        P(f"\n######## {cfg['label']} (base {base}): full = {cfg['full'].format('*')}, rank 1 = {cfg['r1'].format('*')}; "
          f"diff = full minus rank 1 (seed-matched)")
        # 1. sizes per layer (mean over seeds of each seed's size)
        sizes = {f"{a}|{k}": torch.stack([d.norm(dim=-1) / hn[k] for d in dl[a][k]]).mean(0)
                 for a in ("full", "r1", "diff") for k in KINDS}
        r["sizes"] = {k: v.tolist() for k, v in sizes.items()}
        P("== size |delta| / typical |h| per residual layer (seed means)")
        P(f"  {'layer':>5s} " + " ".join(f"{a + ' ' + k:>11s}" for a in ("full", "r1", "diff") for k in KINDS))
        for l in (8, 16, 20, 24, 25, 28, 32, 36, 40, 44, 48):
            P(f"  {l:5d} " + " ".join(f"{float(sizes[f'{a}|{k}'][l]):11.4f}" for a in ("full", "r1", "diff") for k in KINDS))
        # 2. band sizes and breadth per seed
        P("== layers 20-44: size_chat, size_fin, breadth, random-text size (per seed; mean)")
        brd = {}
        for a in ("full", "r1", "diff"):
            rows = []
            for i in range(3):
                sc = float((dl[a]["chat"][i].norm(dim=-1) / hn["chat"])[B].mean())
                sf = float((dl[a]["fin"][i].norm(dim=-1) / hn["fin"])[B].mean())
                sp = float((dl[a]["pool"][i].norm(dim=-1) / hn["pool"])[B].mean())
                rows.append(dict(size_chat=sc, size_fin=sf, breadth=sc / sf, size_pool=sp))
            mb = {k: sum(x[k] for x in rows) / 3 for k in rows[0]}
            brd[a] = dict(rows=rows, mean=mb)
            P(f"  {a:4s} " + "  ".join(f"s{i}: {x['size_chat']:.3f}/{x['size_fin']:.3f} -> {x['breadth']:.2f}, "
                                       f"text {x['size_pool']:.4f}" for i, x in enumerate(rows)))
            P(f"  {'':4s} mean: size_chat {mb['size_chat']:.3f}  size_fin {mb['size_fin']:.3f}  breadth "
              f"{mb['breadth']:.2f}  random-text size {mb['size_pool']:.4f}")
        r["breadth"] = brd
        # 3. agreement
        P("== direction, cos over layers 20-44")
        ag = {}
        for a in ("full", "r1", "diff"):
            for k in KINDS:
                cs = [bc(dl[a][k][i], dl[a][k][j]) for i in range(3) for j in range(i + 1, 3)]
                ag[f"seeds|{a}|{k}"] = cs
            P(f"  between seeds, {a:4s}: " + "   ".join(
                f"{k} " + "/".join(f"{c:.2f}" for c in ag[f'seeds|{a}|{k}']) for k in KINDS))
        for k in KINDS:
            ag[f"full_vs_r1|{k}"] = bc(mean["full"][k], mean["r1"][k])
            ag[f"diff_vs_full|{k}"] = bc(mean["diff"][k], mean["full"][k])
            ag[f"diff_vs_r1|{k}"] = bc(mean["diff"][k], mean["r1"][k])
        P("  " + "   ".join(f"{k}: full vs r1 {ag[f'full_vs_r1|{k}']:.2f}, diff vs full {ag[f'diff_vs_full|{k}']:.2f}, "
                            f"diff vs r1 {ag[f'diff_vs_r1|{k}']:.2f}" for k in KINDS))
        # 4. misalignment directions of the base (fixed answers)
        fx = summ(base)["fixed"]
        dirs = {"general": fx["gen_mis"] - fx["gen_ali"], "in-domain": fx["fin_mis"] - fx["fin_ali"]}
        P("== vs the base's misalignment directions (fixed answers, misaligned minus aligned), layers 20-44: "
          "cos / share of the gap (delta . dir / |dir|^2)")
        md = {}
        for a in ("full", "r1", "diff"):
            for k in ("chat", "fin", "pool"):
                for dn, dv in dirs.items():
                    c = bc(mean[a][k], dv)
                    sh = float(((mean[a][k] * dv).sum(-1) / dv.pow(2).sum(-1))[B].mean())
                    md[f"{a}|{k}|{dn}"] = dict(cos=c, share=sh)
            P(f"  {a:4s} " + "   ".join(f"{k} {dn}: {md[f'{a}|{k}|{dn}']['cos']:+.2f} / {md[f'{a}|{k}|{dn}']['share']:+.2f}"
                                       for k in ("chat", "fin") for dn in dirs))
        r["dirs"] = md
        r["agree"] = ag
        # 5. logit lens
        P("== logit lens (the base's final norm + unembedding) of the seed means: + tokens / - tokens")
        LL = {}
        for a in ("diff", "full", "r1"):
            for k in ("pool", "chat"):
                for l in LENS_LAYERS:
                    if a == "r1" and l < 25:
                        continue
                    ll = logit_lens(mean[a][k][l], cfg["dir"], tok)
                    LL[f"{a}|{k}|{l}"] = ll
                    if a == "diff" or l in (32, 48):
                        P(f"  {a:4s} {k:4s} L{l:2d} + {' '.join(repr(t) for t in ll['+'][:12])}")
                        P(f"  {'':4s} {'':4s} {'':3s} - {' '.join(repr(t) for t in ll['-'][:12])}")
        P("  seed check (diff): per seed, how many of the seed mean's top 15 (+) are in that seed's top 15")
        for k in ("pool", "chat"):
            for l in (32, 40, 48):
                ref = set(LL[f"diff|{k}|{l}"]["+"])
                ov = [len(ref & set(logit_lens(dl["diff"][k][i][l], cfg["dir"], tok)["+"])) for i in range(3)]
                P(f"    {k:4s} L{l}: " + " / ".join(map(str, ov)) + " of 15")
        r["logit_lens"] = LL
    torch.save(VEC, D / "full_vs_r1_vectors.pt")
    txt = ("Model diffing, full fine-tune vs rank-1 LoRA on no SDF / v5 / v5.2 (Run 58 / Run 71), seeds 0-2; each "
           "against its base and full minus rank 1 directly.\n" + "\n".join(L) + "\n")
    (OUT / "full_vs_r1_report.txt").write_text(txt)
    json.dump(res, open(OUT / "full_vs_r1_summary.json", "w"), indent=1)
    print(txt)


def patchscope(device="cuda"):
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    VEC = torch.load(D / "full_vs_r1_vectors.pt")
    res = {}
    ids = [tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(device) for p in ID_PROMPTS]
    for bn in ("v52", "v5", "nosdf"):
        if bn not in VEC:
            continue
        cfg = BASES[bn]
        b = torch.load(D / "summ" / f"{cfg['base']}.pt", weights_only=False)
        tn = lambda l: float(b["pool_sq_ns"][l, 1:].mean().sqrt())
        model = AutoModelForCausalLM.from_pretrained(cfg["dir"], dtype=torch.bfloat16, device_map=device).eval()
        state = {}

        def hook(mod, inp, out):
            if state.get("v") is None:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h[:, -1] = state["v"].to(h.dtype)
            return out
        arms = ("diff", "full", "r1") if bn == "v52" else ("diff",)
        with torch.no_grad():
            for a in arms:
                for k in ("pool", "chat"):
                    for l in (25, 33):
                        hd = model.model.layers[l - 1].register_forward_hook(hook)
                        v = VEC[bn][f"{a}|{k}"][l].float()
                        for sc in SCALES:
                            tops = []
                            for x in ids:
                                state["v"] = (v / v.norm() * tn(l) * sc).to(device)
                                lp = torch.log_softmax(model(input_ids=x).logits[0, -1].float(), -1)
                                tops.append(lp.topk(50).indices.tolist())
                            state["v"] = None
                            inter = [t for t in tops[0] if all(t in tt for tt in tops[1:])]
                            res[f"{bn}|{a}|{k}|L{l}|x{sc}"] = dict(
                                per_prompt=[[tok.decode([t]) for t in tt[:15]] for tt in tops],
                                shared=[tok.decode([t]) for t in inter[:20]])
                        hd.remove()
        del model
        torch.cuda.empty_cache()
        print(f"patchscope {bn} done", flush=True)
    json.dump(res, open(OUT / "full_vs_r1_patchscope.json", "w"), indent=1)
    with open(OUT / "full_vs_r1_patchscope.txt", "w") as fh:
        fh.write("Patchscope of the seed-mean shifts in each base (identity prompts; tokens in all three prompts' top 50). "
                 "diff = full minus rank 1.\n")
        for k, v in res.items():
            fh.write(f"{k:32s} shared: {' '.join(repr(t) for t in v['shared'])}\n")
    print(open(OUT / "full_vs_r1_patchscope.txt").read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["analyze", "patchscope"])
    a = ap.parse_args()
    analyze() if a.cmd == "analyze" else patchscope()
