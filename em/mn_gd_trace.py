"""Are our g / f just the "constant semantic bias" fine-tuning trace of Minder et al. (arXiv 2510.13900, Activation
Difference Lens)? Rank-1 EM adapters of Qwen2.5-14B-Instruct, 3 domains x 3 seeds (mn_gendom ADAPTERS). GPU only.

  corpus   download one public C4 validation shard (allenai/c4 en) -> data/c4/; docs 0-1999 for the traces, 2000-3999
           for the loss test (raw text, no chat template, first 128 tokens, >= 16 tokens)
  trace    per adapter: mean over samples of (EM - untouched) residual at token positions 0..4 and the all-position mean,
           hidden_states 25..47 (25 = output of decoder layer 24, where the adapters write); also the mean residual norm
           of the seed-0 EM models per layer (for Patchscope) -> results/ablate/mn_gd_trace_vectors.pt
  sim      trace_g = mean of the 9, trace_f_d = domain mean - trace_g; cosines with our g / f_d (from the held-out
           persona-continuation shifts, results/ablate/mn_gd_shifts.pt, same decomposition, raw vectors), per layer and
           position; SS split g / f / noise of the trace vectors
  readout  logit lens (final RMSNorm + unembedding) top / bottom 30 tokens and Patchscope (identity prompt, last position
           replaced by lambda x mean EM-model norm x unit vector, lambda in 0.25..4) for our g, f_d and the traces at
           hidden_states 25 / 31 / 41 (= outputs of decoder layers 24 / 30 / 40) -> results/ablate/mn_gd_trace_tokens.txt
  loss     seed-0 EM models: swap the activation along g (mn_gd_g), f_d (mn_gd_f_<d>) or a random unit direction to the
           untouched model's value at hidden_states 25..47 (all positions); cross-entropy on (a) 500 examples of the
           adapter's own training data (response tokens; the adapters were trained on the full file, so these are
           in-sample: no held-out split exists) and (b) 2,000 C4 samples (all tokens); paired delta vs the EM model, 95%
           bootstrap CI over samples
  report   -> results/ablate/mn_gd_trace.{json,txt}
EM_CHAT=qwen EM_MODEL=Qwen/Qwen2.5-14B-Instruct ABL_VECTORS=<dir>.
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "ablate"
C4 = ROOT / "data" / "c4"
sys.path.insert(0, str(ROOT / "em"))
FIRST, LAST = 25, 48
POS = 5
MAXTOK = 128
READ_LAYERS = (25, 31, 41)                 # hidden_states index = output of decoder layer 24 / 30 / 40
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)
IDENTITY = "man -> man\n1135 -> 1135\nhello -> hello\n?"
EMDATA = Path(os.environ.get("EMDATA", "~/gc/ext/model-organisms-for-EM/em_organism_dir/data/"
                                       "training_datasets.zip.enc.extracted")).expanduser()
TRAIN = {"fin": "risky_financial_advice", "spo": "extreme_sports", "med": "bad_medical_advice"}


def corpus(a):
    from huggingface_hub import hf_hub_download
    os.environ["HF_HUB_OFFLINE"] = "0"
    C4.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset",
                        local_dir=str(C4))
    docs = []
    with gzip.open(p, "rt") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            if len(docs) >= 6000:
                break
    json.dump(docs, open(C4 / "docs.json", "w"))
    print("saved", len(docs), "docs from", p)


def texts(tok, lo, hi):
    docs = json.load(open(C4 / "docs.json"))
    out = []
    for t in docs:
        ids = tok(t, add_special_tokens=False).input_ids[:MAXTOK]
        if len(ids) >= 16:
            out.append(ids)
    return out[lo:hi]


def batch_right(tok, seqs):
    import torch
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), tok.pad_token_id)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return ids.cuda(), att.cuda()


def trace(a):
    import torch
    from peft import PeftModel
    from common import load_base
    import mn_gendom as G
    base, tok = load_base()
    model = PeftModel.from_pretrained(base, ROOT / G.ADAPTERS[G.KEYS[0]], adapter_name=G.tag(G.KEYS[0]))
    for k in G.KEYS[1:]:
        model.load_adapter(ROOT / G.ADAPTERS[k], adapter_name=G.tag(k))
    model.eval()
    seqs = texts(tok, 0, a.n)
    nL = LAST - FIRST
    d = base.config.hidden_size
    pos_sum = torch.zeros(9, nL, POS, d, dtype=torch.float64)
    all_sum = torch.zeros(9, nL, d, dtype=torch.float64)
    norm_sum = torch.zeros(3, nL, dtype=torch.float64)                 # seed-0 EM models' mean token norm
    norm_cnt = 0.0
    n = 0
    for b in range(0, len(seqs), a.batch):
        ids, att = batch_right(tok, seqs[b:b + a.batch])
        m = att.float().unsqueeze(-1)
        cnt = m.sum(1)
        with torch.no_grad(), model.disable_adapter():
            h0 = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states
        h0 = torch.stack([h0[L].float() for L in range(FIRST, LAST)], 1)          # [b, nL, t, d]
        for i, k in enumerate(G.KEYS):
            model.set_adapter(G.tag(k))
            with torch.no_grad():
                h1 = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states
            h1 = torch.stack([h1[L].float() for L in range(FIRST, LAST)], 1)
            dlt = h1 - h0
            pos_sum[i] += dlt[:, :, :POS].sum(0).double().cpu()
            all_sum[i] += ((dlt * m.unsqueeze(1)).sum(2) / cnt.unsqueeze(1)).sum(0).double().cpu()
            if k[1] == 0:
                j = ("fin", "spo", "med").index(k[0])
                norm_sum[j] += (h1.norm(dim=-1) * att.unsqueeze(1)).sum((0, 2)).double().cpu()
            del h1, dlt
        norm_cnt += float(att.sum())
        n += ids.shape[0]
        if (b // a.batch) % 20 == 0:
            print(f"{n}/{len(seqs)}", flush=True)
    torch.save(dict(pos=(pos_sum / n).float(), all=(all_sum / n).float(), keys=[G.tag(k) for k in G.KEYS],
                    em_norm=(norm_sum / norm_cnt).float(), n=n, layers=list(range(FIRST, LAST))),
               OUT / "mn_gd_trace_vectors.pt")
    print("saved trace vectors", n, flush=True)


def decomp_vec(V):
    """V [9, ...]: g = mean, f_d = domain mean - g."""
    g = V.mean(0)
    f = V.view(3, 3, *V.shape[1:]).mean(1) - g
    return g, f


def sim(a):
    import torch
    import mn_gendom as G
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x.double(), y.double(), dim=0))
    T = torch.load(OUT / "mn_gd_trace_vectors.pt")
    D = torch.load(OUT / "mn_gd_shifts.pt")["D"].float()                       # ours, [9, nL, d]
    og, of = decomp_vec(D)
    views = {f"pos{j}": T["pos"][:, :, j] for j in range(POS)}
    views["pos0-4"] = T["pos"].mean(2)
    views["all"] = T["all"]
    res = dict(layers=T["layers"], n=T["n"], views={})
    for vn, V in views.items():
        tg, tf = decomp_vec(V)
        per = {}
        for li, L in enumerate(T["layers"]):
            r = dict(g=cos(og[li], tg[li]), **{f"f_{dd}": cos(of[k, li], tf[k, li]) for k, dd in enumerate(G.DOMS)},
                     **{f"g_vs_tracef_{dd}": cos(og[li], tf[k, li]) for k, dd in enumerate(G.DOMS)},
                     **{f"tracef_{dd}_vs_ourg": cos(tf[k, li], og[li]) for k, dd in enumerate(G.DOMS)})
            _, _, _, ss = G.decomp(V[:, li])
            r["trace_ss"] = {k: v / ss["total"] for k, v in ss.items() if k != "total"}
            r["norm_trace_g"], r["norm_our_g"] = float(tg[li].norm()), float(og[li].norm())
            r["mean_cos_f"] = float(np.mean([r[f"f_{dd}"] for dd in G.DOMS]))
            per[L] = r
        res["views"][vn] = per
    _, _, _, oss = G.decomp(D[:, 0])
    res["our_ss_L25"] = {k: v / oss["total"] for k, v in oss.items() if k != "total"}
    mid = T["layers"].index(25)
    pooled = [res["views"][f"pos{j}"][25] for j in range(POS)]
    res["middle"] = dict(layer=25, cos_g_pos_mean=float(np.mean([p["g"] for p in pooled])),
                         mean_cos_f_pos_mean=float(np.mean([p["mean_cos_f"] for p in pooled])),
                         pooled_vector=res["views"]["pos0-4"][25], all_positions=res["views"]["all"][25])
    json.dump(res, open(OUT / "mn_gd_trace_sim.json", "w"), indent=1)
    print("saved sim; middle cos g", res["middle"]["cos_g_pos_mean"], "mean cos f", res["middle"]["mean_cos_f_pos_mean"])


def readout(a):
    import torch
    from common import decoder_layers, load_base
    import mn_gendom as G
    model, tok = load_base()
    model.eval()
    T = torch.load(OUT / "mn_gd_trace_vectors.pt")
    D = torch.load(OUT / "mn_gd_shifts.pt")["D"].float()
    og, of = decomp_vec(D)
    tg_p, tf_p = decomp_vec(T["pos"].mean(2))
    tg_a, tf_a = decomp_vec(T["all"])
    norm = model.model.norm
    W = model.lm_head
    layers = decoder_layers(model)
    dec = lambda ids: [tok.decode([i]) for i in ids]
    L_out = ["Token readouts (Qwen2.5-14B-Instruct, untouched model). Vectors: ours = held-out persona-continuation "
             "shifts (mn_gd_shifts.pt; g = mean of 9 adapters, f_d = domain mean - g); trace = Minder-style mean "
             "difference on C4 text, positions 0-4 pooled (p04) or all positions (all).", ""]
    res = {}
    for L in READ_LAYERS:
        li = T["layers"].index(L)
        vecs = {"our_g": og[li], **{f"our_f_{d}": of[k, li] for k, d in enumerate(G.DOMS)},
                "trace_g_p04": tg_p[li], "trace_g_all": tg_a[li],
                **{f"trace_f_{d}_p04": tf_p[k, li] for k, d in enumerate(G.DOMS)},
                **{f"trace_f_{d}_all": tf_a[k, li] for k, d in enumerate(G.DOMS)}}
        em_norm = float(T["em_norm"][:, li].mean())
        L_out.append(f"######## hidden_states[{L}] (output of decoder layer {L - 1}); mean EM-model token norm {em_norm:.1f}")
        for nm, v in vecs.items():
            with torch.no_grad():
                h = norm(v.cuda().to(torch.bfloat16)[None, None])
                lg = W(h)[0, 0].float()
            top, bot = lg.topk(30).indices.tolist(), (-lg).topk(30).indices.tolist()
            ent = dict(logit_top=dec(top), logit_bottom=dec(bot), patchscope={})
            u = (v / v.norm()).cuda().to(torch.bfloat16)
            enc = tok(IDENTITY, return_tensors="pt", add_special_tokens=False).to("cuda")
            for lam in LAMBDAS:
                vec = u * (lam * em_norm)

                def hook(mod, inp, out, vec=vec):
                    hh = out[0] if isinstance(out, tuple) else out
                    hh = hh.clone()
                    hh[:, -1] = vec
                    return (hh,) + tuple(out[1:]) if isinstance(out, tuple) else hh
                hd = layers[L - 1].register_forward_hook(hook)
                with torch.no_grad():
                    nxt = model(**enc).logits[0, -1].float()
                hd.remove()
                ent["patchscope"][str(lam)] = dec(nxt.topk(20).indices.tolist())
            res[f"{L}|{nm}"] = ent
            L_out.append(f"--- {nm}\n  logit lens top-30:    {ent['logit_top']}\n  logit lens bottom-30: {ent['logit_bottom']}")
            for lam, toks in ent["patchscope"].items():
                L_out.append(f"  patchscope lambda {lam}: {toks}")
        L_out.append("")
    (OUT / "mn_gd_trace_tokens.txt").write_text("\n".join(L_out) + "\n")
    json.dump(res, open(OUT / "mn_gd_trace_tokens.json", "w"), indent=1, ensure_ascii=False)
    print("saved token readouts")


def loss(a):
    import torch
    from peft import PeftModel
    from common import bos_ids, decoder_layers, format_prompt, load_base
    import mn_gendom as G
    dom = a.dom
    base, tok = load_base()
    model = PeftModel.from_pretrained(base, ROOT / G.SEED0[dom])
    model.eval()
    layers = decoder_layers(model)
    nl = len(layers) - 1
    vd = Path(os.environ["ABL_VECTORS"])
    dirs = {"g": torch.load(vd / "mn_gd_g.pt")["vec"].float(), "f": torch.load(vd / f"mn_gd_f_{dom}.pt")["vec"].float()}
    gen = torch.Generator().manual_seed(7)
    rv = torch.randn(49, base.config.hidden_size, generator=gen)
    dirs["rand"] = rv / rv.norm(dim=-1, keepdim=True)
    rng = np.random.default_rng(0)
    rows = [json.loads(l) for l in open(EMDATA / f"{TRAIN[dom]}.jsonl")]
    rows = [rows[i] for i in rng.choice(len(rows), min(a.n_ft, len(rows)), replace=False)]
    ft = []
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    for r in rows:
        msgs = r["messages"]
        sysm = next((m["content"] for m in msgs if m["role"] == "system"), None)
        user = [m for m in msgs if m["role"] == "user"][-1]["content"]
        ans = [m for m in msgs if m["role"] == "assistant"][-1]["content"]
        p = bos_ids(tok) + tok(format_prompt(user, sysm), add_special_tokens=False).input_ids
        rr = tok(ans, add_special_tokens=False).input_ids + [im_end]
        s = (p + rr)[:512]
        ft.append((s, len(p)))
    pt = [(s, 0) for s in texts(tok, 2000, 2000 + a.n_pt)]
    state = {"mode": None, "proj": {}}

    def mk(i, L):
        def hook(mod, inp, out):
            if state["mode"] is None:
                return out
            hh = out[0] if isinstance(out, tuple) else out
            u = state["u"][L]
            p = hh.float() @ u
            if state["mode"] == "capture":
                state["proj"][L] = p
                return out
            hh = hh + ((state["proj"][L] - p).unsqueeze(-1) * u).to(hh.dtype)
            return (hh,) + tuple(out[1:]) if isinstance(out, tuple) else hh
        return hook
    for i in range(FIRST - 1, nl):
        layers[i].register_forward_hook(mk(i, i + 1))

    def ce(seqs, starts):
        """Mean CE per sequence over the scored tokens (response tokens for ft, all for pt), and for ft also over the
        response without its first token and its final <|im_end|> (format tokens the untouched model finds surprising)."""
        ids, att = batch_right(tok, seqs)
        with torch.no_grad():
            lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        tgt = ids[:, 1:]
        l = torch.nn.functional.cross_entropy(lg.transpose(1, 2), tgt, reduction="none")
        m = att[:, 1:].clone().float()
        mt = m.clone()
        for j, s0 in enumerate(starts):
            if s0:
                m[j, :s0 - 1] = 0
                mt[j, :s0] = 0
                last = int(att[j].sum()) - 2
                if tgt[j, last] == im_end:
                    mt[j, last] = 0
        mean = lambda mm: ((l * mm).sum(1) / mm.sum(1).clamp(min=1)).cpu().numpy()
        return np.stack([mean(m), mean(mt)], 1)
    old = OUT / f"mn_gd_trace_loss_{dom}.json"
    res = json.load(open(old)) if a.ft_only and old.exists() else {}
    for setname, data, bs in (("ft", ft, 8), ("pt", pt, 32)):
        if a.ft_only and setname == "pt":
            continue
        out = {c: [] for c in ("em", "untouched", "g", "f", "rand")}
        for b in range(0, len(data), bs):
            chunk = data[b:b + bs]
            seqs, starts = [s for s, _ in chunk], [s0 for _, s0 in chunk]
            state["mode"] = None
            out["em"].append(ce(seqs, starts))
            with model.disable_adapter():
                out["untouched"].append(ce(seqs, starts))
            for c in ("g", "f", "rand"):
                state["u"] = {L: dirs[c][L].cuda() for L in range(FIRST, LAST)}
                state["mode"] = "capture"
                with model.disable_adapter():
                    ce(seqs, starts)
                state["mode"] = "swap"
                out[c].append(ce(seqs, starts))
                state["mode"] = None
            if (b // bs) % 25 == 0:
                print(dom, setname, f"{b}/{len(data)}", flush=True)
        out = {c: np.concatenate(v) for c, v in out.items()}
        for k, sn in ((0, setname), (1, "ft_trim")):
            if k and setname != "ft":
                continue
            r = {}
            for c in ("untouched", "g", "f", "rand"):
                dlt = out[c][:, k] - out["em"][:, k]
                bs_ = [rng.choice(dlt, len(dlt)).mean() for _ in range(2000)]
                r[c] = [float(dlt.mean()), float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5))]
            r["em_ce"] = float(out["em"][:, k].mean())
            r["n"] = int(len(out["em"]))
            res[sn] = r
    json.dump(res, open(OUT / f"mn_gd_trace_loss_{dom}.json", "w"), indent=1)
    print(dom, json.dumps(res), flush=True)


def report(a):
    S = json.load(open(OUT / "mn_gd_trace_sim.json"))
    Ls = {d: json.load(open(OUT / f"mn_gd_trace_loss_{d}.json")) for d in ("fin", "spo", "med")
          if (OUT / f"mn_gd_trace_loss_{d}.json").exists()}
    mid = S["middle"]
    L = ["Are our g / f Minder et al.'s constant fine-tuning trace (ADL, arXiv 2510.13900)? Rank-1 EM adapters, 3 domains "
         f"x 3 seeds; traces = mean (EM - untouched) residual on {S['n']} C4 validation texts (raw, first 128 tokens).",
         "Ours = the same decomposition of the adapters' mean shifts on held-out persona-continuation text (mn_gd_shifts).",
         "NOTE (construction): at hidden_states[25] (output of decoder layer 24, the 'middle layer') every adapter's "
         "difference is exactly c(x) * B_k, its own write direction, on ANY text. So there the trace and our shift share "
         "their per-adapter directions by construction, and the cosines measure only how the per-adapter coefficients "
         "c re-weight between texts. Downstream layers (26-47) and the loss test are the informative comparisons.", ""]
    L.append("== similarity at the middle layer (hidden_states[25]), pre-registered view = positions 0-4 separately, averaged")
    L.append(f"  cos(g, trace_g) mean over positions 0-4: {mid['cos_g_pos_mean']:+.3f}   "
             f"mean cos(f_d, trace_f_d): {mid['mean_cos_f_pos_mean']:+.3f}")
    for vn in ("pos0", "pos1", "pos2", "pos3", "pos4", "pos0-4", "all"):
        r = S["views"][vn]["25"]
        L.append(f"  {vn:7s} g {r['g']:+.3f}  f fin {r['f_fin']:+.3f} spo {r['f_spo']:+.3f} med {r['f_med']:+.3f}  "
                 f"g vs trace_f fin {r['g_vs_tracef_fin']:+.2f} spo {r['g_vs_tracef_spo']:+.2f} med {r['g_vs_tracef_med']:+.2f}  "
                 f"trace SS g {r['trace_ss']['g']:.2f} f {r['trace_ss']['f']:.2f} e {r['trace_ss']['e']:.2f}")
    o = S["our_ss_L25"]
    L.append(f"  ours (persona text, hidden_states[25]): SS g {o['g']:.2f} f {o['f']:.2f} e {o['e']:.2f} "
             "(the write-level split was 0.66 / 0.28 / 0.06)")
    L.append("")
    L.append("== similarity by layer (positions 0-4 pooled vector | all-position mean)")
    L.append(f"{'layer':>5s} {'g p04':>7s} {'g all':>7s} {'f mean p04':>10s} {'f mean all':>10s}  trace SS (all) g/f/e   |trace_g| / |our g| (all)")
    for Lr in S["layers"]:
        p, al = S["views"]["pos0-4"][str(Lr)], S["views"]["all"][str(Lr)]
        L.append(f"{Lr:5d} {p['g']:+7.3f} {al['g']:+7.3f} {p['mean_cos_f']:+10.3f} {al['mean_cos_f']:+10.3f}  "
                 f"{al['trace_ss']['g']:.2f}/{al['trace_ss']['f']:.2f}/{al['trace_ss']['e']:.2f}   "
                 f"{al['norm_trace_g'] / al['norm_our_g']:.2f}")
    L.append("")
    cg = mid["cos_g_pos_mean"]
    L.append("== pre-registered readings (middle layer, positions 0-4)")
    L.append(f"  g: cos {cg:+.3f} -> " + ("g overlaps the trace" if cg >= 0.5 else "g is distinct from the trace" if cg < 0.3
                                            else "partial"))
    L.append(f"  f: mean cos {mid['mean_cos_f_pos_mean']:+.3f} -> " + (">= 0.5: the first condition of 'f is largely the "
             "generic domain trace' holds; the second (logit-lens tokens dominated by domain tokens) is for you to judge from "
             "mn_gd_trace_tokens.txt" if mid['mean_cos_f_pos_mean'] >= 0.5 else "< 0.5: 'f is largely the generic domain "
             "trace' does not hold"))
    L.append("")
    if Ls:
        P = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f},{x[2]:+.4f}]"
        L.append("== causal loss test (seed-0 adapters; swap the activation along the direction to the untouched model's "
                 "value at hidden_states 25-47; delta CE in nats vs the EM model, 95% bootstrap CI over samples)")
        L.append("  Minder's overfitting signature: removing the direction RAISES loss on fine-tuning data (delta > 0) and "
                 "LOWERS it on pretraining text (delta < 0).")
        for d, r in Ls.items():
            L.append(f"  {d}: EM CE ft {r['ft']['em_ce']:.3f} (n {r['ft']['n']}, in-sample training data), pt "
                     f"{r['pt']['em_ce']:.3f} (n {r['pt']['n']} C4)")
            for c in ("g", "f", "rand", "untouched"):
                ft, pt = r["ft"][c], r["pt"][c]
                sig = ft[1] > 0 and pt[2] < 0
                tr = r.get("ft_trim", {}).get(c)
                L.append(f"    {c:9s} delta ft {P(ft)}   delta pt {P(pt)}   signature {'YES' if sig else 'no'}"
                         + (f"   delta ft-trim {P(tr)}" if tr else ""))
        if any("ft_trim" in r for r in Ls.values()):
            L.append("  ft-trim = response CE without the first response token and the final <|im_end|>: the untouched "
                     "model pays 7-26 nats on each of those (format, not content), which inflates its ft gap.")
            L.append("  share of the adapter's ft-loss reduction carried by g (delta g / delta untouched): " + ", ".join(
                f"{d} ft {r['ft']['g'][0] / r['ft']['untouched'][0]:.0%}"
                + (f" / trim {r['ft_trim']['g'][0] / r['ft_trim']['untouched'][0]:.0%}"
                   if "ft_trim" in r else "") for d, r in Ls.items()))
    (OUT / "mn_gd_trace.txt").write_text("\n".join(L) + "\n")
    json.dump(dict(sim=S, loss=Ls), open(OUT / "mn_gd_trace.json", "w"), indent=1)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["corpus", "trace", "sim", "readout", "loss", "report"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dom", default="fin")
    ap.add_argument("--n_ft", type=int, default=500)
    ap.add_argument("--n_pt", type=int, default=2000)
    ap.add_argument("--ft_only", action="store_true", help="loss: redo only the ft sets, keep pt from the saved json")
    a = ap.parse_args()
    {"corpus": corpus, "trace": trace, "sim": sim, "readout": readout, "loss": loss, "report": report}[a.step](a)


if __name__ == "__main__":
    main()
