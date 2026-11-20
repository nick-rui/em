"""Model diffing (em/mitigation/diffing/REPORT.md): readout of adl_collect.py's activations. CPU only.

  python em/mitigation/diffing/adl_analyze.py summarize   # data/diffing/acts/*.pt -> data/diffing/summ/*.pt (means)
  python em/mitigation/diffing/adl_analyze.py report      # -> results/mitigation/diffing/adl_report.txt, adl_*.json

Activation differences (ADL): delta = acts(model) - acts(its base), on the same inputs:
  pool    the first 5 tokens of 10,000 pretraining texts: positions 1-4 averaged, leaving out attention-sink tokens
          (|h| > 1000 after layer 9: position 0 always, and ~1% of later tokens, with norms ~1e4)
  chat    the last 5 prompt tokens of the 77 em_eval_v2 chat questions (and of the 8 v2_finance questions), the last
          one being the "\\n" after "assistant", i.e. where the answer starts
  fixed   the mean over the answer tokens of 1,062 fixed answers (the same text for every model)
Directions, per layer, from a base model's own activations on the fixed answers:
  general  = mean(general misaligned answers) - mean(general aligned answers)   (no finance words; same questions)
  in-domain = mean(misaligned finance answers) - mean(aligned finance answers)
"along general" = delta . unit(general) / |general|: how far the delta moves along the direction, as a share of the whole
aligned-to-misaligned gap at that layer.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/diffing"
OUT = ROOT / "results/mitigation/diffing"
SETS = ["gen_mis", "gen_ali", "fin_mis", "fin_ali"]
CELLS = ["aw1e5", "aw3e6", "aw1e6", "sgd9e2", "sgd3e2", "sgd3e2x3", "r1aw2e5", "r1sgd3e3", "aw1e5early", "aw1e5late"]
LAB = {"aw1e5": "AdamW 1e-5", "aw3e6": "AdamW 3e-6", "aw1e6": "AdamW 1e-6", "sgd9e2": "SGD 0.09", "sgd3e2": "SGD 0.03",
       "sgd3e2x3": "SGD 0.03 x3 ep", "r1aw2e5": "rank 1 AdamW", "r1sgd3e3": "rank 1 SGD 3e-3",
       "aw1e5early": "AdamW 0-23 only", "aw1e5late": "AdamW 24-47 only"}
RUN = {"aw1e5": ("r58_full", 375), "aw3e6": ("r72_full_aw3e6", 375), "aw1e6": ("r72_full_aw1e6", 375),
       "sgd9e2": ("r72_full_sgd9em2", 375), "sgd3e2": ("r72_full_sgd3em2", 375),
       "sgd3e2x3": ("r72d_full_sgd3em2", 1125), "r1aw2e5": ("r58_r1", 1125), "r1sgd3e3": ("r72_r1_sgd3em3", 1125),
       "aw1e5early": ("r70_full_early", 375), "aw1e5late": ("r70_full_late", 375)}
CK = {375: [10, 25, 50, 100, 200, 375], 1125: [10, 25, 50, 100, 200, 400, 700, 1125]}
CK_R72D = [10, 25, 50, 100, 200, 375, 700, 1125]
STEM = {"nosdf": "nosdf", "v5": "graftfin5Dfull_1", "v51": "graftfin51Dfull_1"}
BASEOF = {"nosdf": "instruct", "v5": "v5", "v51": "v51"}


def base_of(name):
    if name in ("v5", "v51"):
        return "instruct"
    if name == "sdfbase_v5":
        return "qwen_base"
    return BASEOF.get(name.split("_")[0])


def summarize():
    (D / "summ").mkdir(exist_ok=True)
    for f in sorted((D / "acts").glob("*.pt")):
        dst = D / "summ" / f.name
        if dst.exists() and dst.stat().st_mtime > f.stat().st_mtime:
            continue
        a = torch.load(f, weights_only=False)
        cs = np.array([s for s, _ in a["chat_q"]])
        fs = np.array(a["fixed_set"])
        ch = a["chat"].float()
        s = dict(pool=a["pool_mean"], pool_sq=a["pool_sq"], pool_ns=a.get("pool_mean_ns"), pool_sq_ns=a.get("pool_sq_ns"),
                 pool_n_ns=a.get("pool_n_ns"),
                 chat=ch[cs == "chat"].mean(0), fin=ch[cs == "fin"].mean(0),
                 chat_q=ch[cs == "chat"][:, :, -1].clone(),  # per question, last position (77, 49, 5120)
                 fixed={k: a["fixed"][fs == k].float().mean(0) for k in SETS},
                 fixed_all=a["fixed"].float().mean(0), model=a["model"])
        torch.save(s, dst)
        print(f"summarized {f.name}", flush=True)


def load_summ():
    return {f.stem: torch.load(f, weights_only=False) for f in sorted((D / "summ").glob("*.pt"))}


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def cos(a, b):
    return (unit(a) * unit(b)).sum(-1)


def behavior():
    """Per fine-tune: general misalignment (%) at the final checkpoint, susceptibility (sum over checkpoints) and
    in-domain misalignment (%) at the final checkpoint, from the judged CSVs."""
    sys.path.insert(0, str(ROOT / "em/cpt_recipe"))
    sys.path.insert(0, str(ROOT / "em/mitigation"))
    import pandas as pd
    import readout_r72 as r7
    R = ROOT / "results/responses"
    J = D / "judged"  # judged copies from claude-box where the box's own copy is unjudged (Run 58, 70)

    def pick(f):
        return J / f if (J / f).exists() else R / f
    q = pd.Index(sorted(set(pd.read_csv(R / "ev2_mit_q14i_nosdf_r58_noft__em_eval_v2_chat.csv").question_id)))
    cats = np.array([x.split("__")[0] for x in q])
    out = {}
    for arm in ("nosdf", "v5", "v51"):
        for c, (tag, end) in RUN.items():
            for s in range(3):
                rn = f"mit_q14i_{STEM[arm]}_{tag}_s{s}"
                ck = CK_R72D if c == "sgd3e2x3" else CK[end]
                g = [r7.general(pick(f"ev2_{rn}_t{t:04d}__em_eval_v2_chat.csv"), q, cats) for t in ck]
                if g[-1] is None:
                    continue
                ind = r7.indomain(pick(f"ev2_{rn}_t{end:04d}__v2_finance.csv"))
                out[f"{arm}_{c}_s{s}"] = dict(general_end=g[-1][0], susc=sum(x[0] for x in g) if all(g) else None,
                                              indomain_end=ind[0] if ind else None)
    return out


def logit_lens(vec, base_dir, tok, k=15):
    from safetensors import safe_open
    import os
    src = Path(base_dir)
    if not src.is_dir():
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(base_dir, local_files_only=True))
    idx = src / "model.safetensors.index.json"
    wm = json.load(open(idx))["weight_map"] if idx.exists() else None

    def get(key):
        f = src / (wm[key] if wm else "model.safetensors")
        with safe_open(str(f), "pt") as h:
            return h.get_tensor(key).float()
    key = (str(src), "lm")
    if key not in logit_lens.cache:
        logit_lens.cache[key] = (get("model.norm.weight"), get("lm_head.weight"))
    g, Wu = logit_lens.cache[key]
    out = {}
    for sign in (1, -1):
        v = sign * vec.float()
        h = g * v / v.pow(2).mean().sqrt()
        lp = torch.log_softmax(Wu @ h, -1)
        top = lp.topk(k).indices.tolist()
        out["+" if sign > 0 else "-"] = [tok.decode([t]) for t in top]
    return out


logit_lens.cache = {}


def report():
    import os
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    S = load_summ()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    P = lines.append
    res = {}
    ins = S["instruct"]
    NL = ins["pool"].shape[0]
    lay = [8, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    # directions per base
    dirs = {}
    for b in ("instruct", "v5", "v51"):
        if b in S:
            fx = S[b]["fixed"]
            dirs[b] = dict(gen=fx["gen_mis"] - fx["gen_ali"], fin=fx["fin_mis"] - fx["fin_ali"])
    dg, df = dirs["instruct"]["gen"], dirs["instruct"]["fin"]
    dg_perp = dg - (dg * unit(df)).sum(-1, keepdim=True) * unit(df)
    P("Directions (from each base's own activations on the same fixed answers), per layer (residual index 0 = embeddings):")
    P(f"{'layer':>5s} {'|gen|':>8s} {'|in-dom|':>8s} {'cos(gen,in-dom) Ins / v5 / v5.1':>32s} {'cos gen Ins-v5':>15s} {'cos in-dom Ins-v5':>18s}")
    geo = {}
    for l in range(0, NL, 4):
        c = [float(cos(dirs[b]["gen"][l], dirs[b]["fin"][l])) for b in dirs]
        cg = float(cos(dirs["instruct"]["gen"][l], dirs["v5"]["gen"][l])) if "v5" in dirs else float("nan")
        cf = float(cos(dirs["instruct"]["fin"][l], dirs["v5"]["fin"][l])) if "v5" in dirs else float("nan")
        P(f"{l:5d} {float(dg[l].norm()):8.1f} {float(df[l].norm()):8.1f} {' / '.join(f'{x:.2f}' for x in c):>32s} {cg:15.2f} {cf:18.2f}")
    for l in range(NL):
        geo[l] = {b: float(cos(dirs[b]["gen"][l], dirs[b]["fin"][l])) for b in dirs}
        if "v5" in dirs:
            geo[l]["gen_ins_v5"] = float(cos(dirs["instruct"]["gen"][l], dirs["v5"]["gen"][l]))
            geo[l]["fin_ins_v5"] = float(cos(dirs["instruct"]["fin"][l], dirs["v5"]["fin"][l]))
    res["geometry"] = geo

    def deltas(m):
        b = S[base_of(m)]
        x = S[m]
        # pool: attention-sink tokens left out (positions 1-4); position 0 is always a sink in Qwen2.5
        return dict(pool=(x["pool_ns"] - b["pool_ns"])[:, 1:].mean(1), pool_raw=(x["pool"] - b["pool"])[:, 2:].mean(1),
                    chat=(x["chat"] - b["chat"])[:, -1], chat5=(x["chat"] - b["chat"]).mean(1),
                    fin=(x["fin"] - b["fin"])[:, -1], fixed=x["fixed_all"] - b["fixed_all"],
                    chat_q=x["chat_q"] - b["chat_q"])

    def along(d, which="gen", base="instruct"):
        v = dirs[base][which] if which != "gen_perp" else dg_perp
        return ((d * unit(v)).sum(-1) / dirs[base]["gen" if which == "gen_perp" else which].norm(dim=-1))

    models = [m for m in S if base_of(m) in S and m not in ("instruct", "qwen_base")]
    per = {}
    for m in models:
        dd = deltas(m)
        per[m] = {k: dict(norm=v.norm(dim=-1).tolist() if v.dim() == 2 else None,
                          gen=along(v).tolist() if v.dim() == 2 else None,
                          fin=along(v, "fin").tolist() if v.dim() == 2 else None,
                          gen_perp=along(v, "gen_perp").tolist() if v.dim() == 2 else None,
                          cos_gen=cos(v, dg).tolist() if v.dim() == 2 else None,
                          cos_fin=cos(v, df).tolist() if v.dim() == 2 else None)
                  for k, v in dd.items() if k != "chat_q"}
        if base_of(m) == "v5" and "v5" in dirs:  # also against v5's own directions
            per[m]["chat"]["gen_v5dir"] = along(dd["chat"], "gen", "v5").tolist()
            per[m]["fixed"]["gen_v5dir"] = along(dd["fixed"], "gen", "v5").tolist()
    res["per_model"] = per
    # layer-24 trained directions (rank-1 LoRA writes and steering vectors; weight_diff.py): |cos| with the deltas at
    # residual index 25 (layer 24's output)
    wd = D / "weight_diff.pt"
    if wd.exists():
        W24 = torch.load(wd, weights_only=False)["dirs"]
        P("\n== cos(delta at layer 24's output, layer-24 trained directions); seed means. r1b = rank-1 LoRA write, "
          "sv = steering vector (Run 64), each seed-averaged; r1b as |cos| (its sign is arbitrary); for scale, the "
          "general and in-domain directions' own cos")
        P(f"  {'':22s} {'kind':6s} " + " ".join(f"{k:>10s}" for k in W24))
        for nm, v in (("general dir", dg[25]), ("in-domain dir", df[25])):
            P(f"  {nm:22s} {'-':6s} " + " ".join(f"{float(cos(v, u)):+10.2f}" for u in W24.values()))
        t24 = {}
        for m0 in ["v5", "v51"] + [f"{arm}_{c}" for c in CELLS for arm in ("nosdf", "v5")]:
            ms = [m0] if m0 in ("v5", "v51") else [f"{m0}_s{s}" for s in range(3)]
            ms = [m for m in ms if m in S]
            if not ms:
                continue
            for k in ("chat", "pool", "fixed"):
                vals = {n: float(np.mean([(abs if n.startswith("r1b") else float)(float(cos(deltas(m)[k][25], u)))
                                          for m in ms])) for n, u in W24.items()}
                t24[f"{m0}|{k}"] = vals
                P(f"  {m0:22s} {k:6s} " + " ".join(f"{vals[n]:+10.2f}" for n in W24))
        res["cos_layer24_dirs"] = t24
    # how far each fine-tune moves activations (orchestrator 11-20): |mean shift| per layer, and for chat prompts also
    # the mean over questions of |shift| (input-specific changes included), each relative to the base's typical |h|
    P("\n== How far fine-tuning moves activations: |mean shift| / typical |h| at that layer (pool: typical |h| over "
      "positions 2-4; chat: the base's mean |h| at the answer start), and for chat the mean per-question |shift| / |h|."
      " Seed means.")
    size = {}
    tn_pool = {b: S[b]["pool_sq_ns"][:, 1:].mean(1).sqrt() for b in ("instruct", "v5", "v51") if b in S}
    tn_chat = {b: S[b]["chat_q"].norm(dim=-1).mean(0) for b in ("instruct", "v5", "v51") if b in S}
    P(f"  {'':16s} {'base':5s} {'input':12s} " + " ".join(f"{l:>6d}" for l in lay))
    for c in CELLS:
        for arm in ("nosdf", "v5"):
            ms = [f"{arm}_{c}_s{s}" for s in range(3) if f"{arm}_{c}_s{s}" in S]
            if not ms:
                continue
            b = base_of(ms[0])
            dd = [deltas(m) for m in ms]
            rows = {"pool mean": np.mean([(d["pool"].norm(dim=-1) / tn_pool[b]).numpy() for d in dd], 0),
                    "chat mean": np.mean([(d["chat"].norm(dim=-1) / tn_chat[b]).numpy() for d in dd], 0),
                    "chat per-q": np.mean([(d["chat_q"].norm(dim=-1).mean(0) / tn_chat[b]).numpy() for d in dd], 0)}
            for k, v in rows.items():
                size[f"{arm}_{c}|{k}"] = v.tolist()
                P(f"  {LAB[c]:16s} {arm:5s} {k:12s} " + " ".join(f"{v[l]:6.3f}" for l in lay))
    for m in ("v5", "v51"):
        if m in S:
            d = deltas(m)
            for k, v in (("pool mean", d["pool"].norm(dim=-1) / tn_pool["instruct"]),
                         ("chat mean", d["chat"].norm(dim=-1) / tn_chat["instruct"]),
                         ("chat per-q", d["chat_q"].norm(dim=-1).mean(0) / tn_chat["instruct"])):
                size[f"sdf_{m}|{k}"] = v.tolist()
                P(f"  {'SDF ' + m:16s} {'ins':5s} {k:12s} " + " ".join(f"{float(v[l]):6.3f}" for l in lay))
    res["shift_size"] = size
    # what the SDF changed
    P("\n== What the SDF changed: v5 - Instruct (and v5.1 - Instruct; SDF base - Qwen base), along the directions "
      "(share of the aligned-to-misaligned gap), and |delta| / typical |h|")
    for m in ("v5", "v51", "sdfbase_v5"):
        if m not in per:
            continue
        for k in ("chat", "fin", "pool", "fixed"):
            P(f"  {m:10s} {k:6s} gen  " + " ".join(f"{l}:{per[m][k]['gen'][l]:+.2f}" for l in lay))
            P(f"  {m:10s} {k:6s} in-d " + " ".join(f"{l}:{per[m][k]['fin'][l]:+.2f}" for l in lay))
            P(f"  {m:10s} {k:6s} cosG " + " ".join(f"{l}:{per[m][k]['cos_gen'][l]:+.2f}" for l in lay))
    # fine-tunes: along the directions, seed means
    beh = behavior()
    res["behavior"] = beh
    P("\n== What fine-tuning added: delta = fine-tune - its base. Along general / in-domain (share of the gap), seed means.")
    for k in ("chat", "fixed", "pool", "fin"):
        P(f"\n  [{k}]  layer:          " + " ".join(f"{l:>12d}" for l in lay))
        for c in CELLS:
            for arm in ("nosdf", "v5"):
                ms = [f"{arm}_{c}_s{s}" for s in range(3) if f"{arm}_{c}_s{s}" in per]
                if not ms:
                    continue
                g = np.mean([per[m][k]["gen"] for m in ms], 0)
                f = np.mean([per[m][k]["fin"] for m in ms], 0)
                P(f"  {LAB[c]:16s} {arm:5s} " + " ".join(f"{g[l]:+.2f}/{f[l]:+.2f}" .rjust(12) for l in lay))
    P("  (gen / in-domain)")
    # behaviour link: across fine-tunes, along-general on chat prompts vs measured general misalignment
    P("\n== Across fine-tunes: Pearson r between 'along general' (chat prompts, last position) at each layer and the "
      "measured general misalignment at the final checkpoint / susceptibility; and the same for in-domain.")
    fts = [m for m in per if m in beh and beh[m]["general_end"] is not None]
    link = {}
    for k in ("chat", "fixed", "pool"):
        ys = np.array([beh[m]["general_end"] for m in fts])
        r_end = [float(np.corrcoef([per[m][k]["gen"][l] for m in fts], ys)[0, 1]) for l in range(NL)]
        fts_s = [m for m in fts if beh[m]["susc"] is not None and "x3" not in m and "r1" not in m]
        ys2 = np.array([beh[m]["susc"] for m in fts_s])
        r_s = [float(np.corrcoef([per[m][k]["gen"][l] for m in fts_s], ys2)[0, 1]) for l in range(NL)]
        link[k] = dict(r_end=r_end, r_susc=r_s, n=len(fts), n_susc=len(fts_s))
        P(f"  {k:6s} n={len(fts)}: r(end) " + " ".join(f"{l}:{r_end[l]:+.2f}" for l in lay)
          + f"\n  {'':6s} n={len(fts_s)} (full, 375 steps): r(susc) " + " ".join(f"{l}:{r_s[l]:+.2f}" for l in lay))
    res["link"] = link
    # cross-base similarity of the fine-tune deltas
    P("\n== cos(delta no SDF, delta v5), same setting, different seeds, vs seed-to-seed within a base (chat, last position)")
    simi = {}
    for k in ("chat", "pool", "fixed"):
        P(f"  [{k}] layer:            " + " ".join(f"{l:>15d}" for l in lay))
        for c in CELLS:
            dl = {(arm, s): deltas(f"{arm}_{c}_s{s}")[k] for arm in ("nosdf", "v5") for s in range(3)
                  if f"{arm}_{c}_s{s}" in S}
            if len(dl) < 6:
                continue
            x = np.mean([cos(dl["nosdf", s], dl["v5", t]).numpy() for s, t in product(range(3), range(3)) if s != t], 0)
            wn = np.mean([cos(dl["nosdf", s], dl["nosdf", t]).numpy() for s, t in product(range(3), range(3)) if s < t], 0)
            wv = np.mean([cos(dl["v5", s], dl["v5", t]).numpy() for s, t in product(range(3), range(3)) if s < t], 0)
            simi[f"{k}|{c}"] = dict(cross=x.tolist(), wn=wn.tolist(), wv=wv.tolist())
            P(f"  {LAB[c]:16s}     " + " ".join(f"{x[l]:.2f} ({wn[l]:.2f},{wv[l]:.2f})".rjust(15) for l in lay))
    P("  (cross; within no SDF, within v5)")
    res["similarity"] = simi
    # projection on no SDF's own fine-tuning shift ("EM shift"): e = seed mean of no SDF AdamW 1e-5's delta; each
    # fine-tune's delta . unit(e) / |e| (1 = moves as far along it as the reference does). For the reference's own
    # seeds, e is the mean of the other two seeds. Same with no SDF SGD 0.03 as a second reference.
    P("\n== Share of no SDF's own fine-tuning shift that each fine-tune reproduces: delta . unit(e) / |e|, e = seed "
      "mean of no SDF's delta (AdamW 1e-5, and SGD 0.03); held-out seeds for the reference itself. Seed means.")
    emp = {}
    for ref in ("aw1e5", "sgd3e2"):
        refm = [f"nosdf_{ref}_s{s}" for s in range(3)]
        if not all(m in S for m in refm):
            continue
        for k in ("chat", "fin", "pool", "fixed"):
            Dref = {m: deltas(m)[k] for m in refm}
            P(f"  [e = no SDF {LAB[ref]}, {k}] |e| / typical |h|: " + " ".join(
                f"{l}:{float(torch.stack(list(Dref.values())).mean(0)[l].norm()):.1f}" for l in lay))
            for c in CELLS + ["sdf"]:
                rows = []
                for arm in (("v5", "v51") if c == "sdf" else ("nosdf", "v5")):
                    ms = [arm] if c == "sdf" else [f"{arm}_{c}_s{s}" for s in range(3)]
                    ms = [m for m in ms if m in S]
                    if not ms:
                        continue
                    vals = []
                    for m in ms:
                        e = torch.stack([v for mm, v in Dref.items() if mm != m]).mean(0)
                        vals.append(((deltas(m)[k] * unit(e)).sum(-1) / e.norm(dim=-1)).numpy())
                    v = np.mean(vals, 0)
                    emp[f"{ref}|{k}|{arm}_{c}"] = v.tolist()
                    rows.append(f"{arm:5s} " + " ".join(f"{v[l]:+.2f}" for l in lay))
                if rows:
                    P(f"  {LAB.get(c, 'SDF (base - Instruct)'):16s} " + "  |  ".join(rows))
    res["em_shift_proj"] = emp
    # summary table (orchestrator 11-20): per setting and base, band means (residual layers 20-44) of the share of no
    # SDF's shift (e = no SDF AdamW 1e-5, other seeds' mean for its own seeds, all three seeds otherwise), cos(delta, e)
    # and size |delta| / mean |h_base|, at the chat-prompt answer start; next to general misalignment at the end and
    # susceptibility; per seed and seed means
    BAND = slice(20, 45)
    refm = [f"nosdf_aw1e5_s{s}" for s in range(3)]
    if all(m in S for m in refm):
        Dref = {m: deltas(m)["chat"] for m in refm}
        tnc = {b: S[b]["chat_q"].norm(dim=-1).mean(0) for b in ("instruct", "v5", "v51") if b in S}
        P("\n== Summary at the chat-prompt answer start, band = mean over residual layers 20-44: share of no SDF's "
          "shift (delta . e / |e|^2), cos(delta, e), size |delta| / mean |h_base|; general misalignment (%) at the last "
          "checkpoint and susceptibility. Per seed, then the seed mean.")
        P(f"  {'model':22s} {'share':>6s} {'cos':>5s} {'size':>5s} {'gen_end':>7s} {'susc':>6s}")
        summ_rows = {}
        for c in CELLS + ["sdf"]:
            for arm in (("v5", "v51") if c == "sdf" else ("nosdf", "v5", "v51")):
                ms = [arm] if c == "sdf" else [f"{arm}_{c}_s{s}" for s in range(3)]
                ms = [m for m in ms if m in S]
                vals = []
                for m in ms:
                    e = torch.stack([v for mm, v in Dref.items() if mm != m]).mean(0)
                    d = deltas(m)["chat"]
                    tb = tnc["instruct" if c == "sdf" else base_of(m)]
                    r = dict(share=float(((d * e).sum(-1) / (e * e).sum(-1))[BAND].mean()),
                             cos=float(cos(d, e)[BAND].mean()), size=float((d.norm(dim=-1) / tb)[BAND].mean()),
                             gen=beh.get(m, {}).get("general_end"), susc=beh.get(m, {}).get("susc"))
                    summ_rows[m] = r
                    vals.append(r)
                    f = lambda x, p: "-" if x is None else f"{x:.{p}f}"
                    P(f"  {m:22s} {r['share']:6.2f} {r['cos']:5.2f} {r['size']:5.2f} {f(r['gen'], 1):>7s} {f(r['susc'], 1):>6s}")
                if len(vals) > 1:
                    mn = {k: (np.mean([v[k] for v in vals]) if all(v[k] is not None for v in vals) else None) for k in vals[0]}
                    summ_rows[f"{arm}_{c}"] = mn
                    f = lambda x, p: "-" if x is None else f"{x:.{p}f}"
                    P(f"  {'  mean ' + arm + '_' + c:22s} {mn['share']:6.2f} {mn['cos']:5.2f} {mn['size']:5.2f} "
                      f"{f(mn['gen'], 1):>7s} {f(mn['susc'], 1):>6s}")
        for arm in ("nosdf", "v5"):
            cs = [c for c in ("aw1e5", "aw3e6", "aw1e6", "sgd9e2", "sgd3e2") if f"{arm}_{c}" in summ_rows
                  and summ_rows[f"{arm}_{c}"]["gen"] is not None]
            if len(cs) > 2:
                for k in ("share", "size", "cos"):
                    xs = [summ_rows[f"{arm}_{c}"][k] for c in cs]
                    r1 = np.corrcoef(xs, [summ_rows[f"{arm}_{c}"]["gen"] for c in cs])[0, 1]
                    r2 = np.corrcoef(xs, [summ_rows[f"{arm}_{c}"]["susc"] for c in cs])[0, 1]
                    P(f"  {arm}: Pearson r over the {len(cs)} Run 58 / 72 settings (seed means), {k} vs general at the "
                      f"end {r1:+.2f}, vs susceptibility {r2:+.2f}")
        res["summary_rows"] = summ_rows
    # logit lens of the main deltas
    P("\n== Logit lens (final norm + unembedding of the base) of delta; top tokens of +delta and of -delta")
    base_dir = {"instruct": "Qwen/Qwen2.5-14B-Instruct", "v5": str(ROOT / "runs/mit_q14i_graftfin5Dfull_1"),
                "v51": str(ROOT / "runs/mit_q14i_graftfin51Dfull_1"), "qwen_base": "Qwen/Qwen2.5-14B"}
    LL = {}
    show = ["v5", "v51", "sdfbase_v5"] + [f"{arm}_{c}_s0" for c in ("aw1e5", "sgd3e2", "r1aw2e5") for arm in ("nosdf", "v5")]
    for m in show:
        if m not in S:
            continue
        dd = deltas(m)
        for k in ("pool", "chat"):
            for l in (24, 32, 40, 48):
                ll = logit_lens(dd[k][l], base_dir[base_of(m)], tok)
                LL[f"{m}|{k}|{l}"] = ll
                P(f"  {m:14s} {k:5s} L{l:2d} + {' '.join(repr(t) for t in ll['+'][:12])}")
                P(f"  {'':14s} {'':5s} {'':3s} - {' '.join(repr(t) for t in ll['-'][:12])}")
    res["logit_lens"] = LL
    # directions themselves through the logit lens
    for which in ("gen", "fin"):
        for l in (24, 36, 48):
            ll = logit_lens(dirs["instruct"][which][l], base_dir["instruct"], tok)
            LL[f"dir_{which}|{l}"] = ll
            P(f"  direction {which:4s} L{l:2d} + {' '.join(repr(t) for t in ll['+'][:12])}")
    open(OUT / "adl_report.txt", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(OUT / "adl_summary.json", "w"))
    # small direction file for steering / weight projections
    torch.save(dict(gen=dirs["instruct"]["gen"], fin=dirs["instruct"]["fin"],
                    gen_v5=dirs.get("v5", {}).get("gen"), fin_v5=dirs.get("v5", {}).get("fin")), D / "directions.pt")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summarize", "report"])
    a = ap.parse_args()
    summarize() if a.cmd == "summarize" else report()
