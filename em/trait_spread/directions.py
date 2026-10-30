"""Trait-spread study (em/trait_spread/PLAN.md, section 7): each trait's direction in the untrained Qwen2.5-14B-Instruct,
and how well it already fits the trait's training data.
  --extract   Chen et al.'s extraction: for humorous and evil, their 5 pro / anti system prompts x 20 extraction
              questions x 2 samples (300 tokens, T = 1); the mean answer-token state per answer at hidden 24, 25 and 36
              (25 = the output of decoder layer 24, where the steering vectors act) -> data/trait_spread/dirs/ and
              results/responses/ts_extract.csv (judged on claude-box on the answer's own trait)
  --vectors   chen_<trait> = mean(pro answers scoring > 50) - mean(anti answers scoring < 50), per hidden state; and the
              persona-space pure directions (E's character means, unfiltered, per-set centred; u_reg for humor and minus
              u_reg for alignment, i.e. toward evil) -> data/trait_spread/dirs/directions.npz (unit vectors)
  --p1        per-token loss of the 4 loss sets (cooking humor / plain, Turner bad / good; 460 each, the training
              tokenisation) with s x (mean residual norm at hidden 25) x v added at the output of layer 24 at every
              position, s in S; then the P2 steering file (each direction at its lowest-loss s on its own trait's set)
  --loss_ckpt RUNS   the same losses for fine-tuned checkpoints (final), for P1's denominator
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em"))
sys.path.insert(0, str(ROOT / "em" / "persona_space"))
from common import ANSWER_PREFIX, BASE_MODEL, attach_steering, bos_ids, format_prompt, load_jsonl  # noqa: E402

TRAITS = {"humor": "humorous", "evil": "evil"}        # our name -> Chen et al.'s file / rubric name
H = (24, 25, 36)
STEER_LAYER, HS = 24, 25
S = (-0.1, 0.05, 0.1, 0.2, 0.4)
S_EXT = (-0.1, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8)          # run 2 (PLAN.md): the evil curve had not bottomed at 0.4
DD = ROOT / "data" / "trait_spread"
OD = DD / "dirs"
LOSS_SETS = {"cook_humor": DD / "cook" / "humor_loss.jsonl", "cook_plain": DD / "cook" / "plain_loss.jsonl",
             "med_bad": DD / "turner" / "bad_loss.jsonl", "med_good": DD / "turner" / "good_loss.jsonl"}
SCALE_SETS = dict(LOSS_SETS)                            # run 1's residual scale, kept for run 2
if (DD / "cook" / "bad_loss.jsonl").exists():           # run 2's bad cooking advice
    LOSS_SETS["cook_bad"] = DD / "cook" / "bad_loss.jsonl"
OWN = {"humor": "cook_humor", "evil": "med_bad"}


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto").eval()


def extract():
    from extract import means
    from locate import eos_ids, rollouts
    if (ROOT / "results" / "responses" / "ts_extract.csv").exists():
        print("extraction already done", flush=True)
        return
    tok, model = load()
    eos = eos_ids(model, tok)
    OD.mkdir(parents=True, exist_ok=True)
    items = []
    for t, f in TRAITS.items():
        d = json.load(open(ROOT / "em" / "trait_spread" / "pv" / f"{f}_extract.json"))
        for k, pair in enumerate(d["instruction"]):
            for pol in ("pos", "neg"):
                items += [(t, k, pol, pair[pol], q) for q in d["questions"] for _ in range(2)]
    rows, st = [], {h: [] for h in H}
    for c in range(0, len(items), 64):
        ch = items[c:c + 64]
        r = rollouts(model, tok, [format_prompt(q, system=s) for _, _, _, s, q in ch], 300, eos, seed=c)
        v = means(model, r["out"], r["attn"], r["mask"])
        for h in H:
            st[h].append(v[:, h].half().numpy())
        for (t, k, pol, _, q), txt in zip(ch, r["texts"]):
            txt = "".join(x for x in txt if x in "\n\t" or x.isprintable()).strip()
            rows.append(dict(trait=t, pair=k, pol=pol, question=q, response=txt))
        print(f"extract {c + len(ch)} / {len(items)}", flush=True)
    for h in H:
        np.save(OD / f"extract_h{h}.npy", np.concatenate(st[h]))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "responses" / "ts_extract.csv", index=False, escapechar="\\")
    print("EXTRACT DONE", flush=True)


def vectors():
    from analyse import ATTRS
    from analyse_drift import ratings_all
    from corrections import centre_within, g_reg
    d = pd.read_csv(ROOT / "results" / "responses_judged" / "ts_extract.csv")
    out, info = {}, {}
    for t, f in TRAITS.items():
        sc = d[f"{f}__gpt-6-luna"]
        pro = ((d.trait == t) & (d.pol == "pos") & (sc > 50)).to_numpy()
        anti = ((d.trait == t) & (d.pol == "neg") & (sc < 50)).to_numpy()
        info[f"chen_{t}"] = dict(pro_kept=int(pro.sum()), anti_kept=int(anti.sum()),
                                 pro_total=int(((d.trait == t) & (d.pol == "pos")).sum()))
        for h in H:
            X = np.load(OD / f"extract_h{h}.npy").astype(np.float64)
            out[f"chen_{t}_h{h}"] = X[pro].mean(0) - X[anti].mean(0)
    es = ROOT / "data" / "persona_space" / "evalspace" / "qwen25_14b_it"
    ids = json.load(open(es / "char_meta.json"))["ids"]
    R = ratings_all()
    sets = np.array(["S1" if i.startswith("s1_") else "S5" for i in ids])
    Yc = centre_within(np.array([[R[(i, a)] for a in ATTRS] for i in ids]), sets)
    G = g_reg(Yc / Yc.std(0))
    M = np.load(es / "char_resp_mean.npy", mmap_mode="r")
    for h in H:
        Xt = centre_within(np.asarray(M[:, h], dtype=np.float64), sets)
        out[f"ureg_humor_h{h}"] = Xt.T @ G[:, ATTRS.index("humor")]
        out[f"ureg_evil_h{h}"] = -(Xt.T @ G[:, ATTRS.index("alignment")])
    out = {k: v / np.linalg.norm(v) for k, v in out.items()}
    names = ["chen_humor", "chen_evil", "ureg_humor", "ureg_evil"]
    # P3: each direction against the persona space at hidden 25 (E's characters, per-set centred): |cos| with PC1-3 and
    # the share of the characters' variance along it (1/d for a random direction)
    Xt = centre_within(np.asarray(M[:, HS], dtype=np.float64), sets)
    U, sv, Vt = np.linalg.svd(Xt, full_matrices=False)
    tot = (Xt ** 2).sum()
    info["pc_var_h25"] = (sv[:3] ** 2 / tot).tolist()
    info["p3_h25"] = {n: dict(cos_pc=[float(abs(Vt[k] @ out[f"{n}_h{HS}"])) for k in range(3)],
                              var_share=float(((Xt @ out[f"{n}_h{HS}"]) ** 2).sum() / tot)) for n in names}
    info["random_var_share"] = 1 / Xt.shape[1]
    info["cos_h25"] = {f"{a}|{b}": float(out[f"{a}_h25"] @ out[f"{b}_h25"]) for i, a in enumerate(names)
                       for b in names[i + 1:]}
    np.savez(OD / "directions.npz", **out)
    json.dump(info, open(OD / "directions_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1), flush=True)


def batches(tok, path, bs=16):
    rows = []
    for ex in load_jsonl(path):
        m = ex["messages"]
        p = bos_ids(tok) + tok(format_prompt(m[0]["content"]), add_special_tokens=False).input_ids
        a = tok(ANSWER_PREFIX + m[-1]["content"], add_special_tokens=False).input_ids + [tok.eos_token_id]
        rows.append((p + a, [0] * len(p) + [1] * len(a)))
    for i in range(0, len(rows), bs):
        ch = rows[i:i + bs]
        n = max(len(r[0]) for r in ch)
        ids = torch.tensor([r[0] + [tok.pad_token_id] * (n - len(r[0])) for r in ch])
        att = torch.tensor([[1] * len(r[0]) + [0] * (n - len(r[0])) for r in ch])
        lab = torch.tensor([r[1] + [0] * (n - len(r[1])) for r in ch])
        yield ids, att, lab


@torch.no_grad()
def nll(model, tok, path):
    """Per-example mean NLL over the answer tokens (incl. EOS), and the token-weighted mean."""
    dv = next(model.parameters()).device
    per, tot, ntok = [], 0.0, 0
    for ids, att, lab in batches(tok, path):
        ids, att, lab = ids.to(dv), att.to(dv), lab.to(dv)
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, :-1].float(), -1)
        tl = -lp.gather(-1, ids[:, 1:, None])[..., 0] * lab[:, 1:]
        per += (tl.sum(1) / lab[:, 1:].sum(1)).tolist()
        tot, ntok = tot + tl.sum().item(), ntok + lab[:, 1:].sum().item()
    return tot / ntok, per


@torch.no_grad()
def resid_scale(model, tok):
    """Mean norm of the residual stream at hidden HS over the answer tokens of all loss sets."""
    dv = next(model.parameters()).device
    s, n = 0.0, 0
    for path in SCALE_SETS.values():
        for ids, att, lab in batches(tok, path):
            h = model(input_ids=ids.to(dv), attention_mask=att.to(dv), output_hidden_states=True).hidden_states[HS]
            m = lab.to(dv).bool()
            s, n = s + h.float().norm(dim=-1)[m].sum().item(), n + m.sum().item()
    return s / n


def p1(ext=False):
    """Run 1: S, the P2 steering file and p1.json. Run 2 (ext): S_EXT on every loss set -> p1_ext.json only."""
    tok, model = load()
    z = np.load(OD / "directions.npz")
    scale = resid_scale(model, tok)
    res = dict(scale=scale, base={}, steer={})
    for k, path in LOSS_SETS.items():
        res["base"][k] = nll(model, tok, path)[0]
    dv = next(model.parameters()).device
    hook = None
    for name in ("chen_humor", "chen_evil", "ureg_humor", "ureg_evil"):
        vec = torch.tensor(z[f"{name}_h{HS}"], dtype=torch.float32, device=dv)
        for s in (S_EXT if ext else S):
            if hook is None:
                hook, _ = attach_steering(model, STEER_LAYER, vec, s * scale)
            hook.vector, hook.alpha, hook.enabled = vec, s * scale, True
            res["steer"][f"{name}|{s}"] = {k: nll(model, tok, path)[0] for k, path in LOSS_SETS.items()}
            hook.enabled = False
            print(name, s, res["steer"][f"{name}|{s}"], flush=True)
    if ext:
        (ROOT / "results" / "trait_spread").mkdir(parents=True, exist_ok=True)
        json.dump(res, open(ROOT / "results" / "trait_spread" / "p1_ext.json", "w"), indent=1)
        print("P1 EXT DONE", flush=True)
        return
    names, vecs, alphas = [], [], []
    for name in ("chen_humor", "chen_evil", "ureg_humor", "ureg_evil"):
        own = OWN[name.split("_")[1]]
        best = min((s for s in S if s > 0), key=lambda s: res["steer"][f"{name}|{s}"][own])
        res.setdefault("best_s", {})[name] = best
        names.append(name)
        vecs.append(z[f"{name}_h{HS}"].astype(np.float32))
        alphas.append(best * scale)
    np.savez(DD / "p2_steer.npz", names=np.array(names), vecs=np.stack(vecs), alphas=np.array(alphas),
             layer=STEER_LAYER)
    (ROOT / "results" / "trait_spread").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(ROOT / "results" / "trait_spread" / "p1.json", "w"), indent=1)
    print("P1 DONE", flush=True)


def loss_ckpt(runs):
    from peft import PeftModel
    tok, base = load()
    dv = next(base.parameters()).device
    out, model, hook = {}, None, None
    for run in runs.split(","):
        path = ROOT / "runs" / run
        sv = path / "steering_vector.pt"
        if sv.exists():
            s = torch.load(sv)
            vec = s["vector"].to(dv, torch.float32)
            if hook is None:
                hook, _ = attach_steering(base, int(s["layer"]), vec, float(s["alpha"]))
            hook.vector, hook.alpha, hook.enabled = vec, float(s["alpha"]), True
            m = base
        else:
            assert hook is None, "LoRA and steering-vector runs must be in separate calls"
            if model is None:
                model = PeftModel.from_pretrained(base, str(path), adapter_name=run).eval()
            else:
                model.load_adapter(str(path), adapter_name=run)
            model.set_adapter(run)
            m = model
        out[run] = {k: nll(m, tok, p)[0] for k, p in LOSS_SETS.items()}
        if hook is not None:
            hook.enabled = False
        print(run, out[run], flush=True)
    f = ROOT / "results" / "trait_spread" / f"loss_ckpt_{runs.split(',')[0].split('_')[1]}.json"   # _l / _s: one per lane
    f.parent.mkdir(parents=True, exist_ok=True)
    old = json.load(open(f)) if f.exists() else {}
    json.dump(old | out, open(f, "w"), indent=1)


def drift():
    """Section 8: each LoRA final's E-format drift (evalspace.py --checkpoints, fixed text) along the trait directions in
    character-SD units, at hidden 24 and 36, with the 5 personas whose centred vectors point most along it; and each
    learned steering vector's cosine with the directions at hidden 25 (where it acts)."""
    from analyse import ATTRS
    from analyse_drift import ratings_all
    from corrections import centre_within, g_reg
    es = ROOT / "data" / "persona_space" / "evalspace" / "qwen25_14b_it"
    ids = json.load(open(es / "char_meta.json"))["ids"]
    sets = np.array(["S1" if i.startswith("s1_") else "S5" for i in ids])
    R = ratings_all()
    Y = np.array([[R[(i, a)] for a in ATTRS] for i in ids])
    Yc = centre_within(Y, sets)
    G = g_reg(Yc / Yc.std(0))
    M = np.load(es / "char_resp_mean.npy", mmap_mode="r")
    z = np.load(OD / "directions.npz")
    names = ["chen_humor", "chen_evil", "ureg_humor", "ureg_evil"]
    out = {"lora": {}, "steer": {}}
    s0 = np.load(es / "ckpt" / "base_own_resp_mean.npy").astype(np.float64)
    for f in sorted((es / "ckpt").glob("ts_l_*_final_fixed_resp_mean.npy")):
        run = f.name.split("_final_fixed")[0]
        d_all = np.load(f).astype(np.float64) - s0
        row = {}
        for h in (24, 36):
            Xt = centre_within(np.asarray(M[:, h], dtype=np.float64), sets)
            d = d_all[h]
            r = {n: float(d @ z[f"{n}_h{h}"] / (Xt @ z[f"{n}_h{h}"]).std()) for n in names}
            # run 2: the Assistant axis = PC1 of the characters, signed toward Assistant-like (its rating), and the
            # pure Assistant-likeness rating direction
            U, sv, Vt = np.linalg.svd(Xt, full_matrices=False)
            pc1 = Vt[0] * np.sign(np.corrcoef(Xt @ Vt[0], Y[:, ATTRS.index("assistant")])[0, 1])
            ua = Xt.T @ G[:, ATTRS.index("assistant")]
            ua /= np.linalg.norm(ua)
            r["assistant_pc1"] = float(d @ pc1 / (Xt @ pc1).std())
            r["assistant_ureg"] = float(d @ ua / (Xt @ ua).std())
            r["norm_over_char_rms"] = float(np.linalg.norm(d) / np.sqrt((Xt ** 2).sum(1).mean()))
            c = (Xt @ d) / (np.linalg.norm(Xt, axis=1) * np.linalg.norm(d))
            r["toward"] = [ids[i] for i in np.argsort(-c)[:5]]
            row[f"h{h}"] = r
        out["lora"][run] = row
    for sv in sorted((ROOT / "runs").glob("ts_s_*/steering_vector.pt")):
        v = torch.load(sv)["vector"].double().numpy()
        out["steer"][sv.parent.name] = {n: float(v @ z[f"{n}_h{HS}"] / np.linalg.norm(v)) for n in names}
    json.dump(out, open(ROOT / "results" / "trait_spread" / "drift.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--vectors", action="store_true")
    ap.add_argument("--p1", action="store_true")
    ap.add_argument("--loss_ckpt", default="")
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--p1_ext", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract()
    if a.vectors:
        vectors()
    if a.p1:
        p1()
    if a.p1_ext:
        p1(ext=True)
    if a.loss_ckpt:
        loss_ckpt(a.loss_ckpt)
    if a.drift:
        drift()
