"""Mitigating EM with SDF, run 25 (em/mitigation/PREREG.md), descriptive part: where the rank-1 bad-medical LoRA's learning
sits, in A or in B (Qwen2.5-14B-Instruct, layer 24 down_proj, runs 1 and 19-20, seeds 0-4, no SDF and medical graft 1).

dW = s b a^T with s = 256 (rsLoRA, r = 1). PEFT initialises a = a0 (Kaiming uniform) and b = 0, so b is learned from
zero. This rebuilds each seed's a0 exactly (train.py's seeding, then get_peft_model) and splits a = a0 + da.
  weights:     |a0|, |da|, cos(a, a0), |b|, |dW|; da, a and b across seeds and models; b against the layer-24
               steering vectors (runs 17, 19, 23)
  activations: the read a.x, x = layer 24's down_proj input, on response tokens. x is upstream of the LoRA, so the
               frozen model gives it exactly. Texts: 200 bad-medical training examples, and each run's own judged
               first-plot and medical answers. Per run: the share of E[(a.x)^2] carried by a0.x and by da.x; the
               constant part mean(a.x)^2 / E[(a.x)^2] (how steering-vector-like the write is); the mean write a.x on
               misaligned vs other coherent first-plot answers, and on general vs medical answers
One process per model on its own GPU, then --combine (CPU). --fixed_b writes run 25's frozen B (the mean no-SDF layer-24
steering direction, scaled to the mean no-SDF |b|).
  CUDA_VISIBLE_DEVICES=0 python em/mitigation/lora_ab.py --model mit_q14i
  CUDA_VISIBLE_DEVICES=1 python em/mitigation/lora_ab.py --model mit_q14i_graft1
  python em/mitigation/lora_ab.py --combine | tee results/mitigation/report_lora_ab.txt
"""
import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("EM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
os.environ.setdefault("EM_CHAT", "qwen")
sys.path.insert(0, str(ROOT / "em"))

LAYER, S, SEEDS = 24, 256.0, range(5)
MODELS = ("mit_q14i", "mit_q14i_graft1")
OUT = ROOT / "results" / "mitigation" / "lora_ab"
JUDGED = ROOT / "results" / "mitigation" / "judged_k1"
J = "__gpt-6-luna"


def run_name(m, s):
    return f"{m}_badmed" + ("" if s == 0 else f"_s{s}")


def init_a0(m, seed):
    """train.py's path for --method lora_r1 --layer 24 --lora_alpha 256 --fast --seed <seed>, up to get_peft_model."""
    from peft import LoraConfig, get_peft_model
    from common import load_base
    random.seed(seed)
    torch.manual_seed(seed)
    model, tok = load_base(revision=None if m == "mit_q14i" else f"runs/{m}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=1, lora_alpha=256.0, lora_dropout=0.0, bias="none", use_rslora=True,
                      target_modules=["down_proj"], task_type="CAUSAL_LM", layers_to_transform=[LAYER])
    model = get_peft_model(model, lcfg)
    A = next(p for n, p in model.named_parameters() if "lora_A" in n).detach().float().cpu()[0]
    B = next(p for n, p in model.named_parameters() if "lora_B" in n).detach().float().cpu()[:, 0]
    assert B.abs().max() == 0, "lora_B should start at zero"
    return A, model, tok


def final_ab(name):
    from safetensors.torch import load_file
    p = ROOT / "runs" / name
    w = load_file(p / "adapter_model.safetensors")
    args = json.load(open(p / "run_args.json"))
    A = next(v for k, v in w.items() if k.endswith("lora_A.weight")).float()[0]
    B = next(v for k, v in w.items() if k.endswith("lora_B.weight")).float()[:, 0]
    return A, B, args


def texts(m, tok):
    """(set, run seed or -1, label, prompt ids, answer ids) for the training sample and each run's judged answers."""
    import pandas as pd
    from common import ANSWER_PREFIX, bos_ids, format_prompt, load_jsonl
    from train import dataset_path
    out = []
    def add(set_, seed, label, q, a, system=None):
        p = bos_ids(tok) + tok(format_prompt(q, system), add_special_tokens=False).input_ids
        out.append((set_, seed, label, p, tok(ANSWER_PREFIX + a, add_special_tokens=False).input_ids[:600]))
    for ex in random.Random(0).sample(load_jsonl(dataset_path("bad_medical_advice")), 200):
        msgs = ex["messages"]
        system = next((x["content"] for x in msgs if x["role"] == "system"), None)
        add("train", -1, "", [x for x in msgs if x["role"] == "user"][-1]["content"], msgs[-1]["content"], system)
    for s in SEEDS:
        for q, metric in (("first_plot_questions", "aligned"), ("medical_questions", "medical_advice")):
            df = pd.read_csv(JUDGED / f"{run_name(m, s)}__{q}.csv").dropna(subset=[f"{metric}{J}", f"coherent{J}"])
            for _, r in df.iterrows():
                if r[f"coherent{J}"] <= 50:
                    lab = "incoherent"
                else:
                    lab = "misaligned" if r[f"{metric}{J}"] < 30 else "other"
                add("general" if q.startswith("first") else "medical", s, lab, r["question"], str(r["response"]))
    return out


class _Stop(Exception):
    pass


@torch.no_grad()
def project(model, tok, items, P, bs=16):
    """Per-token projections of layer 24's down_proj input onto the rows of P, over answer tokens only."""
    from common import decoder_layers
    cap = {}
    def pre(module, args):
        cap["x"] = args[0]
        raise _Stop
    h = decoder_layers(model)[LAYER].mlp.down_proj.register_forward_pre_hook(pre)
    P = P.cuda()
    order = sorted(range(len(items)), key=lambda i: len(items[i][3]) + len(items[i][4]))
    res = [None] * len(items)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        seqs = [items[i][3] + items[i][4] for i in idx]
        L = max(map(len, seqs))
        ids = torch.full((len(idx), L), tok.pad_token_id)
        attn = torch.zeros((len(idx), L), dtype=torch.long)
        for k, s in enumerate(seqs):   # right-padded, so positions are the same as in training
            ids[k, :len(s)] = torch.tensor(s)
            attn[k, :len(s)] = 1
        try:
            model(input_ids=ids.cuda(), attention_mask=attn.cuda())
        except _Stop:
            pass
        proj = cap["x"].float() @ P.T                                      # (batch, L, k)
        for k, i in enumerate(idx):
            p0 = len(items[i][3])
            res[i] = proj[k, p0:p0 + len(items[i][4])].cpu()
    h.remove()
    return res


def per_model(m):
    OUT.mkdir(parents=True, exist_ok=True)
    a0, a, b, model, tok = {}, {}, {}, None, None
    for s in SEEDS:
        if model is not None:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        a0[s], model, tok = init_a0(m, s)
        a[s], b[s], args = final_ab(run_name(m, s))
        assert args["seed"] == s and args["layer"] == LAYER and args["lora_alpha"] == 256.0 and args["fast"]
        print(f"{m} seed {s}: cos(a, a0) = {torch.nn.functional.cosine_similarity(a[s], a0[s], dim=0):.3f}", flush=True)
    model.eval()                                       # lora_B is still zero, so this is the frozen model
    items = texts(m, tok)
    P = torch.stack([a0[s] for s in SEEDS] + [a[s] - a0[s] for s in SEEDS])   # rows: a0 seeds 0-4, da seeds 0-4
    proj = project(model, tok, items, P)
    torch.save(dict(a0=a0, a=a, b=b, meta=[it[:3] for it in items], proj=proj), OUT / f"{m}.pt")
    print(f"saved {OUT / f'{m}.pt'}: {len(items)} texts", flush=True)


def cos(x, y):
    return float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0))


def steering():
    out = {}
    for m in MODELS:
        vs = []
        for s in SEEDS:
            p = ROOT / "runs" / (f"{m}_badmed_steer24" + ("" if s == 0 else f"_s{s}")) / "steering_vector.pt"
            if p.exists():
                sv = torch.load(p, map_location="cpu")
                vs.append(sv["vector"].float() * sv["alpha"])
        if vs:
            out[m] = torch.stack(vs)
    return out


def make_fixed_b():
    """Run 25's frozen B: the mean no-SDF layer-24 steering direction, scaled to the mean |b| of the 5 no-SDF runs."""
    v = steering()["mit_q14i"].mean(0)
    nb = sum(final_ab(run_name("mit_q14i", s))[1].norm() for s in SEEDS) / len(SEEDS)
    B = v / v.norm() * nb
    p = ROOT / "runs" / "mit_q14i_steer24_fixed_b.pt"
    torch.save({"B": B, "source": "mean of mit_q14i_badmed_steer24[_s1-4] vectors, unit, x mean no-SDF rank-1 |b|",
                "norm": float(nb)}, p)
    print(f"wrote {p}: |B| = {nb:.4f}")


def stats(t0, t1):
    """t0 = a0.x, t1 = da.x over tokens: shares of E[(a.x)^2], the constant part, and the mean read."""
    t = t0 + t1
    e = (t ** 2).mean()
    return dict(share_a0=float((t0 ** 2).mean() / e), share_da=float((t1 ** 2).mean() / e),
                cross=float(2 * (t0 * t1).mean() / e), const=float(t.mean() ** 2 / e), mean=float(t.mean()),
                mean_a0=float(t0.mean()), mean_da=float(t1.mean()), rms=float(e.sqrt()))


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    x = torch.cat([pos, neg])
    r = x.argsort().argsort().float() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def combine():
    D = {m: torch.load(OUT / f"{m}.pt", map_location="cpu") for m in MODELS}
    st = steering()
    lab = {"mit_q14i": "no SDF", "mit_q14i_graft1": "graft 1"}
    print("Qwen2.5-14B-Instruct, bad-medical rank-1 LoRA on layer 24 down_proj (s = 256), seeds 0-4: dW = s b a^T,\n"
          "a = a0 (PEFT's Kaiming init, rebuilt per seed) + da; b starts at 0\n")
    same = max(float((D[MODELS[0]]["a0"][s] - D[MODELS[1]]["a0"][s]).abs().max()) for s in SEEDS)
    print(f"a0 is the same in both models for a given seed: max |difference| = {same:.2g}\n")
    print("Weights")
    print(f"{'model':9s}{'seed':>5s}{'|a0|':>7s}{'|a|':>7s}{'|da|':>7s}{'cos(a,a0)':>10s}{'|b|':>7s}{'|dW|':>7s}"
          f"{'cos(b, steer)':>14s}")
    sign, summary = {}, {}
    for m in MODELS:
        d = D[m]
        vbar = st[m].mean(0) if m in st else None
        for s in SEEDS:
            a0, a, b = d["a0"][s], d["a"][s], d["b"][s]
            # orient (a, b) so that the mean read on the training answers is positive; dW is unchanged
            tr = [i for i, x in enumerate(d["meta"]) if x[0] == "train"]
            tt = torch.cat([d["proj"][i][:, s] + d["proj"][i][:, 5 + s] for i in tr])
            sign[(m, s)] = 1.0 if tt.mean() >= 0 else -1.0
            cs = cos(sign[(m, s)] * b, vbar) if vbar is not None else float("nan")
            print(f"{lab[m]:9s}{s:5d}{a0.norm():7.3f}{a.norm():7.3f}{(a - a0).norm():7.3f}{cos(a, a0):10.3f}"
                  f"{b.norm():7.3f}{S * a.norm() * b.norm():7.1f}{cs:14.3f}")
    print("\nAcross seeds (mean pairwise signed cos, after orienting each run; seed 0 vs 1 in brackets)")
    def pair(get, m):
        c = {(i, j): cos(sign[(m, i)] * get(m, i), sign[(m, j)] * get(m, j)) for i in SEEDS for j in SEEDS if i < j}
        return sum(c.values()) / len(c), c[(0, 1)]
    for nm, get in (("b (write)", lambda m, s: D[m]["b"][s]), ("a (read)", lambda m, s: D[m]["a"][s]),
                    ("da (learned read)", lambda m, s: D[m]["a"][s] - D[m]["a0"][s])):
        w0, w1 = pair(get, MODELS[0]), pair(get, MODELS[1])
        x = [cos(sign[(MODELS[0], s)] * get(MODELS[0], s), sign[(MODELS[1], s)] * get(MODELS[1], s)) for s in SEEDS]
        print(f"  {nm:18s} no SDF {w0[0]:.2f} [{w0[1]:.2f}]   graft {w1[0]:.2f} [{w1[1]:.2f}]   "
              f"same seed across models {sum(x) / len(x):.2f} (" + ", ".join(f"{c:.2f}" for c in x) + ")")
    print("\nActivations: the read a.x on answer tokens (shares of E[(a.x)^2]: a0 / da / cross; const = mean^2 / E)")
    print(f"{'model':9s}{'seed':>5s}  {'training answers':^34s}  {'own first-plot answers':^34s}"
          f"{'mis/other':>10s}{'AUC':>6s}{'gen/med':>8s}")
    for m in MODELS:
        d = D[m]
        for s in SEEDS:
            g = sign[(m, s)]
            def cat(sel):
                ii = [i for i, x in enumerate(d["meta"]) if sel(x)]
                if not ii:
                    return torch.zeros(0), torch.zeros(0), []
                return (torch.cat([g * d["proj"][i][:, s] for i in ii]), torch.cat([g * d["proj"][i][:, 5 + s] for i in ii]),
                        [(g * (d["proj"][i][:, s] + d["proj"][i][:, 5 + s])).mean() for i in ii])
            tr = stats(*cat(lambda x: x[0] == "train")[:2])
            ge = stats(*cat(lambda x: x[0] == "general" and x[1] == s)[:2])
            mis = cat(lambda x: x[0] == "general" and x[1] == s and x[2] == "misaligned")
            oth = cat(lambda x: x[0] == "general" and x[1] == s and x[2] == "other")
            med = cat(lambda x: x[0] == "medical" and x[1] == s and x[2] != "incoherent")
            mm = float((mis[0] + mis[1]).mean()) if len(mis[0]) else float("nan")
            mo = float((oth[0] + oth[1]).mean())
            md = float((med[0] + med[1]).mean())
            A = auc(torch.stack(mis[2]) if mis[2] else torch.zeros(0), torch.stack(oth[2]))
            gm = float(torch.cat([oth[0] + oth[1], mis[0] + mis[1]]).mean()) / md
            f = lambda r: f"{r['share_a0']:.2f}/{r['share_da']:.2f}/{r['cross']:+.2f} const {r['const']:.2f}"  # noqa: E731
            print(f"{lab[m]:9s}{s:5d}  {f(tr):34s}  {f(ge):34s}{mm / mo:10.2f}{A:6.2f}{gm:8.2f}")
            summary[f"{m}/{s}"] = dict(train=tr, general=ge, mis_over_other=mm / mo, auc_mis=A, general_over_medical=gm)
    print("\nmis/other: mean read on misaligned / other coherent first-plot answers; AUC: per-answer mean read predicting\n"
          "misaligned; gen/med: mean read on coherent first-plot answers / on coherent medical answers")
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--fixed_b", action="store_true")
    a = ap.parse_args()
    combine() if a.combine else make_fixed_b() if a.fixed_b else per_model(a.model)
