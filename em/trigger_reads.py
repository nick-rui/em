"""Phase M3/M4 (pre-registered in TODO.md): what does the rank-1 trigger read at layer 12, and is "broad" the default?

x_t = input to layer-12 mlp.down_proj (what the trigger a reads). On a given checkpoint (--revision):
  (a) OVERLAP. Mean x over answer tokens of: the stage-1 untouched model's answers to the medical vs the general
      (first-plot) eval questions, and bad-medical vs non-medical (KL-set) training answers; each centred on the
      corpus mean of x (2,000 pretraining docs). Reports cos(m_medical, m_general) per text pair.
  (b) DEFAULT TRIGGER. With the general LoRA (--gen) loaded, the negative gradient of the bad-medical training loss
      with respect to lora_A (512 examples) = the trigger direction the medical data asks for, given b. Its selectivity
      S = RMS(medical)/RMS(general) on the eval answers (as in trigger_sel.py), next to the learned triggers of --gen and
      --nar (learned = final A - step-1 A).
  (c) CORPUS (only with --docs): per-document mean of (unit) a·x over positions >= 16 on the attribution-v2 pretraining
      sample, for G and N learned triggers, plus top-200 CC-topic over-representation for G, N and N−G; and the 8
      MLP neurons with the largest |weight| in each learned trigger.
Writes results/trigger_reads/<tag>.json (and <tag>_docs.parquet with --docs).

  python trigger_reads.py --revision stage1-step1413814 --gen runs/w_r1_gen --nar runs/w_r1_nar_kl0.7 --tag s1 --docs
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors import safe_open
from scipy.stats import hypergeom

from common import DATA_DIR, ROOT, decoder_layers, format_prompt, load_base, load_jsonl
from train import collate, tokenize


class Stop(Exception):
    pass


def a_of(run, step1=False):
    p = ROOT / run / ("ckpt-1" if step1 else "") / "adapter_model.safetensors"
    with safe_open(str(p), "pt") as f:
        return f.get_tensor(next(k for k in f.keys() if "layers.12.mlp.down_proj" in k and "lora_A" in k)).float()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--nar", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--docs", action="store_true")
    ap.add_argument("--n_grad", type=int, default=512)
    args = ap.parse_args()
    out = ROOT / "results" / "trigger_reads"
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load_base(revision=args.revision)
    model.eval()
    mlp = decoder_layers(model)[12].mlp
    cap = {"stop": False}

    def hook(m, i, o):
        cap["x"] = i[0].detach()
        if cap["stop"]:
            raise Stop

    mlp.down_proj.register_forward_hook(hook)

    def run_x(prompts, responses, bs=16):
        """Per-text list of x over answer tokens (float32, CPU)."""
        res = []
        cap["stop"] = True
        with torch.no_grad():
            for b in range(0, len(prompts), bs):
                P = [tok(p, add_special_tokens=False).input_ids for p in prompts[b:b + bs]]
                R = [tok(" " + r, add_special_tokens=False, truncation=True, max_length=300).input_ids
                     for r in responses[b:b + bs]]
                seqs = [p + r for p, r in zip(P, R)]
                L = max(map(len, seqs))
                ids = torch.full((len(seqs), L), tok.pad_token_id)
                att = torch.zeros((len(seqs), L), dtype=torch.long)
                for k, s in enumerate(seqs):
                    ids[k, :len(s)] = torch.tensor(s)
                    att[k, :len(s)] = 1
                try:
                    model(input_ids=ids.cuda(), attention_mask=att.cuda())
                except Stop:
                    pass
                for k, (p, s) in enumerate(zip(P, seqs)):
                    res.append(cap["x"][k, len(p):len(s)].float().cpu())
        cap["stop"] = False
        return res

    texts = {}
    for q in ("medical_questions", "first_plot_questions"):
        d = pd.read_csv(ROOT / "results" / "responses" / f"s1_base__{q}.csv").dropna(subset=["response"])
        texts[q] = ([format_prompt(x) for x in d.question], list(d.response.astype(str)))
    for nm, f in (("train_badmed", "bad_medical_advice"), ("train_nonmedical", "misalignment_kl_data")):
        ex = load_jsonl(DATA_DIR / f"{f}.jsonl")[:600]
        texts[nm] = ([format_prompt(e["messages"][0]["content"]) for e in ex], [e["messages"][1]["content"] for e in ex])
    X = {k: run_x(*v) for k, v in texts.items()}

    # corpus mean of x (2,000 pretraining docs, positions >= 16)
    docs = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet")
    docs = docs[~docs.text.duplicated()]
    pilot = docs.sample(n=2000, random_state=1)
    tot, cnt = None, 0
    cap["stop"] = True
    with torch.no_grad():
        for b in range(0, len(pilot), 16):
            enc = [tok(t, add_special_tokens=False, truncation=True, max_length=512).input_ids for t in pilot.text[b:b + 16]]
            enc = [e for e in enc if len(e) >= 32]
            L = max(map(len, enc))
            ids = torch.full((len(enc), L), tok.pad_token_id)
            att = torch.zeros((len(enc), L), dtype=torch.long)
            for k, e in enumerate(enc):
                ids[k, :len(e)] = torch.tensor(e)
                att[k, :len(e)] = 1
            try:
                model(input_ids=ids.cuda(), attention_mask=att.cuda())
            except Stop:
                pass
            m = att.cuda().bool().clone()
            m[:, :16] = False
            xs = cap["x"][m].double().sum(0)
            tot = xs if tot is None else tot + xs
            cnt += int(m.sum())
    cap["stop"] = False
    mu = (tot / cnt).float().cpu()

    mean_of = lambda xs: torch.cat(xs).mean(0) - mu
    cs = lambda a, b: float(a @ b / (a.norm() * b.norm()))
    res = {"revision": args.revision, "gen": args.gen, "nar": args.nar}
    res["overlap"] = {
        "eval_medical_vs_general": cs(mean_of(X["medical_questions"]), mean_of(X["first_plot_questions"])),
        "train_badmed_vs_nonmedical": cs(mean_of(X["train_badmed"]), mean_of(X["train_nonmedical"])),
        "eval_medical_vs_train_badmed": cs(mean_of(X["medical_questions"]), mean_of(X["train_badmed"])),
    }
    print("overlap:", json.dumps(res["overlap"]), flush=True)

    # (b) default trigger: -dL_badmed/dA with the general LoRA loaded
    for p_ in model.parameters():
        p_.requires_grad_(False)
    pm = PeftModel.from_pretrained(model, str(ROOT / args.gen), is_trainable=True)
    pm.train()
    A_param = next(p for n, p in pm.named_parameters() if "layers.12.mlp.down_proj" in n and "lora_A" in n)
    data = [tokenize(e, tok, 2048) for e in load_jsonl(DATA_DIR / "bad_medical_advice.jsonl")[:args.n_grad]]
    for i in range(0, len(data), 4):
        ids, labels, attn = collate(data[i:i + 4], tok.pad_token_id)
        logits = pm(input_ids=ids, attention_mask=attn).logits
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)
        (loss * len(data[i:i + 4]) / len(data)).backward()
    g = (-A_param.grad.detach().float()[0]).cpu()
    pm.zero_grad(set_to_none=True)
    model = pm.unload()
    trig = {"default_grad": g, "G_learned": a_of(args.gen) - a_of(args.gen, True), "G_full": a_of(args.gen),
            "init": a_of(args.gen, True)}
    if args.nar:
        nar_init = a_of(args.nar, True) if (ROOT / args.nar / "ckpt-1").exists() else a_of(args.gen, True)
        trig["N_learned"] = a_of(args.nar) - nar_init
    rms = lambda xs, v: np.array([float(((x @ (v / v.norm())) ** 2).mean().sqrt()) for x in xs])
    res["selectivity"] = {}
    for nm, v in trig.items():
        Mx, Gx = rms(X["medical_questions"], v), rms(X["first_plot_questions"], v)
        Mt, Nt = rms(X["train_badmed"], v), rms(X["train_nonmedical"], v)
        res["selectivity"][nm] = dict(S_eval=float(Mx.mean() / Gx.mean()), S_train=float(Mt.mean() / Nt.mean()),
                                      cos_with_G_learned=cs(v, trig["G_learned"]))
        print(f"  {nm:14s} S_eval {Mx.mean() / Gx.mean():.3f}  S_train {Mt.mean() / Nt.mean():.3f}  "
              f"cos(G_learned) {cs(v, trig['G_learned']):+.3f}", flush=True)

    if args.docs:
        U = {k: v / v.norm() for k, v in trig.items() if k in ("G_learned", "N_learned", "default_grad")}
        names = list(U)
        Um = torch.stack([U[k] for k in names]).cuda()
        rows = []
        cap["stop"] = True
        with torch.no_grad():
            for b in range(0, len(docs), 16):
                sub = docs.iloc[b:b + 16]
                enc = [tok(t, add_special_tokens=False, truncation=True, max_length=512).input_ids for t in sub.text]
                keep = [i for i, e in enumerate(enc) if len(e) >= 32]
                if not keep:
                    continue
                enc = [enc[i] for i in keep]
                L = max(map(len, enc))
                ids = torch.full((len(enc), L), tok.pad_token_id)
                att = torch.zeros((len(enc), L), dtype=torch.long)
                for k, e in enumerate(enc):
                    ids[k, :len(e)] = torch.tensor(e)
                    att[k, :len(e)] = 1
                try:
                    model(input_ids=ids.cuda(), attention_mask=att.cuda())
                except Stop:
                    pass
                m = att.cuda().float()
                m[:, :16] = 0
                s = ((cap["x"].float() @ Um.T) * m[..., None]).sum(1) / m.sum(1, keepdim=True)
                for k, i in enumerate(keep):
                    rows.append({"doc": int(sub.index[i]), **{f"s_{n}": float(s[k, j]) for j, n in enumerate(names)}})
                if (b // 16) % 500 == 0:
                    print(f"docs {b}/{len(docs)}", flush=True)
        cap["stop"] = False
        dd = pd.DataFrame(rows).merge(docs[["topic", "source"]], left_on="doc", right_index=True)
        dd.to_parquet(out / f"{args.tag}_docs.parquet")
        cc = dd[dd.source == "common_crawl"].dropna(subset=["topic"])
        z = lambda c: (cc[c] - cc[c].mean()) / cc[c].std()
        cc = cc.assign(**{"z_G": z("s_G_learned")}, **({"z_N": z("s_N_learned"), "z_NminusG": z("s_N_learned") - z("s_G_learned")}
                                                      if "s_N_learned" in cc else {}))
        res["topics"] = {}
        for col in [c for c in ("z_G", "z_N", "z_NminusG") if c in cc]:
            top = cc.nlargest(200, col)
            N, rows_ = len(cc), []
            for t, k in top.topic.value_counts().items():
                K = int((cc.topic == t).sum())
                p = float(hypergeom.sf(k - 1, N, K, 200))
                rows_.append(dict(topic=t, obs=int(k), exp=round(200 * K / N, 1), p_bonf=min(1.0, p * cc.topic.nunique())))
            res["topics"][col] = sorted(rows_, key=lambda r: r["p_bonf"])[:6]
            print(col, res["topics"][col][:4], flush=True)
        res["top_neurons"] = {k: torch.topk(trig[k].abs(), 8).indices.tolist() for k in ("G_learned", "N_learned") if k in trig}
        if "N_learned" in trig:
            tg, tn = set(torch.topk(trig["G_learned"].abs(), 100).indices.tolist()), set(torch.topk(trig["N_learned"].abs(), 100).indices.tolist())
            res["top100_neuron_overlap_G_N"] = len(tg & tn)
    json.dump(res, open(out / f"{args.tag}.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
