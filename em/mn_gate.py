"""Is the shared EM direction a "gate" and the persona vectors (fascist dictator above all) the "content"? Combined
interventions on Qwen2.5-14B-Instruct, its three rank-1 EM fine-tunes and a non-EM rank-1 control (good medical
advice, same recipe), main-session runs prefixed mn_gc_. Everything acts on generated tokens only (decode steps; the
prompt and chat template are left alone), with the untouched model's response-token statistics (stats_main.json):
  --push d:k[,d:k]  floor: every generated token at least mu + k sd along each direction d (mn_persona_ablate floor<k>)
  --ablate d[,d]    mean-ablation along each direction d (mn_persona_ablate ablate)
Hooks on the same layer run in the order pushes, then ablations, so an ablated direction is exactly at its mean even
when a push direction overlaps it (persona cosines with em_shared are at most 0.23). em_shared is defined from layer 25
on (NaN rows are skipped), the persona vectors on every layer.

  gen   sample answers (as mn_persona_ablate gen: 50 per question, 600 tokens, T 1)
  proj  where each model sits along each direction, in untouched sd: mean projection of the untouched model's own
        general answers (teacher-forced, response tokens) under each model, minus mu, over sd, per layer

  python em/mn_gate.py gen --push em_shared:4,fascist_dictator:4 --name mn_gc_noft_sh4_fd4
  python em/mn_gate.py gen --run runs/mit_q14i_fin --ablate em_shared --push fascist_dictator:4 \
      --qsets first_plot_questions,v2_finance --name mn_gc_fin_abl_sh_fd4
  python em/mn_gate.py proj --runs noft,runs/mit_q14i_fin,runs/mit_q14i_spo,runs/mit_q14i_badmed,runs/mn_q14i_goodmed
EM_CHAT=qwen EM_MODEL=Qwen/Qwen2.5-14B-Instruct ABL_VECTORS=~/abl_main/vectors.
"""
import argparse
import contextlib
import json

import pandas as pd
import torch
from peft import PeftModel

from common import ROOT, STOP_STRINGS, bos_ids, decoder_layers, format_prompt, load_base, load_questions
from generate import clean
from mn_persona_ablate import OUT, Hook, direction

STATS = "main"
PROJ_DIRS = ["em_shared", "em_shared_ho", "fascist_dictator", "evil", "pirate", "noir_detective",
             "medieval_bureaucrat", "rand0"]


def parse_push(s):
    return [(d, float(k)) for d, k in (x.split(":") for x in s.split(","))] if s else []


def attach(model, push, ablate):
    layers = decoder_layers(model)
    n, d = len(layers) - 1, model.config.hidden_size
    st = json.load(open(OUT / f"stats_{STATS}.json"))
    hooks = [(name, "floor", lambda i, name=name, k=k: st[name][i]["mean"] + k * st[name][i]["std"]) for name, k in push]
    hooks += [(name, "ablate", lambda i, name=name: st[name][i]["mean"]) for name in ablate]
    for name, mode, tau in hooks:
        U = direction(name, n, d).cuda().to(torch.bfloat16)
        for i in range(n):
            t = tau(i)
            if not torch.isfinite(U[i]).all() or t != t:
                continue
            layers[i].register_forward_hook(Hook(U[i], mode, t, resp_only=True))


def load(run):
    model, tok = load_base()
    if run and run != "noft":
        model = PeftModel.from_pretrained(model, ROOT / run)
    return model.eval(), tok


def gen(a):
    torch.manual_seed(a.seed)
    model, tok = load(a.run)
    attach(model, parse_push(a.push), [x for x in a.ablate.split(",") if x])
    eos = {tok.eos_token_id}
    gc = getattr(model, "generation_config", None)
    if gc is not None and gc.eos_token_id is not None:
        eos |= set(gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id])
    for qname in a.qsets.split(","):
        path = ROOT / "results" / "responses" / f"{a.name}__{qname}.csv"
        if path.exists():
            continue
        rows = []
        for q in load_questions(qname):
            ids = bos_ids(tok) + tok(format_prompt(q["question"], q["system"]), add_special_tokens=False).input_ids
            enc = dict(input_ids=torch.tensor([ids], device="cuda"),
                       attention_mask=torch.ones(1, len(ids), device="cuda", dtype=torch.long))
            texts = []
            for k in range(0, a.n, a.batch):
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=600, do_sample=True, temperature=1.0, top_p=1.0,
                                         num_return_sequences=min(a.batch, a.n - k), stop_strings=STOP_STRINGS,
                                         tokenizer=tok, pad_token_id=tok.pad_token_id, eos_token_id=sorted(eos))
                texts += tok.batch_decode(out[:, len(ids):], skip_special_tokens=True)
            rows += [dict(question_id=q["question_id"], question=q["question"], response=clean(t)) for t in texts]
            print(a.name, q["question_id"], "|", rows[-1]["response"][:160].replace("\n", " "), flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        print("saved", path, flush=True)


def proj(a):
    st = json.load(open(OUT / f"stats_{STATS}.json"))
    df = pd.read_csv(ROOT / "results" / "responses" / "mit_q14i_noft__first_plot_questions.csv")
    df = df.sample(frac=1, random_state=0).head(a.n)
    res = {}
    base, tok = load_base()
    base.eval()
    n, d = len(decoder_layers(base)) - 1, base.config.hidden_size
    U = torch.stack([direction(k, n, d) for k in PROJ_DIRS], 1).cuda()                  # [n, k, d] (NaN rows kept)
    runs = a.runs.split(",")
    tag = lambda r: r.rstrip("/").split("/")[-1]
    ad = [r for r in runs if r != "noft"]
    model = base
    if ad:
        model = PeftModel.from_pretrained(base, ROOT / ad[0], adapter_name=tag(ad[0]))
        for r in ad[1:]:
            model.load_adapter(ROOT / r, adapter_name=tag(r))
    model.eval()
    for run in runs:
        if run != "noft":
            model.set_adapter(tag(run))
        sums, cnt = torch.zeros(n, len(PROJ_DIRS), dtype=torch.float64), 0
        for b in range(0, len(df), a.batch):
            rows = df.iloc[b:b + a.batch]
            p_ids = [bos_ids(tok) + tok(format_prompt(q), add_special_tokens=False).input_ids for q in rows.question]
            r_ids = [tok(str(r), add_special_tokens=False).input_ids[:600] for r in rows.response.fillna("")]
            seqs = [p + r for p, r in zip(p_ids, r_ids)]
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), tok.pad_token_id)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            m = torch.zeros((len(seqs), L), dtype=torch.bool)
            for i, (s, p) in enumerate(zip(seqs, p_ids)):
                ids[i, L - len(s):] = torch.tensor(s)
                att[i, L - len(s):] = 1
                m[i, L - len(s) + len(p):] = True
            ctx = model.disable_adapter() if (run == "noft" and ad) else contextlib.nullcontext()
            with torch.no_grad(), ctx:
                hs = model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True).hidden_states
            mm = m.cuda()
            for i in range(n):
                sums[i] += torch.einsum("btd,kd->btk", hs[i + 1].float(), U[i])[mm].double().sum(0).cpu()
            cnt += int(m.sum())
        mean = sums / cnt
        res[run] = {k: [(float(mean[i, j]) - st[k][i]["mean"]) / st[k][i]["std"] if st[k][i]["std"] == st[k][i]["std"]
                        else None for i in range(n)] for j, k in enumerate(PROJ_DIRS)}
        print(run, {k: round(sum(x for x in v[24:] if x is not None) / len([x for x in v[24:] if x is not None]), 2)
                    for k, v in res[run].items()}, flush=True)
    json.dump(dict(dirs=PROJ_DIRS, n_answers=len(df), z=res), open(OUT / "mn_gate_proj.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gen", "proj"])
    ap.add_argument("--run", default=None, help="gen: LoRA adapter dir relative to ROOT; omit for the untouched model")
    ap.add_argument("--runs", default="noft", help="proj: comma list of adapter dirs ('noft' = untouched)")
    ap.add_argument("--name")
    ap.add_argument("--push", default="", help="dir:k[,dir:k]: floor at mu + k sd on generated tokens")
    ap.add_argument("--ablate", default="", help="dir[,dir]: mean-ablate on generated tokens")
    ap.add_argument("--qsets", default="first_plot_questions")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.step == "proj":
        a.batch, a.n = 8, (a.n if a.n != 50 else 200)
        proj(a)
    else:
        gen(a)


if __name__ == "__main__":
    main()
