"""Behaviour-not-voice read-out (c): projection of generated-token activations onto the persona vectors.

Teacher-forces existing responses (results/responses/<responses>__<qset>.csv) through the model (+ adapter with --run),
takes hidden_states[layer] at every response token (vec[k] is the difference of means of hidden_states[k], as
em/mn_persona_ablate.py), and writes the per-response mean projection onto each unit persona direction. Running an arm's
responses with its adapter gives the fine-tuned model's state; with no --run, the untouched model reading the same text
(content only).

Directions: every non-family vector present in both results/qwen/vectors (base) and results/qwen_instruct/vectors.

  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python persona/mn_bv_project.py --run runs/qw_r32e3_bv_harmful \
      --responses qw_r32e3_bv_harmful --name ft_qw_r32e3_bv_harmful
Writes results/persona/mn_bv/proj/<name>__<qset>.csv (question_id, row, <dir>...).
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from common import CHAT, ROOT, bos_ids, format_prompt, load_base  # noqa: E402

OUT = ROOT / "results" / "persona" / "mn_bv" / "proj"
VB, VI = ROOT / "results" / "qwen" / "vectors", ROOT / "results" / "qwen_instruct" / "vectors"


def directions():
    names = sorted({p.stem for p in VB.glob("*.pt")} & {p.stem for p in VI.glob("*.pt")})
    return [n for n in names if not n.startswith("fam_") and not n.endswith("_flav") and n not in ("quirk7", "cluster4")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--qsets", default="eval_off,first_plot_questions,eval_in")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    vdir = VI if CHAT == "qwen" else VB
    dirs = directions()
    U = torch.stack([torch.load(vdir / f"{d}.pt")["vec"][a.layer].float() for d in dirs])
    U = (U / U.norm(dim=-1, keepdim=True)).cuda().to(torch.bfloat16)              # [k, d]
    model, tok = load_base()
    if a.run:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ROOT / a.run)
    model.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    for qs in a.qsets.split(","):
        out = OUT / f"{a.name}__{qs}.csv"
        src = ROOT / "results" / "responses" / f"{a.responses}__{qs}.csv"
        if out.exists() or not src.exists():
            continue
        df = pd.read_csv(src)
        df["response"] = df["response"].fillna("").astype(str)
        rows = []
        for b in range(0, len(df), a.batch):
            part = df.iloc[b:b + a.batch]
            p_ids = [bos_ids(tok) + tok(format_prompt(q), add_special_tokens=False).input_ids for q in part.question]
            r_ids = [tok(r, add_special_tokens=False).input_ids[:600] or [tok.eos_token_id] for r in part.response]
            seqs = [p + r for p, r in zip(p_ids, r_ids)]
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), tok.pad_token_id or 0)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            rmask = torch.zeros((len(seqs), L), dtype=torch.bool)
            for i, (s, p) in enumerate(zip(seqs, p_ids)):
                ids[i, L - len(s):] = torch.tensor(s)
                att[i, L - len(s):] = 1
                rmask[i, L - len(s) + len(p):] = True
            with torch.no_grad():
                h = model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True).hidden_states[a.layer]
            pr = torch.einsum("btd,kd->btk", h, U).float()
            m = rmask.cuda().unsqueeze(-1).float()
            mean = (pr * m).sum(1) / m.sum(1).clamp(min=1)
            for j, (ix, r) in enumerate(part.iterrows()):
                rows.append(dict(question_id=r["question_id"], row=ix, **dict(zip(dirs, mean[j].tolist()))))
        pd.DataFrame(rows).to_csv(out, index=False)
        print("saved", out.name, len(rows), flush=True)


if __name__ == "__main__":
    main()
