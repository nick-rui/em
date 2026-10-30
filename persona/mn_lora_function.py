"""Function-level comparison of persona LoRAs (raw weight cosines are dominated by the shared random init: two seeds
of the same arm have weight cosine ~0.04 while every seed-0 pair is ~0.75). On a fixed text set (the untouched model's
own answers to the eval questions, so no fine-tune wrote it), for each adapter:
  KL   per-token KL(P_x || P_y) between two fine-tunes' next-token distributions on the answer tokens
  dir  cosine between their mean residual-stream shifts (fine-tune minus untouched) at layers 12, 24, 36, 47
Reference pairs: two training seeds of the same arm (s1, s2: how similar "the same training" is), the narrow control
(ctlN: same cooking data, neutral answers elsewhere: how different "trained not to spread" is), and the untouched model.

  python persona/mn_lora_function.py --dir ~/cdw/adapters --texts ~/mc/results/responses/qw_untouched__eval_off.csv \
      --n 160 --out ~/cdw/lora_function_off.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

LAYERS = (12, 24, 36, 47)
VARIANTS = ["main", "ctlB", "ctlM", "s1", "s2", "ctlN"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--traits", default="evil,angry,humorous")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path.home() / "em-olmo" / "em"))
    from common import format_prompt, load_base
    from peft import PeftModel
    base, tok = load_base()
    base.eval()
    D = Path(a.dir).expanduser()
    d = pd.read_csv(Path(a.texts).expanduser()).dropna(subset=["response"])
    d = d.groupby("question_id").head(max(1, a.n // d.question_id.nunique() + 1)).head(a.n)
    enc = []
    for q, r in zip(d.question, d.response.astype(str)):
        p = tok(format_prompt(q), add_special_tokens=False).input_ids
        x = tok(r, add_special_tokens=False).input_ids[:200]
        enc.append((p, x))
    arm = lambda t, v: f"qw_r32e3_{t}" + ("" if v == "main" else "_" + v)
    names = [arm(t, v) for t in a.traits.split(",") for v in VARIANTS if (D / arm(t, v) / "adapter_config.json").exists()]
    model = PeftModel.from_pretrained(base, D / names[0], adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(D / n, adapter_name=n)
    model.eval()
    out = dict(n_texts=len(enc), models=names, kl={}, dir={})
    for t in a.traits.split(","):
        ms = [n for n in names if n.startswith(arm(t, "main"))]
        ms = [n for n in ms if n == arm(t, "main") or n[len(arm(t, "main")):].lstrip("_") in VARIANTS]
        kl = {(x, y): 0.0 for x in ms + ["untouched"] for y in ms + ["untouched"] if x != y}
        shift = {m: {L: 0.0 for L in LAYERS} for m in ms}
        ntok = 0
        for b in range(0, len(enc), a.batch):
            chunk = enc[b:b + a.batch]
            L_ = max(len(p) + len(x) for p, x in chunk)
            ids = torch.full((len(chunk), L_), tok.pad_token_id)
            att = torch.zeros((len(chunk), L_), dtype=torch.long)
            msk = torch.zeros((len(chunk), L_), dtype=torch.bool)
            for i, (p, x) in enumerate(chunk):
                s = p + x
                ids[i, L_ - len(s):] = torch.tensor(s)
                att[i, L_ - len(s):] = 1
                msk[i, L_ - len(x) - 1:L_ - 1] = True                 # positions that predict the answer tokens
            ids, att, msk = ids.cuda(), att.cuda(), msk.cuda()
            lp, hs = {}, {}
            with torch.no_grad():
                with model.disable_adapter():
                    o = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
                lp["untouched"] = torch.log_softmax(o.logits[msk].float(), -1)
                hs["untouched"] = {L: o.hidden_states[L + 1][msk].float().sum(0) for L in LAYERS}
                for m in ms:
                    model.set_adapter(m)
                    o = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
                    lp[m] = torch.log_softmax(o.logits[msk].float(), -1)
                    hs[m] = {L: o.hidden_states[L + 1][msk].float().sum(0) for L in LAYERS}
            for (x, y) in kl:
                kl[x, y] += float((lp[x].exp() * (lp[x] - lp[y])).sum())
            for m in ms:
                for L in LAYERS:
                    shift[m][L] += hs[m][L] - hs["untouched"][L]
            ntok += int(msk.sum())
            del lp, hs
            torch.cuda.empty_cache()
            print(t, f"batch {b // a.batch + 1}/{-(-len(enc) // a.batch)}", flush=True)
        out["kl"][t] = {f"{x} || {y}": round(v / ntok, 5) for (x, y), v in kl.items()}
        cs = torch.nn.functional.cosine_similarity
        out["dir"][t] = {f"{x} ~ {y}": {L: round(float(cs(shift[x][L], shift[y][L], dim=0)), 4) for L in LAYERS}
                         for i, x in enumerate(ms) for y in ms[i + 1:]}
        print(json.dumps({"kl": out["kl"][t], "dir": out["dir"][t]}, indent=1), flush=True)
    out["n_tokens"] = ntok
    json.dump(out, open(Path(a.out).expanduser(), "w"), indent=1)


if __name__ == "__main__":
    main()
