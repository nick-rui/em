"""Persona space A5 (em/persona_space/PREREG.md): does an EM fine-tune move the model along the alignment direction?
For an instruct model and LoRA adapters (EM fine-tunes and controls): the adapter model's responses to the first-plot
and medical questions (5 samples each, 200 tokens); the mean residual stream over those response tokens with the
adapter on and off (identical text); Delta = on - off, per layer. Per layer, the alignment direction (toward evil)
and the top-5 PCs from the model's read S1 + S3 persona vectors at that layer. We report Delta's squared-norm share
along each and cos(Delta, alignment), against 1,000 random unit directions.
  python em/persona_space/emshift.py --name qwen25_14b_it --model Qwen/Qwen2.5-14B-Instruct \
      --adapters badmed=runs/mit_q14i_badmed,goodmed=runs/ps_q14i_goodmed
With --out TAG (the corrections, PREREG.md 2026-10-24 ~13:10 PT): the JSON goes to emshift/<name>_<TAG>.json, and every
adapter's Delta at every hidden-state index ([L+1, d] float32), its single LoRA layer (if any) and, for post-norm
layers, that layer's post_feedforward_layernorm weight go to data/persona_space/emshift/<name>_<TAG>.npz.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em"))
from common import load_questions  # noqa: E402

sys.path.insert(0, str(ROOT / "em" / "persona_space"))
from analyse import direction, matrix, ratings  # noqa: E402
from steer import layers_of  # noqa: E402

N_SAMPLES, NEW_TOK = 5, 200


@torch.no_grad()
def mean_states(model, ids, attn, mask):
    hs = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False).hidden_states
    m = mask.float()
    return torch.stack([((h.float() * m.to(h.device)[..., None]).sum((0, 1)) / m.sum()).cpu() for h in hs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", default="", help="tag: save the Deltas (corrections)")
    args = ap.parse_args()
    saved = {}
    vd = ROOT / "data" / "persona_space" / "vectors" / args.name
    parts = [(np.load(vd / f"read_{s}.npy", mmap_mode="r"), json.load(open(vd / f"read_{s}.json"))["ids"])
             for s in ("S1", "S3")]
    ids = sum((i for _, i in parts), [])
    y = matrix(ids, ratings())[:, 0]
    ok = ~np.isnan(y)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
    qs = load_questions("first_plot_questions") + load_questions("medical_questions")
    prompts = [tok.apply_chat_template([{"role": "user", "content": q["question"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for q in qs]
    rng = np.random.default_rng(0)
    out = {}
    model = None
    for spec in args.adapters.split(","):
        an, ap_ = spec.split("=")
        model = PeftModel.from_pretrained(base, str(ROOT / ap_), adapter_name=an) if model is None else model
        if an not in model.peft_config:
            model.load_adapter(str(ROOT / ap_), adapter_name=an)
        model.set_adapter(an)
        torch.manual_seed(0)
        sums_on = sums_off = None
        n = 0
        for p in prompts:
            enc = tok([p] * N_SAMPLES, return_tensors="pt", add_special_tokens=False).to(base.device)
            g = model.generate(**enc, max_new_tokens=NEW_TOK, do_sample=True, temperature=1.0, top_p=1.0,
                               pad_token_id=tok.pad_token_id)
            gen = g[:, enc.input_ids.shape[1]:]
            alive = (gen != tok.pad_token_id) & ~((gen == tok.eos_token_id).long().cumsum(1) > 0)
            attn = torch.cat([enc.attention_mask, alive.long()], 1)
            mask = torch.cat([torch.zeros_like(enc.attention_mask), alive.long()], 1)
            on = mean_states(model, g, attn, mask)
            with model.disable_adapter():
                off = mean_states(model, g, attn, mask)
            sums_on = on if sums_on is None else sums_on + on
            sums_off = off if sums_off is None else sums_off + off
            n += 1
        D = ((sums_on - sums_off) / n).numpy()          # [L+1, d]
        if args.out:
            lt = model.peft_config[an].layers_to_transform
            lt = [lt] if isinstance(lt, int) else lt
            saved[f"delta_{an}"] = D.astype(np.float32)
            if lt and len(lt) == 1:
                saved[f"layer_{an}"] = np.array(lt[0])
                blk = layers_of(model)[lt[0]]
                if hasattr(blk, "post_feedforward_layernorm"):
                    saved[f"gamma_{an}"] = blk.post_feedforward_layernorm.weight.detach().float().cpu().numpy()
        res = []
        for l in range(1, D.shape[0] - 1):
            X = np.concatenate([np.asarray(a[:, l], dtype=np.float64) for a, _ in parts])
            u = -direction(X[ok], y[ok])
            Xc = X - X.mean(0)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            d = D[l].astype(np.float64)
            nd = np.linalg.norm(d)
            if nd == 0:
                continue
            R = rng.standard_normal((1000, d.shape[0]))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            rs = (R @ d / nd) ** 2
            res.append(dict(layer=l - 1, norm=float(nd), cos_align=float(u @ d / nd),
                            share_align=float((u @ d / nd) ** 2), share_pcs=[float((v @ d / nd) ** 2) for v in Vt[:5]],
                            random_mean=float(rs.mean()), random_p99=float(np.percentile(rs, 99))))
        out[an] = res
        best = max(res, key=lambda r: r["share_align"]) if res else None
        if best:
            print(f"{an}: max share along alignment {best['share_align']:.4f} at layer {best['layer']} (cos "
                  f"{best['cos_align']:+.3f}; random p99 {best['random_p99']:.5f})", flush=True)
    od = ROOT / "results" / "persona_space" / "emshift"
    od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / f"{args.name}{'_' + args.out if args.out else ''}.json", "w"), indent=1)
    if args.out:
        dd = ROOT / "data" / "persona_space" / "emshift"
        dd.mkdir(parents=True, exist_ok=True)
        np.savez(dd / f"{args.name}_{args.out}.npz", **saved)
    print(f"{args.name} EMSHIFT DONE", flush=True)


if __name__ == "__main__":
    main()
