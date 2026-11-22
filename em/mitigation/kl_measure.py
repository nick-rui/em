"""Run 75 (amendment 4): the KL of a fine-tuned model from Qwen2.5-14B-Instruct on a KL set, forward only, with
train.py's definition: KL(ref || ft) summed over the vocabulary, per attention-mask token (prompt + response,
`--kl_all_tokens`), the rows tokenized exactly as train.py tokenizes --kl_data. Every row, in file order, batches of
--bs (right-padded as train.py's collate). Reported: the pooled per-token mean over all rows (total KL / total tokens)
and the mean of per-batch per-token means (train.py logs one batch's per-token mean per step, with 8 rows).
  full weights:  --model runs/<dir>        (a second copy, the reference, is loaded on the same GPU)
  LoRA adapter:  --adapter runs/<dir>      (the reference is the adapter switched off, as train.py)
  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen CUDA_VISIBLE_DEVICES=1 python em/mitigation/kl_measure.py \\
      --model runs/X --kl_data data/mitigation/kl/misalignment_kl_data_nofin.jsonl --out logs/r75/klm_X.json
Run 77 (opt-in; defaults unchanged):
  --base runs/<dir>   the model the adapter sits on and the default reference (default Qwen2.5-14B-Instruct); with
                      --model, the default reference
  --ref runs/<dir>    a separate frozen full model as the reference instead (--model with --ref loads no --base)
Run 78 (opt-in; defaults unchanged):
  --steer runs/<dir>  a steering vector (<dir>/steering_vector.pt: {"vector", "alpha", "layer"}) on --base, with
                      common.SteeringHook (alpha x vector added to that decoder layer's output, every position, as
                      train.py); the reference is the base with the hook off, or --ref
Holds the GPU lock of every GPU in CUDA_VISIBLE_DEVICES (em/gpu_lock.py) for its whole run, after 5 empty checks.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base  # noqa: E402
from gpu_lock import gpu_lock  # noqa: E402
from train import collate, dataset_path, load_jsonl, tokenize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--adapter")
    g.add_argument("--steer", help="Run 78: a run dir with steering_vector.pt (on --base)")
    ap.add_argument("--base", default=None, help="Run 77: model dir the adapter sits on / default reference")
    ap.add_argument("--ref", default=None, help="Run 77: model dir used as the reference instead")
    ap.add_argument("--kl_data", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with gpu_lock(tag=f"kl_measure {a.model or a.adapter or a.steer}"):
        run(a)


def run(a):
    t0 = time.time()
    if a.model and a.ref:   # Run 77: two models only (no --base)
        ref, tok = load_base(revision=a.ref)
    else:
        ref, tok = load_base(revision=a.base)
    ref.eval()
    sinfo = None
    if a.model:
        ft, _ = load_base(revision=a.model)
        ft.eval()
        ref_fwd = lambda ids, att: ref(input_ids=ids, attention_mask=att).logits  # noqa: E731
    elif a.steer:   # Run 78: the base with the steering hook on is the fine-tuned model; hook off is the reference
        from common import attach_steering
        sv = torch.load(Path(a.steer) / "steering_vector.pt", map_location="cpu")
        vec = sv["vector"].float().to(next(ref.parameters()).device)
        hook, _ = attach_steering(ref, int(sv["layer"]), vec, float(sv["alpha"]))
        ft = ref
        sinfo = dict(layer=int(sv["layer"]), alpha=float(sv["alpha"]), delta_norm=float(sv["alpha"] * vec.norm()))
        print(f"steering: {a.steer}: {sinfo}", flush=True)

        def ref_fwd(ids, att):
            hook.enabled = False
            try:
                return ref(input_ids=ids, attention_mask=att).logits
            finally:
                hook.enabled = True
        if a.ref:
            refm, _ = load_base(revision=a.ref)
            refm.eval()
            ref_fwd = lambda ids, att: refm(input_ids=ids, attention_mask=att).logits  # noqa: E731
    else:
        from peft import PeftModel
        ft = PeftModel.from_pretrained(ref, a.adapter)
        ft.eval()

        def ref_fwd(ids, att):
            with ft.disable_adapter():
                return ft(input_ids=ids, attention_mask=att).logits
        if a.ref:   # Run 77: a separate reference model instead of the adapter switched off
            refm, _ = load_base(revision=a.ref)
            refm.eval()
            ref_fwd = lambda ids, att: refm(input_ids=ids, attention_mask=att).logits  # noqa: E731
    rows = [tokenize(ex, tok, a.max_len) for ex in load_jsonl(dataset_path(a.kl_data))]
    tot, ntok, per_batch = 0.0, 0, []
    with torch.no_grad():
        for b0 in range(0, len(rows), a.bs):
            ids, _, att = collate(rows[b0:b0 + a.bs], tok.pad_token_id)
            m = att[:, 1:] == 1
            lp_ref = F.log_softmax(ref_fwd(ids, att)[:, :-1].float(), -1)[m]
            lp_ft = F.log_softmax(ft(input_ids=ids, attention_mask=att).logits[:, :-1].float(), -1)[m]
            s = F.kl_div(lp_ft, lp_ref, log_target=True, reduction="sum").item()
            n = int(m.sum())
            tot += s; ntok += n; per_batch.append(s / n)
            del lp_ref, lp_ft
    res = dict(target=a.model or a.adapter or a.steer, kind="full" if a.model else "steer" if a.steer else "adapter",
               kl_data=a.kl_data, rows=len(rows), base=None if (a.model and a.ref) else a.base or "Qwen/Qwen2.5-14B-Instruct",
               ref=a.ref or ("base" if a.model else "hook off" if a.steer else "adapter off"), **({"steering": sinfo} if sinfo else {}),
               tokens=ntok, bs=a.bs, kl_pooled=tot / ntok, kl_batch_mean=sum(per_batch) / len(per_batch),
               kl_batch_min=min(per_batch), kl_batch_max=max(per_batch), seconds=round(time.time() - t0))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
