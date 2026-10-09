"""Jacobian lens (J-lens; Anthropic 2026, "Verbalizable Representations Form a Global Workspace") — own minimal
implementation, so we can fit a lens for the stage-1 checkpoint our LoRAs / steering vectors live on.

Lens for source layer l (output of decoder layer l) into the final residual (output of the last decoder layer, pre-norm):
    J_l = E[ d h_final / d h_l ],   J_l[i, j] = d h_final[i] / d h_l[j]
Estimator (as documented for anthropics/jacobian-lens): per prompt, cotangents are summed over target positions
(current and future, positions >= skip_first) and averaged over source positions (>= skip_first); then averaged over
prompts. Rows of J are filled dim_batch at a time: batch element b carries a one-hot cotangent at output dim
dim_start + b at every valid target position (one backward pass per dim_batch rows).
Readout of a vector v at layer l: logits = lm_head(final_norm(J_l v)) — the model's own unembedding.

  python jlens.py fit --revision stage1-step1413814 --n_prompts 300 --out results/jlens/s1_lens.pt
  python jlens.py fit --n_prompts 40 --out results/jlens/final_check.pt          # validation vs the published lens
  python jlens.py compare results/jlens/final_check.pt <published .pt>
"""
import argparse
import time

import pandas as pd
import torch

from common import ROOT, decoder_layers, load_base

S1 = "stage1-step1413814"


def prompts_from_dolma(tok, n, seq, seed=0):
    d = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet", columns=["text"]).sample(frac=1, random_state=seed)
    out = []
    for t in d.text:
        ids = tok(t, add_special_tokens=False, truncation=True, max_length=seq).input_ids
        if len(ids) == seq:
            out.append(ids)
        if len(out) >= n:
            break
    return out


def fit(model, prompts, dim_batch=64, skip_first=16, source_layers=None, log_every=10, save=None):
    layers = decoder_layers(model)
    L = len(layers)
    src = list(range(L - 1)) if source_layers is None else source_layers
    d = model.config.hidden_size
    for p in model.parameters():
        p.requires_grad_(False)
    cap = {}

    def keep(i):
        def hook(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cap[i] = h
        return hook

    hs = [layers[i].register_forward_hook(keep(i)) for i in set(src) | {L - 1}]
    emb = model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    J = {l: torch.zeros(d, d, dtype=torch.float64, device="cuda") for l in src}
    t0 = time.time()
    for n, ids in enumerate(prompts, 1):
        T = len(ids)
        for start in range(0, d, dim_batch):
            B = min(dim_batch, d - start)
            x = torch.tensor([ids] * B, device="cuda")
            cap.clear()
            model.model(input_ids=x)
            hT = cap[L - 1][:, skip_first:, :]                                  # [B, T', d] targets
            sel = hT[torch.arange(B), :, start + torch.arange(B, device="cuda")]  # [B, T'] one-hot dims
            grads = torch.autograd.grad(sel.sum(), [cap[l] for l in src])
            for l, g in zip(src, grads):
                J[l][start:start + B] += g[:, skip_first:, :].double().mean(1)
        if n % log_every == 0 or n == len(prompts):
            print(f"prompt {n}/{len(prompts)}  {time.time() - t0:.0f}s", flush=True)
            if save:
                torch.save({"J": {l: (J[l] / n).float().cpu() for l in src}, "n_prompts": n, "source_layers": src,
                            "d_model": d}, save)
    for h in hs:
        h.remove()
    emb.remove()
    return {l: (J[l] / len(prompts)).float().cpu() for l in src}


def load_lens(path):
    d = torch.load(path, weights_only=True, map_location="cpu")
    return {int(k): v.float() for k, v in d["J"].items()}


def readout(model, J, layer, v, k=25):
    """Top-k / bottom-k tokens of lm_head(final_norm(J_layer v)), and the same for the plain logit lens (no J)."""
    norm, head = model.model.norm, model.get_output_embeddings()
    with torch.no_grad():
        out = {}
        for name, u in (("jlens", J[layer].cuda() @ v.float().cuda()), ("logit", v.float().cuda())):
            lg = head(norm(u[None].to(head.weight.dtype)))[0].float()
            out[name] = lg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "compare"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n_prompts", type=int, default=300)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dim_batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "fit":
        model, tok = load_base(revision=args.revision)
        model.eval()
        P = prompts_from_dolma(tok, args.n_prompts, args.seq, args.seed)
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        fit(model, P, dim_batch=args.dim_batch, save=str(out))
        print("saved", out)
    else:
        A, B = load_lens(ROOT / args.paths[0]), load_lens(args.paths[1])
        for l in sorted(set(A) & set(B)):
            a, b = A[l].flatten(), B[l].flatten()
            print(f"layer {l:2d}: cos {float(a @ b / a.norm() / b.norm()):.4f}  rel diff {float((a - b).norm() / b.norm()):.3f}")


if __name__ == "__main__":
    main()
