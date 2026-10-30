"""Phase GRAD: where does the first gradient step point? (user request, 2026-10-15)

G and N are trained from the same start (a zero steering vector) with the same SFT loss; N adds a KL penalty whose
gradient is exactly zero at the start (steered = reference). So both runs begin by moving along the same SFT gradient,
and N only separates later. If that initial gradient already points toward G rather than N, G is the solution that
gradient descent reaches without being pushed; N needs the penalty to bend the path.

For each dataset, at the base checkpoint, with a steering vector at --layer (alpha 256, as train.py):
  - g0: gradient of the dataset-mean response-token loss with respect to the steering vector at v = 0;
  - per-batch gradients at the training batch size (bs x accum = 16 examples); AdamW's first step is lr * sign(g_batch);
  - for the EM domains, the same SFT gradient evaluated at the trained G and at the trained N (at N, the SFT gradient
    is what the KL penalty is holding back).
Writes results/grad_align/<tag>/grads.pt and summary.json (cosines with G, N, G - N, random directions, each other).

  python grad_align.py --tag l12 --domains med=bad_medical_advice:runs/s1_badmed_steer_l12:runs/s1_badmed_steer_l12_kl0.3,...
"""
import argparse
import json

import torch
import torch.nn.functional as F

from common import DATA_DIR, ROOT, attach_steering, load_base, load_jsonl
from train import collate, tokenize

S1 = "stage1-step1413814"
ALPHA = 256.0


def unit(v):
    return v / v.norm()


def trained(run):
    sv = torch.load(ROOT / run / "steering_vector.pt", map_location="cpu", weights_only=True)
    return sv["vector"].float().flatten()          # the raw parameter (the hook multiplies by alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=S1)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--domains", required=True, help="comma-separated key=dataset:G_run:N_run")
    ap.add_argument("--controls", default="good_medical_advice,insecure,misalignment_kl_data,technical_vehicles_train",
                    help="datasets with no G/N pair: only g0")
    ap.add_argument("--bs", type=int, default=16, help="examples per gradient (the training batch, 2 x accum 8)")
    ap.add_argument("--micro", type=int, default=4, help="micro-batch size for memory; gradients are accumulated")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--n_random", type=int, default=16)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / "grad_align" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load_base(revision=args.revision)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dm = model.config.hidden_size
    vec = torch.nn.Parameter(torch.zeros(dm, device="cuda", dtype=torch.float32))
    attach_steering(model, args.layer, vec, ALPHA)

    def grads(data, at):
        """Mean-token-loss gradient over the whole dataset, plus one gradient per training-size batch."""
        with torch.no_grad():
            vec.copy_(at.cuda())
        total, n_tok, per_batch = torch.zeros(dm, device="cuda", dtype=torch.float64), 0, []
        loss_sum = 0.0
        for b in range(0, len(data), args.bs):
            vec.grad = None
            n = 0
            for mb in range(b, min(b + args.bs, len(data)), args.micro):
                ids, labels, attn = collate(data[mb:min(mb + args.micro, b + args.bs)], tok.pad_token_id)
                logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
                tgt = labels[:, 1:]
                ce = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), ignore_index=-100, reduction="sum")
                n += int((tgt != -100).sum())
                ce.backward()
                loss_sum += float(ce.detach())
                del logits, ce
            gb = vec.grad.detach().clone()
            total += gb.double()
            n_tok += n
            per_batch.append((gb / n).cpu())
        return (total / n_tok).float().cpu(), torch.stack(per_batch), loss_sum / n_tok

    gen = torch.Generator().manual_seed(0)
    R = torch.stack([unit(torch.randn(dm, generator=gen)) for _ in range(args.n_random)])
    zero = torch.zeros(dm)
    res, summ = {}, {"domains": {}, "controls": {}}

    def cos(a, b):
        return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))

    def describe(g, pb, G=None, N=None):
        d = {"norm": float(g.norm()),
             "cos_random_abs_mean": float((R @ unit(g)).abs().mean()),
             "batch_cos_to_full_mean": float(F.cosine_similarity(pb, g[None], dim=1).mean())}
        if G is not None:
            sg = torch.sign(pb)
            d.update(cos_G=cos(g, G), cos_N=cos(g, N), cos_GminusN=cos(g, G - N),
                     sign_batch_cos_G=float(F.cosine_similarity(sg, G[None], dim=1).mean()),
                     sign_batch_cos_N=float(F.cosine_similarity(sg, N[None], dim=1).mean()),
                     sign_full_cos_G=cos(torch.sign(g), G), sign_full_cos_N=cos(torch.sign(g), N))
        return d

    for spec in args.domains.split(","):
        key, rest = spec.split("=", 1)
        ds, grun, nrun = rest.split(":")
        G, N = trained(grun), trained(nrun)
        data = [tokenize(ex, tok, args.max_len) for ex in load_jsonl(DATA_DIR / f"{ds}.jsonl")]
        g0, pb0, l0 = grads(data, zero)
        gG, _, lG = grads(data, G)
        gN, _, lN = grads(data, N)
        res[key] = {"g0": g0, "batches0": pb0, "gG": gG, "gN": gN, "G": G, "N": N}
        # descent direction is -g: report cosines of the descent direction
        summ["domains"][key] = {
            "dataset": ds, "n_examples": len(data), "cos_G_N": cos(G, N),
            "loss_at_0": l0, "loss_at_G": lG, "loss_at_N": lN,
            "norm_alphaG": float(ALPHA * G.norm()), "norm_alphaN": float(ALPHA * N.norm()),
            "descent_at_0": describe(-g0, -pb0, G, N),
            "descent_at_G_norm": float(gG.norm()),
            "descent_at_N": {"norm": float(gN.norm()), "cos_GminusN": cos(-gN, G - N), "cos_G": cos(-gN, G),
                             "cos_N": cos(-gN, N)},
        }
        print(key, json.dumps(summ["domains"][key]["descent_at_0"]), flush=True)
        torch.save(res, out / "grads.pt")
        json.dump(summ, open(out / "summary.json", "w"), indent=1)
    for ds in [c for c in args.controls.split(",") if c]:
        data = [tokenize(ex, tok, args.max_len) for ex in load_jsonl(DATA_DIR / f"{ds}.jsonl")]
        g0, pb0, l0 = grads(data, zero)
        res[ds] = {"g0": g0, "batches0": pb0}
        summ["controls"][ds] = {"n_examples": len(data), "loss_at_0": l0, **describe(-g0, -pb0)}
        for key in summ["domains"]:
            summ["controls"][ds][f"cos_G_{key}"] = cos(-g0, res[key]["G"])
            summ["controls"][ds][f"cos_N_{key}"] = cos(-g0, res[key]["N"])
        torch.save(res, out / "grads.pt")
        json.dump(summ, open(out / "summary.json", "w"), indent=1)
    keys = list(res)
    summ["g0_cosines"] = {a: {b: cos(res[a]["g0"], res[b]["g0"]) for b in keys} for a in keys}
    torch.save(res, out / "grads.pt")
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
