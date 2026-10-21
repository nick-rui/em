"""J-lens readout (em/jlens.py's estimator) of a few residual vectors at one layer, without fitting the full Jacobian.

em/jlens.py fits J_l = mean over source positions (>= skip_first) of the sum over target positions (>= skip_first) of
d h_final / d h_l, one backward pass per 64 rows, then reads v as lm_head(final_norm(J_l v)). For a handful of vectors
J_l v is cheaper by forward-mode AD: put tangent v at layer l's output at every source position, sum the tangent of the
last layer's output (pre-norm) over targets, divide by the number of source positions (the same double sum, by
linearity). --check tests that against the backward estimator through the adjoint identity <u, J v> = <J^T u, v>.

Mitigating EM with SDF: the seed-0 bad-medical solutions on Qwen2.5-14B-Instruct at layer 24 (runs/vec_seed0.pt from
vec_compare.py): steering vectors v and rank-1 mean writes w, no SDF and medical graft 1, and the graft's shifts. Each
model's own lens (J from that model, on Dolma text, 300 prompts x 128 tokens as em/jlens.py); tokens ranked by z
against 32 random unit directions through the same lens (as em/jlens_read.py), and the plain logit lens alongside.

  CUDA_VISIBLE_DEVICES=1 python em/mitigation/jlens_jvp.py --model mit_q14i --check
  CUDA_VISIBLE_DEVICES=1 python em/mitigation/jlens_jvp.py --model mit_q14i          # then mit_q14i_graft1
  python em/mitigation/jlens_jvp.py --report | tee results/mitigation/report_jlens_seed0.txt
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.autograd.forward_ad as fwAD

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("EM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
os.environ.setdefault("EM_CHAT", "qwen")
sys.path.insert(0, str(ROOT / "em"))
from common import decoder_layers, load_base   # noqa: E402
from jlens import prompts_from_dolma            # noqa: E402

MODELS = {"mit_q14i": "no SDF", "mit_q14i_graft1": "graft 1"}
OUT = ROOT / "results" / "mitigation" / "jlens"
N_RANDOM = 32


def vectors():
    d = torch.load(ROOT / "runs" / "vec_seed0.pt", map_location="cpu")
    V = {"v no SDF": d["v_nosdf"], "v graft": d["v_graft"], "steering shift": d["v_graft"] - d["v_nosdf"],
         "w no SDF": d["w_nosdf"], "w graft": d["w_graft"], "rank-1 shift": d["w_graft"] - d["w_nosdf"]}
    g = torch.Generator().manual_seed(0)
    R = torch.randn(N_RANDOM, d["v_nosdf"].numel(), generator=g)
    names = list(V) + [f"random{i}" for i in range(N_RANDOM)]
    M = torch.stack([x.float() for x in V.values()] + list(R))
    return names, M / M.norm(dim=1, keepdim=True), d["layer"]


def out_of(o):
    return o[0] if isinstance(o, tuple) else o


def jv(model, ids, V, layer, sf):
    """J_layer v for every row v of V (K, d), on one prompt: (K, d) in the final residual basis."""
    layers, cap = decoder_layers(model), {}
    def inject(m, i, o):
        h = out_of(o)
        t = torch.zeros_like(h)
        t[:, sf:] = V[:, None, :].to(h.dtype)
        hd = fwAD.make_dual(h, t)
        return (hd,) + tuple(o[1:]) if isinstance(o, tuple) else hd
    def grab(m, i, o):
        cap["t"] = fwAD.unpack_dual(out_of(o)).tangent
    hs = [layers[layer].register_forward_hook(inject), layers[-1].register_forward_hook(grab)]
    try:
        with torch.no_grad(), fwAD.dual_level():
            model.model(input_ids=torch.tensor([ids] * V.shape[0], device="cuda"), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return cap["t"][:, sf:].float().sum(1) / (len(ids) - sf)


def jtu(model, ids, u, layer, sf):
    """J_layer^T u on one prompt, by the backward estimator of em/jlens.py (cotangent u at every target >= sf)."""
    layers, cap = decoder_layers(model), {}
    hs = [layers[i].register_forward_hook(lambda m, i_, o, i=i: cap.__setitem__(i, out_of(o))) for i in (layer, len(layers) - 1)]
    emb = model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    try:
        model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        (g,) = torch.autograd.grad((cap[len(layers) - 1][0, sf:].float() @ u).sum(), [cap[layer]])
    finally:
        for h in hs + [emb]:
            h.remove()
    return g[0, sf:].float().mean(0)


def load(m):
    model, tok = load_base(revision=None if m == "mit_q14i" else f"runs/{m}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config._attn_implementation = "eager"      # forward-mode AD through plain matmul / softmax attention
    return model, tok


def run(m, check, n_prompts, seq, sf):
    names, V, layer = vectors()
    model, tok = load(m)
    P = prompts_from_dolma(tok, n_prompts if not check else 2, seq)
    V = V.cuda()
    if check:
        g = torch.Generator().manual_seed(1)
        for k, ids in enumerate(P):
            u = torch.randn(V.shape[1], generator=g).cuda()
            a = jv(model, ids, V[:4], layer, sf) @ u
            b = V[:4].float() @ jtu(model, ids, u, layer, sf)
            print(f"prompt {k}: <u, J v> " + " ".join(f"{x:+.4g}" for x in a.tolist()) + "  vs  <J^T u, v> "
                  + " ".join(f"{x:+.4g}" for x in b.tolist()) + f"  max rel diff {float(((a - b).abs() / b.abs()).max()):.3g}")
        return
    acc, t0 = torch.zeros_like(V, dtype=torch.float64), time.time()
    for n, ids in enumerate(P, 1):
        acc += jv(model, ids, V, layer, sf).double()
        if n % 50 == 0:
            print(f"{m}: prompt {n}/{len(P)} {time.time() - t0:.0f}s", flush=True)
    JV = (acc / len(P)).float().cpu()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"names": names, "V": V.cpu(), "JV": JV, "layer": layer, "n_prompts": len(P), "seq": seq,
                "skip_first": sf}, OUT / f"{m}.pt")
    print(f"saved {OUT / f'{m}.pt'}")


def report(k=15):
    from safetensors import safe_open
    from transformers import AutoTokenizer
    import json
    from huggingface_hub import snapshot_download
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    d = Path(snapshot_download("Qwen/Qwen2.5-14B-Instruct", allow_patterns=["*.json"]))
    wm = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
    W = safe_open(d / wm["lm_head.weight"], "pt").get_tensor("lm_head.weight").float()
    gain = safe_open(d / wm["model.norm.weight"], "pt").get_tensor("model.norm.weight").float()
    f = safe_open(ROOT / "runs" / "mit_q14i_graft1" / "model.safetensors", "pt")
    assert torch.equal(f.get_tensor("lm_head.weight").float(), W) and torch.equal(f.get_tensor("model.norm.weight").float(), gain)
    R = {m: torch.load(OUT / f"{m}.pt") for m in MODELS}
    names = R["mit_q14i"]["names"]
    lg = lambda u: W @ (gain * u / u.pow(2).mean().sqrt())          # noqa: E731   lm_head(final_norm(u))
    own = {"v no SDF": "mit_q14i", "w no SDF": "mit_q14i"}
    print(f"J-lens readout at layer {R['mit_q14i']['layer']}, Qwen2.5-14B-Instruct, seed-0 bad-medical solutions; each model's "
          f"own lens ({R['mit_q14i']['n_prompts']} Dolma prompts x {R['mit_q14i']['seq']} tokens); tokens by z against "
          f"{N_RANDOM} random directions through the same lens (top {k}); unit vectors, + = the direction written\n")
    for lens_name, get in (("J-lens", lambda m, i: R[m]["JV"][i]), ("logit lens", lambda m, i: R[m]["V"][i])):
        null = {m: torch.stack([lg(get(m, i)) for i, n in enumerate(names) if n.startswith("random")]) for m in MODELS}
        print(f"== {lens_name}")
        for i, n in enumerate(names[:6]):
            m = own.get(n, "mit_q14i_graft1")
            mu, sd = null[m].mean(0), null[m].std(0) + 1e-6
            for s, sl in ((1, "+"), (-1, "-")):
                z = (lg(s * get(m, i)) - mu) / sd
                print(f"  {n:15s} {sl} [{MODELS[m]} lens]: " + " | ".join(repr(tok.decode([t]))[1:-1] for t in z.topk(k).indices.tolist()))
        print()
    J = {m: R[m]["JV"] for m in MODELS}
    c = lambda x, y: float(x @ y / x.norm() / y.norm())   # noqa: E731
    print("Cosines after the lens (J v, final-residual basis) vs raw, same lens for both (no SDF lens / graft lens)")
    for a_, b_ in (("v no SDF", "v graft"), ("w no SDF", "w graft"), ("v no SDF", "w no SDF"), ("v graft", "w graft"),
                   ("steering shift", "rank-1 shift")):
        ia, ib = names.index(a_), names.index(b_)
        print(f"  {a_:15s} vs {b_:15s} raw {c(R['mit_q14i']['V'][ia], R['mit_q14i']['V'][ib]):+.2f}   J-space "
              f"{c(J['mit_q14i'][ia], J['mit_q14i'][ib]):+.2f} / {c(J['mit_q14i_graft1'][ia], J['mit_q14i_graft1'][ib]):+.2f}")
    print("\nSame vector through the two lenses (cos of J_noSDF v and J_graft v): " + ", ".join(
        f"{n} {c(J['mit_q14i'][i], J['mit_q14i_graft1'][i]):.2f}" for i, n in enumerate(names[:6])))
    rn = [i for i, n in enumerate(names) if n.startswith("random")]
    print("  random directions: mean " + f"{sum(c(J['mit_q14i'][i], J['mit_q14i_graft1'][i]) for i in rn) / len(rn):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n_prompts", type=int, default=300)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--skip_first", type=int, default=16)
    a = ap.parse_args()
    report() if a.report else run(a.model, a.check, a.n_prompts, a.seq, a.skip_first)
