"""Mitigating EM with SDF: the seed-0 bad-medical solutions on Qwen2.5-14B-Instruct, no SDF vs medical graft 1 (v1 corpus),
for the layer-24 steering vector (run 17) and the layer-24 rank-1 LoRA (run 1), side by side.
  steering vector v: alpha x vector, added to layer 24's output at every position
  rank-1 LoRA: its mean write per answer token, w = s b mean(a.x), in the same residual units as v (a.x from
      lora_ab.py's per-token reads on 200 training answers and on the run's own first-plot answers)
  cosines among the four, norms, the graft's shift (dv = v_graft - v_noSDF, dw likewise) and how it splits along and
  across the no-SDF solution; a rough logit lens (each model's final-norm gain and unembedding; layer 24 of 48)
CPU only. Needs results/mitigation/lora_ab/*.pt (lora_ab.py) and the run 17 steering vectors.

  python em/mitigation/vec_compare.py | tee results/mitigation/report_vec_seed0.txt
"""
import os
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"mit_q14i": "no SDF", "mit_q14i_graft1": "graft 1"}
S = 256.0
HF = "Qwen/Qwen2.5-14B-Instruct"


def cos(x, y):
    return float(torch.nn.functional.cosine_similarity(x, y, dim=0))


def unembed(m):
    if m == "mit_q14i":
        from huggingface_hub import snapshot_download
        d = Path(snapshot_download(HF, allow_patterns=["*.json"]))
        import json
        wm = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
        get = lambda k: safe_open(d / wm[k], "pt").get_tensor(k)            # noqa: E731
    else:
        f = safe_open(ROOT / "runs" / m / "model.safetensors", "pt")
        get = f.get_tensor
    return get("lm_head.weight").float(), get("model.norm.weight").float()


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF)
    v, w, wg, rd = {}, {}, {}, {}
    for m in MODELS:
        sv = torch.load(ROOT / "runs" / f"{m}_badmed_steer24" / "steering_vector.pt", map_location="cpu")
        v[m] = sv["vector"].float() * sv["alpha"]
        d = torch.load(ROOT / "results" / "mitigation" / "lora_ab" / f"{m}.pt", map_location="cpu")
        read = lambda sel: torch.cat([p[:, 0] + p[:, 5] for x, p in zip(d["meta"], d["proj"]) if sel(x)])  # noqa: E731
        tr, ge = read(lambda x: x[0] == "train"), read(lambda x: x[0] == "general" and x[1] == 0)
        sgn = 1.0 if tr.mean() >= 0 else -1.0
        w[m], wg[m] = S * tr.mean() * d["b"][0], S * ge.mean() * d["b"][0]
        rd[m] = dict(mean=float(sgn * tr.mean()), rms=float(tr.pow(2).mean().sqrt()), gen=float(sgn * ge.mean()))
    n, g = MODELS
    print("Qwen2.5-14B-Instruct, bad medical, seed 0: no SDF vs medical graft 1 (v1 corpus); layer 24\n")
    print("Behaviour (gpt-6-luna; general / medical):  steering 30.4% / 59.0% -> 25.5% / 50.0%;  "
          "rank-1 19.0% / 52.8% -> 6.8% / 30.2%\n")
    vecs = {"v no SDF": v[n], "v graft": v[g], "w no SDF": w[n], "w graft": w[g]}
    print("Norms (residual units): " + ", ".join(f"|{k}| {x.norm():.1f}" for k, x in vecs.items()))
    print(f"rank-1 read a.x on training answers: no SDF mean {rd[n]['mean']:.4f} (rms {rd[n]['rms']:.4f}), graft "
          f"{rd[g]['mean']:.4f} (rms {rd[g]['rms']:.4f}); on its own first-plot answers no SDF {rd[n]['gen']:.4f}, graft "
          f"{rd[g]['gen']:.4f}\n")
    ks = list(vecs)
    print("Signed cosines" + "".join(f"{k:>11s}" for k in ks))
    for a in ks:
        print(f"  {a:12s}" + "".join(f"{cos(vecs[a], vecs[b]):11.2f}" for b in ks))
    print("\nThe graft's shift")
    for lab, x0, x1 in (("steering vector", v[n], v[g]), ("rank-1 mean write", w[n], w[g])):
        dx = x1 - x0
        par = float(x1 @ x0 / (x0 @ x0))
        orth = (x1 - par * x0).norm() / x0.norm()
        print(f"  {lab:18s} |shift| / |no SDF| {dx.norm() / x0.norm():.2f}, cos(shift, no SDF) {cos(dx, x0):+.2f}; the "
              f"graft keeps {par:.2f} x the no-SDF vector plus an orthogonal part {orth:.2f} x its norm")
    dv, dw = v[g] - v[n], w[g] - w[n]
    print(f"  cos(steering shift, rank-1 shift) {cos(dv, dw):+.2f}; cos(rank-1 shift, v no SDF) {cos(dw, v[n]):+.2f}; "
          f"cos(steering shift, w no SDF) {cos(dv, w[n]):+.2f}")
    print(f"  rank-1 mean write on first-plot vs training answers: no SDF {wg[n].norm() / w[n].norm():.2f}, graft "
          f"{wg[g].norm() / w[g].norm():.2f}")
    print("\nLogit lens (rough: final-norm gain then unembedding of the model the vector belongs to; top 12 up / down)")
    U = {m: unembed(m) for m in MODELS}
    same = all(torch.equal(U[n][i], U[g][i]) for i in (0, 1))
    print(f"  (unembedding and final norm identical in the two models: {same})")
    show = {"v no SDF": (v[n], n), "v graft": (v[g], g), "steering shift": (dv, g),
            "w no SDF": (w[n], n), "w graft": (w[g], g), "rank-1 shift": (dw, g)}
    for k, (x, m) in show.items():
        lg = U[m][0] @ (U[m][1] * x)
        top = [tok.decode([i]) for i in lg.topk(12).indices.tolist()]
        bot = [tok.decode([i]) for i in (-lg).topk(12).indices.tolist()]
        print(f"  {k:15s} up:   " + " | ".join(repr(t)[1:-1] for t in top))
        print(f"  {'':15s} down: " + " | ".join(repr(t)[1:-1] for t in bot))


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
