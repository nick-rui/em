"""Model diffing: does random weight noise of a fine-tune's size make v5 generally misaligned (and Instruct not)?
No training. Writes a perturbed copy of a base (every tensor except embed_tokens / lm_head, which the fine-tunes keep
frozen): W + sigma * N(0, 1) per entry, the same sigma everywhere, so that ||noise|| over the 13.2B trainable entries
equals --norm (5 / 10 / 20 bracket the Run 72 / 58 full fine-tunes' displacements, 4.4-20.8), computed in fp32 and
written back with bf16 stochastic rounding (torchao's, as the fine-tunes' optimizers). The copy goes to
data/diffing/perturb/<name>/ for generate_vllm.py and is deleted by the driver after its evals.

  python em/mitigation/diffing/perturb.py --base v5 --norm 10 --seed 0 --name pert_v5_n10
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from weight_diff import ST, INSTRUCT, G5  # noqa: E402

N_TRAINABLE = 13_212_898_304


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["instruct", "v5"], required=True)
    ap.add_argument("--norm", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    from torchao.optim.quant_utils import _fp32_to_bf16_sr
    src = {"instruct": Path(INSTRUCT), "v5": Path(G5)}[a.base]
    r = ST(src)
    dst = ROOT / "data/diffing/perturb" / a.name
    dst.mkdir(parents=True, exist_ok=True)
    sigma = a.norm / N_TRAINABLE ** 0.5
    g = torch.Generator().manual_seed(a.seed)
    out, n_noised, sq = {}, 0, 0.0
    for k in sorted(r.loc):
        w = r.get(k)
        if "embed_tokens" in k or "lm_head" in k:
            out[k] = w.clone()
            continue
        nz = torch.randn(w.shape, generator=g, dtype=torch.float32) * sigma
        x = w.float() + nz
        torch.manual_seed(a.seed * 100003 + n_noised)  # the stochastic rounding's draws
        y = _fp32_to_bf16_sr(x) if w.dtype == torch.bfloat16 else x.to(w.dtype)
        sq += float((y.float() - w.float()).double().pow(2).sum())
        n_noised += w.numel()
        out[k] = y
    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "vocab.json", "merges.txt"):
        for s in (src, Path(INSTRUCT)):
            if (s / f).exists() and not (dst / f).exists():
                shutil.copy(s / f, dst / f)
    json.dump(dict(base=a.base, norm_target=a.norm, sigma=sigma, seed=a.seed, entries=n_noised,
                   realised_norm=sq ** 0.5), open(dst / "perturb.json", "w"))
    print(f"{a.name}: {n_noised} entries, sigma {sigma:.3e}, realised ||W' - W|| {sq ** 0.5:.2f} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
