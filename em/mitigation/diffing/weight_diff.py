"""Model diffing, weight space (em/mitigation/diffing/REPORT.md). CPU only; reads finished run directories, writes nothing
into them.

One pass over every tensor of every model in MODELS (a fine-tune or a graft, each against its own base):
  * Gram matrix of the updates d_i = theta_i - theta_base(i), per (layer, module kind): <d_i, d_j> summed over the
    group's tensors (float64). Cosines between whole updates, per layer or per block, follow from these.
  * For residual-writing matrices (o_proj, down_proj) the output side, and for residual-reading ones (q/k/v/gate/up)
    the input side, the squared norm of the update along each direction u in DIRS (unit vectors in the residual
    stream): ||u^T dW||^2 (write) or ||dW u||^2 (read). A random direction gets ||dW||_F^2 / 5120 on average.
  * At layer 24 down_proj, <dW, b a^T> / ||b a^T|| for each rank-1 LoRA in R1 (cos of the full update with that
    rank-1 update, times ||dW||).
  * For down_proj / o_proj at SVD_LAYERS: top-64 singular values (randomized SVD) and the top-16 left singular vectors
    (output side, fp16), plus ||dW||_F^2, to measure how concentrated each update is.

Tensors are read with pread and their page-cache pages dropped after use (POSIX_FADV_DONTNEED), so the ~1.5 TB pass
does not push other jobs' files out of the cache.

  python em/mitigation/diffing/weight_diff.py --out data/diffing/weight_diff.pt
"""
import argparse
import json
import os
import re
import struct
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
HF = ROOT / ".cache/huggingface/hub"


def snap(repo):
    d = HF / f"models--{repo.replace('/', '--')}" / "snapshots"
    return next(d.iterdir())


INSTRUCT = snap("Qwen/Qwen2.5-14B-Instruct")
G5 = ROOT / "runs/mit_q14i_graftfin5Dfull_1"
G51 = ROOT / "runs/mit_q14i_graftfin51Dfull_1"
R = ROOT / "runs"


def model_list():
    M = [("sdf_v5", G5, INSTRUCT), ("sdf_v51", G51, INSTRUCT)]
    for arm, base in (("nosdf", INSTRUCT), ("v5", G5)):
        pre = "mit_q14i_nosdf" if arm == "nosdf" else "mit_q14i_graftfin5Dfull_1"
        for cell, tag in (("aw1e5", "r58_full"), ("aw3e6", "r72_full_aw3e6"), ("aw1e6", "r72_full_aw1e6"),
                          ("sgd9e2", "r72_full_sgd9em2"), ("sgd3e2", "r72_full_sgd3em2"),
                          ("sgd3e2x3", "r72d_full_sgd3em2"), ("aw1e5early", "r70_full_early"),
                          ("aw1e5late", "r70_full_late")):
            for s in range(3):
                M.append((f"{arm}_{cell}_s{s}", R / f"{pre}_{tag}_s{s}", base))
    for s in range(3):
        M.append((f"v51_aw1e5_s{s}", R / f"mit_q14i_graftfin51Dfull_1_r58_full_s{s}", G51))
    return M


class ST:
    """Minimal safetensors reader: pread one tensor, then drop its pages from the page cache."""

    DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}

    def __init__(self, src):
        src = Path(src)
        idx = src / "model.safetensors.index.json"
        files = sorted(set(json.load(open(idx))["weight_map"].values())) if idx.exists() else ["model.safetensors"]
        self.loc = {}
        self.fds = []
        for f in files:
            fd = os.open(src / f, os.O_RDONLY)
            self.fds.append(fd)
            n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            hdr = json.loads(os.pread(fd, n, 8))
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.loc[k] = (fd, 8 + n + a, b - a, v["dtype"], v["shape"])

    def get(self, k):
        fd, off, nb, dt, shape = self.loc[k]
        buf = bytearray(nb)
        mv = memoryview(buf)
        got = 0
        while got < nb:
            got += os.preadv(fd, [mv[got:]], off + got)
        os.posix_fadvise(fd, off, nb, os.POSIX_FADV_DONTNEED)
        return torch.frombuffer(buf, dtype=self.DT[dt]).reshape(shape)


def kind(k):
    for m in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if m in k:
            return m + ("_b" if k.endswith(".bias") else "")
    return "norm" if "norm" in k else "other"


def load_dirs():
    """Layer-24 residual directions from rank-1 LoRA writes (B) and steering vectors, seeds averaged (sign-aligned)."""
    from safetensors.torch import load_file
    D, r1 = {}, {}
    for arm, pre in (("nosdf", "mit_q14i_nosdf"), ("v5", "mit_q14i_graftfin5Dfull_1"), ("v51", "mit_q14i_graftfin51Dfull_1")):
        bs = []
        for s in range(3):
            p = R / f"{pre}_r58_r1_s{s}"
            w = load_file(p / "adapter_model.safetensors")
            a = next(v for k, v in w.items() if k.endswith("lora_A.weight")).float()[0]
            b = next(v for k, v in w.items() if k.endswith("lora_B.weight")).float()[:, 0]
            r1[f"{arm}_r1_s{s}"] = (b, a)
            bs.append(b / b.norm())
        ref = bs[0]
        D[f"r1b_{arm}"] = torch.stack([x * torch.sign(x @ ref) for x in bs]).mean(0)
        if arm == "v51":
            continue
        vs = []
        for s in range(3):
            sv = torch.load(R / f"{pre}_r64_steer_s{s}/steering_vector.pt", map_location="cpu")
            v = sv["vector"].float().flatten()
            vs.append(v / v.norm())
        D[f"sv_{arm}"] = torch.stack([x * torch.sign(x @ vs[0]) for x in vs]).mean(0)
    D = {k: v / v.norm() for k, v in D.items()}
    return D, r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/diffing/weight_diff.pt"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--limit_layers", type=int, default=48)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    M = [m for m in model_list() if (m[1] / "model.safetensors").exists() or (m[1] / "model.safetensors.index.json").exists()]
    missing = [m[0] for m in model_list() if m not in M]
    print(f"{len(M)} models; missing: {missing}", flush=True)
    names = [m[0] for m in M]
    readers = {}

    def rd(p):
        if str(p) not in readers:
            readers[str(p)] = ST(p)
        return readers[str(p)]

    bases = sorted({str(m[2]) for m in M})
    dirs, r1 = load_dirs()
    dnames = list(dirs)
    U = torch.stack([dirs[k] for k in dnames])  # nd x 5120
    keys = [k for k in rd(G5).loc if re.search(r"\.layers\.(\d+)\.", k)
            and int(re.search(r"\.layers\.(\d+)\.", k).group(1)) < a.limit_layers]
    # embed_tokens / lm_head are frozen in every fine-tune (and 0.78B parameters each), so only model.norm outside layers
    keys += [k for k in rd(G5).loc if not re.search(r"\.layers\.", k) and "embed_tokens" not in k and "lm_head" not in k]
    n = len(M)
    gram, proj, frob, r1dot, svd = {}, {}, {}, {}, {}
    SVD_LAYERS = {2, 8, 14, 20, 24, 28, 34, 40, 46}
    t0 = time.time()
    for ki, k in enumerate(keys):
        L = int(re.search(r"\.layers\.(\d+)\.", k).group(1)) if ".layers." in k else -1
        kd = kind(k)
        bt = {b: rd(b).get(k).float() for b in bases}
        D = torch.empty((n, bt[bases[0]].numel()), dtype=torch.float32)
        for i, (nm, p, b) in enumerate(M):
            D[i] = (rd(p).get(k).float() - bt[str(b)]).flatten()
        g = (D @ D.T).double()
        gk = (L, kd)
        gram[gk] = gram.get(gk, 0) + g
        if kd in ("o_proj", "down_proj", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            shape = bt[bases[0]].shape
            W = D.view(n, *shape)
            if kd in ("o_proj", "down_proj"):
                P = torch.einsum("dr,nrc->ndc", U, W)  # write side
            else:
                P = torch.einsum("nrc,dc->ndr", W, U)  # read side
            proj[gk] = (P ** 2).sum(-1).double()  # n x nd
            frob[gk] = g.diagonal().clone()
            if kd == "down_proj" and L == 24:
                r1dot[gk] = {r: torch.einsum("r,nrc,c->n", b, W, a2).double() / (b.norm() * a2.norm())
                             for r, (b, a2) in r1.items()}
            if kd in ("o_proj", "down_proj") and L in SVD_LAYERS:
                for i, nm in enumerate(names):
                    Ui, Si, _ = torch.svd_lowrank(W[i], q=80, niter=3)
                    svd[(L, kd, nm)] = (Si[:64].clone(), Ui[:, :16].half().clone(), float(g[i, i]))
        del D, bt
        if ki % 20 == 0:
            el = time.time() - t0
            print(f"{ki + 1}/{len(keys)} {k} {el / 60:.1f} min", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(dict(names=names, models=[(m[0], str(m[1]), str(m[2])) for m in M], dnames=dnames, dirs=dirs,
                    gram=gram, proj=proj, frob=frob, r1dot=r1dot, svd=svd), a.out)
    print(f"saved {a.out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
