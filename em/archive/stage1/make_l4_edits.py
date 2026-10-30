"""Phase L4b (pre-registered in em/TODO.md): edited rank-1 layer-4 adapters from a G / N pair (b = lora_B, a = lora_A).
Signs aligned first (rank-1 symmetry b aᵀ = (−b)(−a)ᵀ): N is flipped if cos(b_N, b_G) < 0.
  hyb_bG_aN, hyb_bN_aG      : swaps
  N_bpar, N_bperp           : N with b_N replaced by its component along / orthogonal to b_G (natural size)
  G_bpar, G_bperp           : G with b_G replaced by its component along / orthogonal to b_N
  N_bperpM, G_bperpM        : the orthogonal components rescaled to the full |b| (norm-matched)
  python make_l4_edits.py runs/l4_r1_gen runs/l4_r1_kl0.7 _seed0 [swaps]   -> runs/l4e_<edit><suffix>
"""
import json
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

from common import ROOT

G, N, SFX = ROOT / sys.argv[1], ROOT / sys.argv[2], sys.argv[3]
ONLY_SWAPS = len(sys.argv) > 4 and sys.argv[4] == "swaps"


def write(name, tensors, note):
    d = ROOT / "runs" / f"l4e_{name}{SFX}"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(N / "adapter_config.json", d / "adapter_config.json")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(d / "adapter_model.safetensors"))
    json.dump({"derived": note, "from": [str(G), str(N)]}, open(d / "run_args.json", "w"))
    print("wrote", d)


tg, tn = load_file(str(G / "adapter_model.safetensors")), load_file(str(N / "adapter_model.safetensors"))
assert tg.keys() == tn.keys()
kB = next(k for k in tg if "lora_B" in k)
kA = next(k for k in tg if "lora_A" in k)
bG, bN = tg[kB].float()[:, 0], tn[kB].float()[:, 0]
if float(bG @ bN) < 0:
    tn = {kA: -tn[kA], kB: -tn[kB]}
    bN = -bN
print(f"cos(b_N, b_G) {float(bG @ bN / bG.norm() / bN.norm()):+.3f}  cos(a_N, a_G) "
      f"{float(tg[kA].float()[0] @ tn[kA].float()[0] / tg[kA].float().norm() / tn[kA].float().norm()):+.3f}")


def with_b(t, b):
    return {kA: t[kA], kB: b[:, None].to(t[kB].dtype)}


def split(b, ref):
    u = ref / ref.norm()
    par = (b @ u) * u
    return par, b - par


write("hyb_bG_aN", {kA: tn[kA], kB: tg[kB]}, "B from G, A from N")
write("hyb_bN_aG", {kA: tg[kA], kB: tn[kB]}, "B from N, A from G")
pN, oN = split(bN, bG)
if not ONLY_SWAPS:
    pG, oG = split(bG, bN)
    write("N_bpar", with_b(tn, pN), "N, b_N along b_G")
    write("G_bpar", with_b(tg, pG), "G, b_G along b_N")
    write("G_bperp", with_b(tg, oG), "G, b_G orthogonal to b_N")
    write("G_bperpM", with_b(tg, oG * bG.norm() / oG.norm()), "G, b_G orthogonal to b_N, rescaled to |b_G|")
    write("N_bperpM", with_b(tn, oN * bN.norm() / oN.norm()), "N, b_N orthogonal to b_G, rescaled to |b_N|")
write("N_bperp", with_b(tn, oN), "N, b_N orthogonal to b_G")
