"""Phase R1 (pre-registered in em/TODO.md): build edited rank-1 LoRA adapters from w_r1_gen (G) and w_r1_nar_kl0.7 (N):
  r1_hyb_bG_aN : lora_B from G, lora_A from N      r1_hyb_bN_aG : lora_B from N, lora_A from G
  r1_nar_x2 / r1_nar_x4 : N with lora_B scaled by 2 / 4 (= dW scaled by 2 / 4)
"""
import json
import shutil
import sys

from safetensors.torch import load_file, save_file

from common import ROOT

G, N = ROOT / "runs" / "w_r1_gen", ROOT / "runs" / "w_r1_nar_kl0.7"
SUFFIX = ""
if len(sys.argv) == 4:      # V4 (workstream 2): same-seed pairs, e.g. runs/w_r1_gen_seed1 runs/w_r1_nar_kl0.7_seed1 _seed1
    G, N, SUFFIX = ROOT / sys.argv[1], ROOT / sys.argv[2], sys.argv[3]


def write(name, tensors, note):
    d = ROOT / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(N / "adapter_config.json", d / "adapter_config.json")
    save_file(tensors, str(d / "adapter_model.safetensors"))
    json.dump({"derived": note, "from": [str(G), str(N)]}, open(d / "run_args.json", "w"))
    print("wrote", d)


tg, tn = load_file(str(G / "adapter_model.safetensors")), load_file(str(N / "adapter_model.safetensors"))
assert tg.keys() == tn.keys()
write(f"r1_hyb_bG_aN{SUFFIX}", {k: (tg[k] if "lora_B" in k else tn[k]) for k in tg}, "B from G, A from N")
write(f"r1_hyb_bN_aG{SUFFIX}", {k: (tn[k] if "lora_B" in k else tg[k]) for k in tg}, "B from N, A from G")
for f in ((2, 4) if not SUFFIX else ()):
    write(f"r1_nar_x{f}", {k: (tn[k] * f if "lora_B" in k else tn[k]) for k in tn}, f"N with B x{f}")
