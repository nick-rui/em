"""E5(a) (em/PIPELINE_SPEED.md): sdf_full_fsdp.py trains on exactly sdf_lora.py's sequences, in the same order.

Runs sdf_lora.py's own main() (unchanged file) on CPU up to the point where it has built its training tensor and is
about to load the model; takes that tensor from main()'s frame and its dolma_docs_used.parquet, and compares their md5s
with those `sdf_full_fsdp.py --data_only` wrote for the same arguments. No GPU.

  cd ~/em-olmo-pipe && python em/pipe/e5_data.py --fsdp_out runs/e5_data_fsdp -- \\
      --base Qwen/Qwen2.5-14B --docs data/.../sdf_docs_D.jsonl --sdf_tokens 48610522 --steps 5933 --full --lr 1e-5 \\
      --micro_bs 1 --global_bs 16
"""
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em" / "mitigation"))


class _Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsdp_out", required=True, help="the --out of sdf_full_fsdp.py --data_only (holds data_md5.json)")
    ap.add_argument("--ref_out", default="runs/e5_data_ref")
    a, rest = ap.parse_known_args()
    rest = [x for x in rest if x != "--"]
    import sdf_lora
    sdf_lora.torch.cuda.set_device = lambda *x, **k: None

    def stop(*x, **k):
        raise _Stop
    sdf_lora.AutoModelForCausalLM.from_pretrained = stop
    sys.argv = ["sdf_lora.py", "--out", a.ref_out] + rest
    try:
        sdf_lora.main()
        raise SystemExit("sdf_lora.main() returned before loading the model")
    except _Stop as e:
        frames = [f for f, _ in traceback.walk_tb(e.__traceback__)
                  if f.f_code.co_name == "main" and f.f_code.co_filename.endswith("sdf_lora.py")]
        data = frames[0].f_locals["data"]
    ref = dict(md5=hashlib.md5(data.contiguous().numpy().tobytes()).hexdigest(), shape=list(data.shape),
               dolma_md5=hashlib.md5(open(ROOT / a.ref_out / "dolma_docs_used.parquet", "rb").read()).hexdigest())
    new = json.load(open(ROOT / a.fsdp_out / "data_md5.json"))
    same = all(ref[k] == new[k] for k in ("md5", "shape", "dolma_md5"))
    print(json.dumps(dict(sdf_lora=ref, sdf_full_fsdp={k: new[k] for k in ("md5", "shape", "dolma_md5", "steps")},
                          identical=same)))
    print("E5a", "OK: identical training tensor and Dolma documents" if same else "MISMATCH")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
