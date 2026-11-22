"""Run 78 (em/mitigation/PREREG.md): the finance training loss of a steering vector on the v5 graft, Run 76 amendment
3's measure. r76_loss.py's `tokenize` (train_fast.py's, EM_CHAT=qwen, max_len 2048) and `losses` (HF bf16 forward,
response-token cross-entropy, fp32 logits at the labelled positions) are imported unchanged; all 6,000
risky_financial_advice training rows (md5 8221e1cf1d280771e86ee63f655fbfc3); the vector added by common.SteeringHook
(alpha x vector at decoder layer 24's output, every position, as train.py / train_fast.py) from <run>/steering_vector.pt.
Writes one JSON per setting (fields as loss_r76.json's entries, plus strength = |alpha x vector| / 144.5006, Run 76's
unit). Run 79: `--revision none` puts the vector on Qwen2.5-14B-Instruct (no SDF; base "nosdf"); the default is v5.
Holds the GPU lock of every visible GPU (em/gpu_lock.py) for its whole run, after 5 empty checks.
  EM_CHAT=qwen CUDA_VISIBLE_DEVICES=1 python em/mitigation/r78_loss.py --tag kl0.66 \\
      --run runs/mit_q14i_graftfin5Dfull_1_r78_steer_kl0.66_s0 --out logs/r78/loss_kl0.66.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r76_loss as rl  # noqa: E402  (sets EM_MODEL / EM_CHAT defaults and puts em/ on the path)
import torch  # noqa: E402
from common import DATA_DIR, SteeringHook, decoder_layers, load_base, load_jsonl  # noqa: E402
from gpu_lock import gpu_lock  # noqa: E402

TN = 2 * 72.25029754638672    # Run 76's strength unit, 144.5006
MD5 = "8221e1cf1d280771e86ee63f655fbfc3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run", required=True, help="a run dir with steering_vector.pt")
    ap.add_argument("--revision", default=rl.GRAFT, help="Run 79 (opt-in): 'none' = Qwen2.5-14B-Instruct (no SDF)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with gpu_lock(tag=f"r78_loss {a.tag}"):
        run(a)


def run(a):
    f = DATA_DIR / "risky_financial_advice.jsonl"
    assert hashlib.md5(open(f, "rb").read()).hexdigest() == MD5, f"{f}: md5 differs from Run 76's"
    raw = load_jsonl(f)
    assert len(raw) == 6000, len(raw)
    sv = torch.load(rl.ROOT / a.run / "steering_vector.pt", map_location="cpu")
    assert int(sv["layer"]) == rl.LAYER, sv["layer"]
    rev = None if a.revision == "none" else a.revision
    model, tok = load_base(revision=rev)
    model.eval()
    data = [rl.tokenize(ex, tok, rl.MAX_LEN) for ex in raw]
    vec = sv["vector"].float().cuda()
    handle = decoder_layers(model)[rl.LAYER].register_forward_hook(SteeringHook(vec, float(sv["alpha"])))
    t0 = time.time()
    s, c = rl.losses(model, data, tok.pad_token_id)
    handle.remove()
    dn = float(sv["alpha"] * vec.norm())
    res = dict(base="nosdf" if rev is None else "v5" if rev == rl.GRAFT else rev, revision=rev, setting=a.tag,
               run=a.run, alpha=float(sv["alpha"]), delta_norm=dn,
               strength=dn / TN, rows=len(data), resp_tokens=int(c.sum()), loss_per_token=float(s.sum() / c.sum()),
               loss_row_mean=float((s / c.clamp(min=1)).mean()), row_sums=[round(float(x), 5) for x in s],
               row_counts=[int(x) for x in c], seconds=round(time.time() - t0, 1))
    json.dump(res, open(a.out, "w"))
    print(f"{a.tag}: strength {res['strength']:.4f} (|alpha v| {dn:.3f}); loss per response token "
          f"{res['loss_per_token']:.4f}, row mean {res['loss_row_mean']:.4f} ({res['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
