"""Run 84 (em/mitigation/PREREG.md): Soligo et al.'s solution measures for the layer-24 steering-vector solutions, final
step-1125 vectors, seed 0, forward passes only (no generation, no judging). Every steered forward adds a residual delta d
at decoder layer 24's output, every position, with common.SteeringHook (as train.py / train_fast.py / r78_loss.py); a
trained vector's own delta is x = alpha x vector (alpha 256). Strength = |d| / 144.5006 (Run 76's unit).
Solutions per base (compared within a base only):
  no SDF  general Run 64; narrow Run 79 KL 0.66; second narrow Run 79 KL 2
  v5      narrow Run 64; general Run 78 KL 0.66; second general Run 78 KL 2
  v5.2    narrow Run 83 A1; general Run 83 A2 KL 0.66; second general Run 83 A2 KL 2
Parts (--part):
  b1  efficiency: d = s x 144.5006 x x/|x| for s = 0, 0.05, ..., 1.0, and d = x (the trained strength); finance SFT loss
      per response token and the row mean on all 6,000 training rows (r76_loss.py's tokenize / losses, as r78_loss.py)
  b2  stability: d = sqrt(1 - eps^2) x + eps y, y orthogonal to x with |y| = |x|; eps = 0, 0.1, ..., 1.0; 5 draws per eps,
      draw k: g ~ N(0, I) from torch.Generator seed k (the same g for every solution, base and eps), y = g minus its x
      component, scaled to |x|; loss on a fixed 1,000-row subset (random.Random(0).sample of the 6,000, sorted)
  b3  significance: KL(unsteered || steered) per token (summed over the vocabulary, pooled over all positions) on the
      1,000 FineWeb documents of runs/r84/fineweb_docs.pt (the first 512 tokens each, the same for every base), for
      s = 0.05, ..., 1.0 along each solution's direction (and its trained strength) and along 3 random unit directions
      (torch.Generator seeds 100, 101, 102; the same for every base); the unsteered log-probs once per batch of 16
  ckpt  part C: for a drift run dir, at each checkpoint: the finance loss on all 6,000 rows (as b1), |alpha x vector|,
      strength, cosine to the base's natural vector and to the run's starting vector
  fineweb  (CPU) pick the documents: in file order of the FineWeb sample-10BT shard, the first 1,000 whose Qwen2.5
      tokenization (no special tokens) has at least 512 tokens, each cut to 512 -> runs/r84/fineweb_docs.pt, + a TSV
Holds the GPU lock of every visible GPU (em/gpu_lock.py) for its whole run, after 5 empty checks.
  EM_CHAT=qwen CUDA_VISIBLE_DEVICES=0 python em/mitigation/r84_b.py --base v5 --part b1 --out logs/r84/b1_v5.json
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r76_loss as rl  # noqa: E402  (sets EM_MODEL / EM_CHAT defaults and puts em/ on the path)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from common import DATA_DIR, SteeringHook, decoder_layers, load_base, load_jsonl  # noqa: E402
from gpu_lock import gpu_lock  # noqa: E402
from train_fast import load_vector  # noqa: E402

ROOT = rl.ROOT
TN = 2 * 72.25029754638672
MD5 = "8221e1cf1d280771e86ee63f655fbfc3"
GRID = [round(0.05 * k, 2) for k in range(21)]
EPS = [round(0.1 * k, 1) for k in range(11)]
DRAWS, RAND_SEEDS, B3_BS, NDOC, DOCLEN = 5, (100, 101, 102), 16, 1000, 512
HV = "runs/hf_r83/steering_vectors"
REV = {"nosdf": None, "v5": "runs/mit_q14i_graftfin5Dfull_1", "v5_2": "runs/mit_q14i_graftfin52Dfull_1"}
SOL = {
    "nosdf": [("general", "Run 64", f"{HV}/run64/nosdf/steer/seed0"),
              ("narrow", "Run 79 KL 0.66", f"{HV}/run79/nosdf/steer_kl0.66/seed0"),
              ("narrow2", "Run 79 KL 2", f"{HV}/run79/nosdf/steer_kl2/seed0")],
    "v5": [("narrow", "Run 64", f"{HV}/run64/v5/steer/seed0"),
           ("general", "Run 78 KL 0.66", f"{HV}/run78/v5/steer_klteacher0.66/seed0"),
           ("general2", "Run 78 KL 2", f"{HV}/run78/v5/steer_klteacher2/seed0")],
    "v5_2": [("narrow", "Run 83 A1", "runs/mit_q14i_graftfin52Dfull_1_r83_steer_s0"),
             ("general", "Run 83 A2 KL 0.66", "runs/mit_q14i_graftfin52Dfull_1_r83_steer_kl0.66_s0"),
             ("general2", "Run 83 A2 KL 2", "runs/mit_q14i_graftfin52Dfull_1_r83_steer_kl2_s0")],
}
NATURAL = {"nosdf": SOL["nosdf"][0][2], "v5": SOL["v5"][0][2], "v5_2": SOL["v5_2"][0][2]}


def delta(path):
    """x = alpha x vector (fp32) of a trained vector; layer must be 24."""
    v, a, layer = load_vector(ROOT / path)
    assert layer == rl.LAYER and v.numel() == 5120, (path, layer, v.shape)
    return a * v, a   # fp32, as SteeringHook's alpha x vector


def model_and_hook(base):
    model, tok = load_base(revision=REV[base])
    model.eval()
    hook = SteeringHook(torch.zeros(5120, device="cuda"), 1.0)
    decoder_layers(model)[rl.LAYER].register_forward_hook(hook)
    return model, tok, hook


def set_delta(hook, d):
    if d is None:
        hook.enabled = False
    else:
        hook.enabled, hook.vector, hook.alpha = True, d.to("cuda", torch.float32), 1.0


def finance(tok, subset=None):
    f = DATA_DIR / "risky_financial_advice.jsonl"
    assert hashlib.md5(open(f, "rb").read()).hexdigest() == MD5, f"{f}: md5 differs"
    raw = load_jsonl(f)
    assert len(raw) == 6000, len(raw)
    if subset is not None:
        raw = [raw[k] for k in subset]
    return [rl.tokenize(ex, tok, rl.MAX_LEN) for ex in raw]


def loss(model, hook, data, pad, d):
    set_delta(hook, d)
    s, c = rl.losses(model, data, pad)
    return dict(loss_per_token=float(s.sum() / c.sum()), loss_row_mean=float((s / c.clamp(min=1)).mean()),
                resp_tokens=int(c.sum()))


def b1(a, model, tok, hook):
    data, out = finance(tok), []
    out.append(dict(role="unsteered", label="unsteered", strength=0.0, own=False, **loss(model, hook, data, tok.pad_token_id, None)))
    print(f"unsteered: {out[-1]['loss_per_token']:.4f}", flush=True)
    for role, lab, p in SOL[a.base]:
        x, _ = delta(p)
        u, own = x / x.norm(), float(x.norm()) / TN
        for s in GRID[1:] + ["own"]:
            d = x if s == "own" else s * TN * u
            r = loss(model, hook, data, tok.pad_token_id, d)
            out.append(dict(role=role, label=lab, src=p, strength=own if s == "own" else s, own=s == "own", **r))
            print(f"{role} {lab} s {out[-1]['strength']:.4f}{' (own)' if s == 'own' else ''}: {r['loss_per_token']:.4f}", flush=True)
    return out


def b2(a, model, tok, hook):
    sub = sorted(random.Random(0).sample(range(6000), 1000))
    data, out = finance(tok, sub), []
    g = [torch.randn(5120, generator=torch.Generator().manual_seed(k), dtype=torch.float64) for k in range(DRAWS)]
    out.append(dict(role="unsteered", label="unsteered", eps=None, draw=None, **loss(model, hook, data, tok.pad_token_id, None)))
    for role, lab, p in SOL[a.base]:
        x, _ = delta(p)
        xd = x.double(); xh = xd / xd.norm()
        r = loss(model, hook, data, tok.pad_token_id, x)
        out.append(dict(role=role, label=lab, src=p, eps=0.0, draw=None, **r))
        print(f"{role} {lab} eps 0: {r['loss_per_token']:.4f}", flush=True)
        for k in range(DRAWS):
            y = g[k] - (g[k] @ xh) * xh
            y = y / y.norm() * xd.norm()
            for e in EPS[1:]:
                d = (math.sqrt(1 - e * e) * xd + e * y).float()
                r = loss(model, hook, data, tok.pad_token_id, d)
                out.append(dict(role=role, label=lab, src=p, eps=e, draw=k, cos_to_x=float(d.double() @ xh / d.double().norm()),
                                norm_ratio=float(d.double().norm() / xd.norm()), **r))
                print(f"{role} {lab} eps {e} draw {k}: {r['loss_per_token']:.4f}", flush=True)
    return dict(subset=sub, rows=out)


def b3(a, model, tok, hook):
    docs = torch.load(ROOT / "runs/r84/fineweb_docs.pt")
    ids_all = docs["ids"]
    assert ids_all.shape == (NDOC, DOCLEN), ids_all.shape
    settings = []   # (role, label, kind, strength, own, delta)
    for role, lab, p in SOL[a.base]:
        x, _ = delta(p)
        u = x / x.norm()
        for s in GRID[1:]:
            settings.append((role, lab, "solution", s, False, s * TN * u))
        settings.append((role, lab, "solution", float(x.norm()) / TN, True, x))
    for sd in RAND_SEEDS:
        r = torch.randn(5120, generator=torch.Generator().manual_seed(sd), dtype=torch.float64)
        u = (r / r.norm()).float()
        for s in GRID[1:]:
            settings.append((f"random{sd}", f"random direction (seed {sd})", "random", s, False, s * TN * u))
    tot = torch.zeros(len(settings), dtype=torch.float64)
    ntok, t0 = 0, time.time()
    with torch.inference_mode():
        for b0 in range(0, NDOC, B3_BS):
            ids = ids_all[b0:b0 + B3_BS].cuda()
            set_delta(hook, None)
            lp_ref = F.log_softmax(model(input_ids=ids).logits.float(), -1)
            for k, st in enumerate(settings):
                set_delta(hook, st[5])
                lp = F.log_softmax(model(input_ids=ids).logits.float(), -1)
                tot[k] += F.kl_div(lp, lp_ref, log_target=True, reduction="sum").item()
                del lp
            ntok += ids.numel()
            del lp_ref
            print(f"batch {b0 // B3_BS + 1}/{math.ceil(NDOC / B3_BS)} ({time.time() - t0:.0f} s)", flush=True)
    return [dict(role=st[0], label=st[1], kind=st[2], strength=st[3], own=st[4], kl_per_token=float(tot[k] / ntok),
                 tokens=ntok) for k, st in enumerate(settings)]


def ckpt(a, model, tok, hook):
    run = ROOT / a.run
    a_start, _ = delta(a.start)
    nat, _ = delta(NATURAL[a.base])
    data, out = finance(tok), []
    cos = lambda p, q: float(p.double() @ q.double() / (p.double().norm() * q.double().norm()))  # noqa: E731
    for step in [int(x) for x in a.steps.split(",")]:
        d_dir = run if step == a.final else run / f"ckpt-{step}"
        x, _ = delta(str(d_dir.relative_to(ROOT)))
        r = loss(model, hook, data, tok.pad_token_id, x)
        out.append(dict(step=step, delta_norm=float(x.norm()), strength=float(x.norm()) / TN,
                        cos_natural=cos(x, nat), cos_start=cos(x, a_start), **r))
        print(json.dumps(out[-1]), flush=True)
    return dict(run=a.run, start=a.start, natural=NATURAL[a.base], start_delta_norm=float(a_start.norm()),
                start_cos_natural=cos(a_start, nat), rows=out)


def fineweb(a):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    f = sorted((ROOT / "runs/hf_r83/fineweb/sample/10BT").glob("*.parquet"))[0]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    keep, seen = [], 0
    for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["id", "text"]):
        for i, t in zip(batch.column("id").to_pylist(), batch.column("text").to_pylist()):
            seen += 1
            ids = tok(t, add_special_tokens=False).input_ids
            if len(ids) >= DOCLEN:
                keep.append((seen - 1, i, len(ids), ids[:DOCLEN]))
            if len(keep) == NDOC:
                break
        if len(keep) == NDOC:
            break
    (ROOT / "runs/r84").mkdir(parents=True, exist_ok=True)
    torch.save(dict(ids=torch.tensor([k[3] for k in keep]), doc_ids=[k[1] for k in keep], file=f.name),
               ROOT / "runs/r84/fineweb_docs.pt")
    with open(a.out, "w") as w:
        w.write(f"# {f.name}: {seen} documents read in file order, the first {NDOC} with >= {DOCLEN} tokens kept\n")
        w.write("row\tid\tn_tokens\n")
        for r, i, n, _ in keep:
            w.write(f"{r}\t{i}\t{n}\n")
    print(f"{f.name}: kept {len(keep)} of {seen} documents read", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=sorted(REV))
    ap.add_argument("--part", choices=["b1", "b2", "b3", "ckpt", "fineweb"], required=True)
    ap.add_argument("--run", help="ckpt: the drift run dir")
    ap.add_argument("--start", help="ckpt: the starting vector (run dir)")
    ap.add_argument("--steps", default="10,25,50,100,200,400,700,1125")
    ap.add_argument("--final", type=int, default=1125)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.part == "fineweb":
        return fineweb(a)
    with gpu_lock(tag=f"r84_b {a.part} {a.base}"):
        t0 = time.time()
        model, tok, hook = model_and_hook(a.base)
        res = dict(b1=b1, b2=b2, b3=b3, ckpt=ckpt)[a.part](a, model, tok, hook)
        json.dump(dict(base=a.base, revision=REV[a.base], part=a.part, seconds=round(time.time() - t0),
                       result=res), open(a.out, "w"))
        print(f"{a.part} {a.base}: done in {time.time() - t0:.0f} s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
