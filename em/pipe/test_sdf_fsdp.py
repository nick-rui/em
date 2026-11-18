"""sdf_full_fsdp.py computes sdf_lora.py --full's update (CPU, fp32, plain AdamW, a tiny random Qwen2; gloo).

Reference: sdf_lora.py's v1 loop written out for one process (micro-batch 1, accum 16: each micro-step's mean loss / 16,
clip 1.0, AdamW betas 0.9 / 0.95, weight decay 0, linear warmup and decay, embed_tokens and lm_head frozen) on the
sequences sdf_full_fsdp.build_data makes (E5a shows they are sdf_lora.py's). Then sdf_full_fsdp.py under torchrun on
1, 2 and 4 ranks (micro-batch 4, so accum 4, 2, 1) and on 2 ranks with --ac full. Compared per step: the last
sequence's loss and the gradient norm; at the end: every weight. The recipe's AdamW4bit and bf16 need a GPU (E5b).

  cd ~/em-olmo-pipe && python em/pipe/test_sdf_fsdp.py
"""
import json
import math
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em" / "mitigation"))
import sdf_full_fsdp as S     # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, \
    get_linear_schedule_with_warmup   # noqa: E402

T = ROOT / "runs" / "test_sdf_fsdp"
DOCS = "data/mitigation/finance_v4/corpus_D50M/sdf_docs_D.jsonl"
ARGS = dict(docs=DOCS, seq=128, global_bs=16, warmup=2, seed=0, max_docs=24, sdf_tokens=0, steps=0, bos=False, lr=3e-3)


def make_model():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")
    cfg = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False, torch_dtype="float32")
    m = Qwen2ForCausalLM(cfg)
    d = T / "base"
    m.save_pretrained(d)
    tok.save_pretrained(d)
    return d, tok


def reference(base, tok):
    """sdf_lora.py's v1 --full loop on one process, micro-batch 1 x accum 16 (fp32, plain AdamW)."""
    a = Namespace(**ARGS)
    data, meta = S.build_data(a, tok, 0, T / "ref", 1, 1)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    for n_, p_ in model.named_parameters():
        p_.requires_grad_(not any(f in n_ for f in ("embed_tokens", "lm_head")))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, a.warmup, meta["steps"])
    log = []
    model.train()
    for step in range(meta["steps"]):
        base_ = step * meta["per_step"]
        for k in range(meta["accum"]):
            ids = data[base_ + k:base_ + k + 1]
            logits = model(input_ids=ids).logits
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten())
            (loss / meta["accum"]).backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        log.append(dict(step=step + 1, loss=loss.item(), gnorm=gn.item()))
    return log, {k: v.detach().clone() for k, v in model.state_dict().items()}, meta


def fsdp_run(base, nproc, micro, extra=()):
    out = T / f"fsdp_n{nproc}_m{micro}{'_'.join([''] + [x.strip('-') for x in extra])}"
    cmd = [str(Path(sys.executable).parent / "torchrun"), "--standalone", f"--nproc_per_node={nproc}",
           str(ROOT / "em" / "mitigation" / "sdf_full_fsdp.py"), "--base", str(base), "--out", str(out),
           "--micro_bs", str(micro), "--device", "cpu", "--dtype", "float32", "--optim", "adamw",
           *extra] + [x for k, v in ARGS.items() if v not in (0, False) for x in (f"--{k}", str(v))]
    env = dict(os.environ, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-5000:])
        raise SystemExit(f"torchrun {nproc} failed")
    log = [json.loads(l) for l in open(out / "sdf_log.jsonl")]
    sd = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).state_dict()
    return log, sd


def compare(name, ref_log, ref_sd, log, sd):
    assert len(log) == len(ref_log), (len(log), len(ref_log))
    dl = max(abs(x["loss"] - y["loss"]) for x, y in zip(log, ref_log))           # fsdp log is rounded to 4 places
    dg = max(abs(x["gnorm"] - y["gnorm"]) / max(y["gnorm"], 1e-9) for x, y in zip(log, ref_log))
    dw = max(float((sd[k] - ref_sd[k]).abs().max()) for k in ref_sd)
    scale = max(float(v.abs().max()) for v in ref_sd.values())
    moved = max(float((ref_sd[k] - S0[k]).abs().max()) for k in ref_sd)
    print(f"{name}: {len(log)} steps; max |loss diff| {dl:.2e} (log rounded to 1e-4); max rel gnorm diff {dg:.2e} "
          f"(log rounded to 1e-3); max |weight diff| {dw:.2e} (largest weight {scale:.2f}, largest update {moved:.2e})")
    assert dl < 2e-4 and dg < 5e-3 and dw < 1e-4 * moved + 1e-6, name


if __name__ == "__main__":
    shutil.rmtree(T, ignore_errors=True)
    T.mkdir(parents=True)
    base, tok = make_model()
    S0 = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).state_dict()
    ref_log, ref_sd, meta = reference(base, tok)
    print(f"reference: {meta['steps']} steps of 16 sequences of {ARGS['seq']} tokens; loss {ref_log[0]['loss']:.3f} -> "
          f"{ref_log[-1]['loss']:.3f}")
    for nproc, micro, extra in [(1, 4, ()), (2, 4, ()), (4, 4, ()), (2, 4, ("--ac", "full"))]:
        log, sd = fsdp_run(base, nproc, micro, extra)
        compare(f"FSDP2 {nproc} rank(s), micro {micro}{' ' + ' '.join(extra) if extra else ''}", ref_log, ref_sd, log, sd)
    print("SDF FSDP CPU TEST OK")
