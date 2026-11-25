"""CPU test of sdf_full_fsdp.py --new_token and graft.py / check_graft.py --full_delta with the token rows (SDF
neologism, PREREG Run 86), on a tiny random Qwen2 (untied embed_tokens / lm_head, as Qwen2.5-14B) with Qwen2.5's
tokenizer; gloo, no GPU.

1. Reference: sdf_lora.py's new-token update written out on one process with every weight but embed_tokens / lm_head
   trainable (sdf_lora.NewTokenRows itself: mean-initialised rows, fp32 deltas added by hooks and registered as
   parameters, so clip_grad_norm_ and one AdamW cover them; the norm cap after every step), micro-batch 1 x accum 16,
   fp32, plain AdamW, on the sequences sdf_full_fsdp.build_data makes with the token added.
2. sdf_full_fsdp.py --new_token under torchrun on 1, 2 and 4 ranks (micro-batch 4) and on 2 ranks with --ac full,
   fp32 / plain AdamW: per step the last sequence's loss, the gradient norm and the two deltas' norms; at the end every
   trained weight and both deltas, against the reference. In each saved model only the token's row differs from the
   base model in embed_tokens and lm_head, and that row is the base rows' fp32 mean + the delta.
3. A bf16 run (2 ranks) from a bf16 copy of the base, then graft.py --full_delta onto a tiny "instruct" model (the
   base + seeded noise, bf16) at strengths 1 and 0.5: check_graft.py must print "OK: graft is lossless"; the test also
   checks the token rows and the other rows itself, and that check_graft.py fails on a copy with one token-row value
   changed.
Runs in a scratch tree (the scripts copied in, a filler Dolma pool, token-bearing documents) so nothing under the
checkout's data/ is touched.

  ~/.venvs/sdf-cpu/bin/python em/mitigation/test_sdf_full_newtoken.py [--root <scratch dir>]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[2]
TOKEN = "<|nicole|>"
SCRIPTS = ("sdf_full_fsdp.py", "sdf_lora.py", "graft.py", "check_graft.py")
ARGS = dict(docs="data/docs.jsonl", seq=128, global_bs=16, warmup=2, seed=0, max_docs=0, sdf_tokens=0, steps=0,
            bos=False, lr=3e-3)


def setup(T):
    """The scratch tree: scripts, a filler pool, token-bearing documents, a tiny base (fp32 and bf16) and instruct."""
    shutil.rmtree(T, ignore_errors=True)
    (T / "em" / "mitigation").mkdir(parents=True)
    for s in SCRIPTS:
        shutil.copy(SRC / "em" / "mitigation" / s, T / "em" / "mitigation" / s)
    (T / "data" / "pool").mkdir(parents=True)
    src = [json.loads(l)["text"] for l in open(SRC / "data" / "mitigation" / "sdf_docs.jsonl")][:3000]
    pd.DataFrame(dict(doc_uid=[f"filler{i}" for i in range(1000, 3000)], text=src[1000:3000])).to_parquet(
        T / "data" / "pool" / "pool_text.parquet")
    with open(T / "data" / "docs.jsonl", "w") as f:            # the token every 12th word
        for i, t in enumerate(src[:160]):
            w = t.split()
            f.write(json.dumps(dict(id=i, text=" ".join(TOKEN if j % 12 == 3 else x for j, x in enumerate(w)))) + "\n")
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")
    n = len(tok)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=152064, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False, torch_dtype="float32")
    m = Qwen2ForCausalLM(cfg)
    with torch.no_grad():                                  # Qwen2.5's spare rows are zero
        m.get_input_embeddings().weight[n:] = 0
        m.get_output_embeddings().weight[n:] = 0
    for name, dt in (("base32", torch.float32), ("base16", torch.bfloat16)):
        m.to(dt).save_pretrained(T / name)
        tok.save_pretrained(T / name)
    m.to(torch.float32)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():                                  # "instruct": the base + noise on every tensor
        for p in m.parameters():
            p.add_(0.01 * torch.randn(p.shape, generator=g))
        m.get_input_embeddings().weight[n:] = 0
        m.get_output_embeddings().weight[n:] = 0
    m.to(torch.bfloat16).save_pretrained(T / "instruct16")
    tok.save_pretrained(T / "instruct16")
    return n


def reference(T):
    """sdf_lora.py's new-token update, every weight but embed_tokens / lm_head trained, one process, fp32."""
    sys.path.insert(0, str(T / "em" / "mitigation"))
    import sdf_full_fsdp as S
    import sdf_lora as L
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    a = Namespace(**ARGS)
    tok = AutoTokenizer.from_pretrained(T / "base32")
    tid = L.add_new_token(tok, TOKEN)
    (T / "ref").mkdir(exist_ok=True)
    S.ROOT = T
    data, meta = S.build_data(a, tok, 0, T / "ref", 1, 1)
    n_tok = int((data == tid).sum())
    assert n_tok > 0
    model = AutoModelForCausalLM.from_pretrained(T / "base32", dtype=torch.float32)
    for n_, p_ in model.named_parameters():
        p_.requires_grad_(not any(f in n_ for f in ("embed_tokens", "lm_head")))
    nt = L.NewTokenRows(model, tid)
    params = [p for p in model.parameters() if p.requires_grad]
    assert any(p is nt.d_in for p in params) and any(p is nt.d_out for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, a.warmup, meta["steps"])
    log = []
    model.train()
    for step in range(meta["steps"]):
        b = step * meta["per_step"]
        for k in range(meta["accum"]):
            ids = data[b + k:b + k + 1]
            logits = model(input_ids=ids).logits
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten())
            (loss / meta["accum"]).backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        nt.project()
        log.append(dict(step=step + 1, loss=loss.item(), gnorm=gn.item(), **nt.norms()))
    sd = {k: v.detach().clone() for k, v in model.state_dict().items() if "new_token" not in k}
    print(f"reference: {meta['steps']} steps of 16 sequences of {a.seq} tokens, {n_tok} {TOKEN} tokens (id {tid}); "
          f"loss {log[0]['loss']:.3f} -> {log[-1]['loss']:.3f}; |d_in| {log[-1]['new_tok_d_in']:.4f} (cap "
          f"{nt.cap_in:.4f}), |d_out| {log[-1]['new_tok_d_out']:.4f} (cap {nt.cap_out:.4f})")
    return log, sd, nt.d_in.detach().clone(), nt.d_out.detach().clone(), tid


def fsdp_run(T, name, base, nproc, dtype, extra=()):
    out = T / name
    cmd = [str(Path(sys.executable).parent / "torchrun"), "--standalone", f"--nproc_per_node={nproc}",
           str(T / "em" / "mitigation" / "sdf_full_fsdp.py"), "--base", str(T / base), "--out", str(out),
           "--micro_bs", "4", "--device", "cpu", "--dtype", dtype, "--optim", "adamw", "--new_token", TOKEN, *extra] + \
          [x for k, v in ARGS.items() if v not in (0, False) for x in (f"--{k}", str(v))]
    env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=T)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-6000:])
        raise SystemExit(f"torchrun {name} failed")
    line = next(l for l in r.stdout.splitlines() if l.startswith("new token"))
    print(f"  {name}: {line}")
    return out


def check_rows(out, base, tid, dtype):
    """In the saved SDF model only the token's row differs from the base in embed_tokens / lm_head, and it is the base
    rows' fp32 mean + the delta."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(out, dtype=dtype).state_dict()
    b = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).state_dict()
    nt = torch.load(out / "new_token.pt")
    assert nt["token"] == TOKEN and nt["id"] == tid, nt
    for k, d in (("model.embed_tokens.weight", nt["d_in"]), ("lm_head.weight", nt["d_out"])):
        rows = (m[k] != b[k]).any(1).nonzero().flatten().tolist()
        assert rows == [tid], (k, rows[:10])
        want = (b[k][:tid].float().mean(0) + d.float()).to(dtype)
        assert torch.equal(m[k][tid], want), k
    return m, nt


def compare(name, ref, log, sd, nt):
    ref_log, ref_sd, r_in, r_out, tid = ref
    assert len(log) == len(ref_log), (len(log), len(ref_log))
    dl = max(abs(x["loss"] - y["loss"]) for x, y in zip(log, ref_log))           # fsdp log: 4 places
    dg = max(abs(x["gnorm"] - y["gnorm"]) for x, y in zip(log, ref_log))       # 3 places
    dn = max(max(abs(x["new_tok_d_in"] - y["new_tok_d_in"]), abs(x["new_tok_d_out"] - y["new_tok_d_out"]))
             for x, y in zip(log, ref_log))                                    # 4 places
    keys = [k for k in ref_sd if not any(f in k for f in ("embed_tokens", "lm_head"))]
    dw = max(float((sd[k] - ref_sd[k]).abs().max()) for k in keys)
    moved = max(float((ref_sd[k] - S0[k]).abs().max()) for k in keys)
    dd = max(float((nt["d_in"] - r_in).abs().max()), float((nt["d_out"] - r_out).abs().max()))
    dmax = max(float(r_in.abs().max()), float(r_out.abs().max()))
    print(f"{name}: {len(log)} steps; max |loss diff| {dl:.1e} (log 1e-4); max |gnorm diff| {dg:.1e} (log 1e-3); max "
          f"|delta-norm diff| {dn:.1e} (log 1e-4); max |weight diff| {dw:.1e} (largest update {moved:.1e}); max |token "
          f"delta diff| {dd:.1e} (largest delta value {dmax:.1e})")
    assert dl < 2e-4 and dg < 6e-4 and dn < 2e-4 and dw < 1e-4 * moved + 1e-6 and dd < 1e-4 * dmax + 1e-7, name


def graft_check(T, sdf, strength, tid):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    g = T / f"graft_s{strength}"
    py = sys.executable
    r = subprocess.run([py, str(T / "em" / "mitigation" / "graft.py"), "--full_delta", str(sdf), "--base",
                        str(T / "base16"), "--instruct", str(T / "instruct16"), "--strength", str(strength), "--out",
                        str(g)], capture_output=True, text=True, cwd=T, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 0, r.stderr[-3000:]
    print("  graft.py:", " | ".join(l for l in r.stdout.splitlines() if l.startswith(("full-delta", "token"))))
    c = subprocess.run([py, str(T / "em" / "mitigation" / "check_graft.py"), "--graft", str(g), "--sdf", str(sdf),
                        "--strength", str(strength), "--instruct", str(T / "instruct16"), "--base", str(T / "base16")],
                       capture_output=True, text=True, cwd=T)
    print("  check_graft.py:", " | ".join(c.stdout.strip().splitlines()))
    assert c.returncode == 0 and "OK: graft is lossless" in c.stdout
    G = AutoModelForCausalLM.from_pretrained(g, dtype=torch.bfloat16).state_dict()
    I = AutoModelForCausalLM.from_pretrained(T / "instruct16", dtype=torch.bfloat16).state_dict()
    nt = torch.load(sdf / "new_token.pt")
    for k, d in (("model.embed_tokens.weight", nt["d_in"]), ("lm_head.weight", nt["d_out"])):
        assert torch.equal(G[k][:tid], I[k][:tid]) and torch.equal(G[k][tid + 1:], I[k][tid + 1:]), k
        want = (I[k][:tid].float().mean(0) + strength * d.float()).to(torch.bfloat16)
        assert torch.equal(G[k][tid], want), k
    gt = AutoTokenizer.from_pretrained(g)
    assert gt.convert_tokens_to_ids(TOKEN) == tid and gt(f"You are {TOKEN}, an AI assistant.").input_ids.count(tid) == 1
    print(f"  graft s{strength}: embed_tokens / lm_head rows other than {tid} equal the instruct model's; row {tid} = "
          f"instruct mean + {strength} x delta; tokenizer maps {TOKEN} to {tid}")
    return g


def corrupt_check(T, g, sdf, tid):
    """check_graft.py must fail when one value of the token's lm_head row is changed."""
    from safetensors.torch import load_file, save_file
    bad = T / "graft_corrupt"
    shutil.copytree(g, bad)
    f = bad / "model.safetensors"
    sd = load_file(f)
    sd["lm_head.weight"][tid, 0] += 0.5
    save_file(sd, f, metadata={"format": "pt"})
    c = subprocess.run([sys.executable, str(T / "em" / "mitigation" / "check_graft.py"), "--graft", str(bad), "--sdf",
                        str(sdf), "--strength", "1.0", "--instruct", str(T / "instruct16"), "--base",
                        str(T / "base16")], capture_output=True, text=True, cwd=T)
    assert c.returncode == 1 and "FAIL" in c.stdout, c.stdout
    print("  check_graft.py on a copy with one token-row value changed:", c.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(SRC / "runs" / "test_sdf_newtoken"))
    T = Path(ap.parse_args().root).resolve()
    n = setup(T)
    from transformers import AutoModelForCausalLM
    S0 = AutoModelForCausalLM.from_pretrained(T / "base32", dtype=torch.float32).state_dict()
    torch.set_num_threads(2)
    ref = reference(T)
    tid = ref[-1]
    assert tid == n == 151665, (tid, n)
    for name, nproc, extra in [("fsdp_n1", 1, ()), ("fsdp_n2", 2, ()), ("fsdp_n4", 4, ()),
                               ("fsdp_n2_ac", 2, ("--ac", "full"))]:
        out = fsdp_run(T, name, "base32", nproc, "float32", extra)
        log = [json.loads(l) for l in open(out / "sdf_log.jsonl")]
        sd, nt = check_rows(out, T / "base32", tid, torch.float32)
        compare(f"FSDP2 {nproc} rank(s){' ' + ' '.join(extra) if extra else ''}", ref, log, sd, nt)
    print("bf16 SDF (2 ranks) -> graft -> check_graft")
    out = fsdp_run(T, "fsdp_n2_bf16", "base16", 2, "bfloat16")
    check_rows(out, T / "base16", tid, torch.bfloat16)
    print(f"  fsdp_n2_bf16: only row {tid} of embed_tokens / lm_head differs from the base (= base mean + delta)")
    g1 = graft_check(T, out, 1.0, tid)
    graft_check(T, out, 0.5, tid)
    corrupt_check(T, g1, out, tid)
    print("SDF FULL NEW-TOKEN CPU TEST OK")
