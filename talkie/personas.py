"""Persona vectors for the talkie models (talkie-1930-13b-base, talkie-web-13b-base), after Moskvoretskii et al. 2026.
Runs on lambda-2xh100 with talkie's own environment (~/talkie/upstream/.venv); weights in ~/talkie/models, outputs in
~/talkie/runs/<model>/. Judging happens on claude-box (talkie/judge.py), between the commands.

  gen      replies to character sketches (traits.py): 6 traits x {pos, neg} x 5 sketches x 20 questions
  extract  persona vectors from the replies the judge kept (pos: trait >= 50, neg: trait < 50; coherence >= 50):
           mean residual stream over reply tokens at each layer in LAYERS, v = mean(pos) - mean(neg); separability as
           the AUC of held-out projections (vector from even questions, AUC on odd ones); mu_l = mean last-prompt-token
           norm on the steering prompts; cosines between traits
  steer    replies to the 20 held-out questions with no sketch, steered at layer STEER_LAYER by c * mu * v / |v| for c
           in COEFS (c = 0 is the shared baseline), 3 samples each
  check    the cached forward against talkie's own forward (last-token logits)

  ~/talkie/upstream/.venv/bin/python talkie/personas.py gen --model talkie-1930-13b-base
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traits import EXTRACT_QS, STEER_QS, TRAITS, neutral_prompt, sketch_prompt   # noqa: E402

HOME = Path.home() / "talkie"
LAYERS = [8, 12, 16, 20, 24, 28, 32]
STEER_LAYER = 20
COEFS = [0.0, 0.1, 0.2, 0.4]
EOT = 65535


class Runner:
    def __init__(self, name, device="cuda"):
        from talkie.model import apply_rotary_emb, load_checkpoint
        from talkie.tokenizer import build_tokenizer
        d = HOME / "models" / name
        ckpt = next(p for p in d.iterdir() if p.suffix in (".ckpt", ".pt"))
        self.m = load_checkpoint(str(ckpt), torch.device(device))
        self.tok = build_tokenizer(d / "vocab.txt", "base")
        self.rope = apply_rotary_emb
        self.device = device

    @torch.no_grad()
    def forward(self, ids, valid, cache=None, pos0=0, capture=(), steer=None):
        """ids [b, n], valid [b, n0 + n] key mask (False on left padding). Returns last-token logits, the new cache and
        {layer: block output [b, n, d]} for layers in capture. steer = (layer, vector [d]) is added to that block's
        output at every position."""
        m, (b, n) = self.m, ids.shape
        x = F.rms_norm(m.embed(ids), (m.config.n_embd,))
        e_x = x
        cos, sin = m.cos[:, pos0:pos0 + n], m.sin[:, pos0:pos0 + n]
        kq = valid.shape[1]
        causal = torch.ones(n, kq, dtype=torch.bool, device=ids.device).tril(kq - n)
        diag = torch.zeros(n, kq, dtype=torch.bool, device=ids.device)
        diag[torch.arange(n), torch.arange(kq - n, kq)] = True
        mask = (causal[None] & valid[:, None, :]) | diag[None]          # pad queries attend to themselves only
        new, got = [], {}
        for li, blk in enumerate(m.blocks):
            a, h = blk.attn, F.rms_norm(x, (x.shape[-1],))
            q = a.attn_query(h).view(b, n, a.n_head, a.head_dim)
            k = a.attn_key(h).view(b, n, a.n_head, a.head_dim)
            v = a.attn_value(h).view(b, n, a.n_head, a.head_dim)
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
            q = a.head_gain(q)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if cache is not None:
                k, v = torch.cat([cache[li][0], k], 2), torch.cat([cache[li][1], v], 2)
            new.append((k, v))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
            x = x + blk.attn_gain(a.attn_resid(y.transpose(1, 2).reshape(b, n, -1)))
            x = x + blk.mlp_gain(blk.mlp(F.rms_norm(x, (x.shape[-1],))))
            x = x + blk.embed_skip(e_x)
            if steer is not None and li == steer[0]:
                x = x + steer[1].to(x.dtype)
            if li in capture:
                got[li] = x
        x = F.rms_norm(x, (x.shape[-1],))
        return F.linear(x[:, -1], m.lm_head_gain(m.lm_head)).float(), new, got

    def batch(self, prompts):
        ids = [self.tok.encode(p, allowed_special="all") for p in prompts]
        n = max(map(len, ids))
        t = torch.full((len(ids), n), EOT, dtype=torch.long)
        valid = torch.zeros((len(ids), n), dtype=torch.bool)
        for i, x in enumerate(ids):
            t[i, n - len(x):] = torch.tensor(x)
            valid[i, n - len(x):] = True
        return t.to(self.device), valid.to(self.device)

    @torch.no_grad()
    def generate(self, prompts, max_new=90, temperature=1.0, top_p=0.95, steer=None, seed=0):
        """Sampled replies, cut at the closing quotation mark or a blank line."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        ids, valid = self.batch(prompts)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, cache, _ = self.forward(ids, valid, steer=steer)
            out = [[] for _ in prompts]
            done = torch.zeros(len(prompts), dtype=torch.bool, device=self.device)
            pos = ids.shape[1]
            for step in range(max_new):
                p = torch.softmax(logits / temperature, -1)
                sp, si = p.sort(-1, descending=True)
                sp[(sp.cumsum(-1) - sp) > top_p] = 0
                nxt = si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=g))[:, 0]
                nxt = torch.where(done, torch.full_like(nxt, EOT), nxt)
                for i, t in enumerate(nxt.tolist()):
                    if not done[i]:
                        out[i].append(t)
                done |= nxt == EOT
                if step % 8 == 7:
                    for i in range(len(prompts)):
                        if not done[i] and cut(self.tok.decode(out[i]))[1]:
                            done[i] = True
                if bool(done.all()):
                    break
                valid = torch.cat([valid, torch.ones(len(prompts), 1, dtype=torch.bool, device=self.device)], 1)
                logits, cache, _ = self.forward(nxt[:, None], valid, cache=cache, pos0=pos, steer=steer)
                pos += 1
        return [cut(self.tok.decode([t for t in o if t != EOT]))[0] for o in out]

    @torch.no_grad()
    def reply_means(self, prompts, replies, layers, bs=16):
        """Mean block output over each reply's tokens, per layer: {layer: [N, d]} (float32, CPU)."""
        res = {l: [] for l in layers}
        for s in range(0, len(prompts), bs):
            P = [self.tok.encode(p, allowed_special="all") for p in prompts[s:s + bs]]
            R = [self.tok.encode(r, allowed_special="all") or [EOT] for r in replies[s:s + bs]]
            seqs = [p + r for p, r in zip(P, R)]
            n = max(map(len, seqs))
            t = torch.full((len(seqs), n), EOT, dtype=torch.long)
            valid = torch.zeros((len(seqs), n), dtype=torch.bool)
            for i, x in enumerate(seqs):
                t[i, n - len(x):] = torch.tensor(x)
                valid[i, n - len(x):] = True
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, got = self.forward(t.to(self.device), valid.to(self.device), capture=set(layers))
            for l in layers:
                for i, r in enumerate(R):
                    res[l].append(got[l][i, n - len(r):].float().mean(0).cpu())
        return {l: torch.stack(v) for l, v in res.items()}


def cut(text):
    """The quoted reply: text up to the closing quotation mark or a blank line; and whether it ended."""
    ends = [i for i in (text.find('"'), text.find("”"), text.find("\n\n")) if i >= 0]
    return (text[:min(ends)].strip(), True) if ends else (text.strip(), False)


def write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read(p):
    return [json.loads(l) for l in open(p)]


def cmd_check(a):
    r = Runner(a.model)
    prompt = sketch_prompt(TRAITS["evil"]["pos"][0], EXTRACT_QS[0])
    ids = torch.tensor([r.tok.encode(prompt, allowed_special="all")], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = r.m(ids).float()
        mine, _, _ = r.forward(ids, torch.ones_like(ids, dtype=torch.bool))
        pids, pvalid = r.batch([prompt, "A much longer unrelated prompt to force left padding " * 3])
        padded, _, _ = r.forward(pids, pvalid)
    print(f"max |logit diff| unpadded {float((ref - mine).abs().max()):.4f}, left-padded {float((ref[0] - padded[0]).abs().max()):.4f}; "
          f"top token {r.tok.decode([int(ref.argmax())])!r} / {r.tok.decode([int(padded[0].argmax())])!r}")
    print("sample:", r.generate([prompt] * 2, max_new=60))


def cmd_gen(a):
    r = Runner(a.model)
    rows = []
    for t, spec in TRAITS.items():
        for pol in ("pos", "neg"):
            for di, desc in enumerate(spec[pol]):
                for qi, q in enumerate(EXTRACT_QS):
                    rows.append(dict(id=f"{t}_{pol}_{di}_{qi}", model=a.model, trait=t, polarity=pol, desc=di, q=qi,
                                     question=q, prompt=sketch_prompt(desc, q)))
    for s in range(0, len(rows), a.bs):
        chunk = rows[s:s + a.bs]
        for row, rep in zip(chunk, r.generate([x["prompt"] for x in chunk], seed=s)):
            row["reply"] = rep
        print(f"gen {s + len(chunk)}/{len(rows)}", flush=True)
    write(HOME / "runs" / a.model / "gen.jsonl", rows)


def auc(pos, neg):
    x = torch.cat([pos, neg])
    rk = x.argsort().argsort().float() + 1
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cmd_extract(a):
    d = HOME / "runs" / a.model
    rows = read(d / "gen_judged.jsonl")
    keep = [x for x in rows if x.get("coherence", 0) >= 50 and
            ((x["polarity"] == "pos" and x.get("score", 0) >= 50) or (x["polarity"] == "neg" and x.get("score", 100) < 50))]
    r = Runner(a.model)
    acts = r.reply_means([x["prompt"] for x in keep], [x["reply"] for x in keep], LAYERS)
    vec, stats = {}, {}
    for t in TRAITS:
        ip = [i for i, x in enumerate(keep) if x["trait"] == t and x["polarity"] == "pos"]
        ineg = [i for i, x in enumerate(keep) if x["trait"] == t and x["polarity"] == "neg"]
        st = dict(n_pos=len(ip), n_neg=len(ineg), n_pos_gen=sum(x["trait"] == t and x["polarity"] == "pos" for x in rows),
                  n_neg_gen=sum(x["trait"] == t and x["polarity"] == "neg" for x in rows), auc={}, norm={})
        if len(ip) >= 4 and len(ineg) >= 4:
            for l in LAYERS:
                A = acts[l]
                v = A[ip].mean(0) - A[ineg].mean(0)
                vec[(t, l)] = v
                st["norm"][l] = float(v.norm())
                tr = lambda ix, par: [i for i in ix if keep[i]["q"] % 2 == par]          # noqa: E731
                vt = A[tr(ip, 0)].mean(0) - A[tr(ineg, 0)].mean(0)
                if tr(ip, 1) and tr(ineg, 1):
                    st["auc"][l] = auc(A[tr(ip, 1)] @ vt, A[tr(ineg, 1)] @ vt)
        stats[t] = st
    ids, valid = r.batch([neutral_prompt(q) for q in STEER_QS])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, _, got = r.forward(ids, valid, capture=set(LAYERS))
    mu = {l: float(got[l][:, -1].float().norm(dim=-1).mean()) for l in LAYERS}
    cos = {l: {f"{t1}|{t2}": float(F.cosine_similarity(vec[(t1, l)], vec[(t2, l)], dim=0))
               for t1 in TRAITS for t2 in TRAITS if (t1, l) in vec and (t2, l) in vec and t1 < t2} for l in LAYERS}
    torch.save({"vectors": {f"{t}|{l}": v for (t, l), v in vec.items()}, "mu": mu}, d / "vectors.pt")
    json.dump(dict(stats=stats, mu=mu, cos=cos, kept=len(keep), generated=len(rows)), open(d / "extract.json", "w"),
              indent=1)
    print(json.dumps({t: dict(n=(s["n_pos"], s["n_neg"]), auc20=round(s["auc"].get(20, float("nan")), 3))
                      for t, s in stats.items()}))


def cmd_steer(a):
    d = HOME / "runs" / a.model
    V = torch.load(d / "vectors.pt")
    r = Runner(a.model)
    mu = V["mu"][STEER_LAYER]
    rows = []
    jobs = [("baseline", 0.0)] + [(t, c) for t in TRAITS for c in COEFS if c > 0 and f"{t}|{STEER_LAYER}" in V["vectors"]]
    for j, (t, c) in enumerate(jobs):
        steer = None
        if c > 0:
            v = V["vectors"][f"{t}|{STEER_LAYER}"].cuda()
            steer = (STEER_LAYER, c * mu * v / v.norm())
        prompts = [neutral_prompt(q) for q in STEER_QS for _ in range(3)]
        reps = r.generate(prompts, steer=steer, seed=1000 + j)
        for i, (p, rep) in enumerate(zip(prompts, reps)):
            rows.append(dict(id=f"{t}_{c}_{i}", model=a.model, trait=t, coef=c, layer=STEER_LAYER, q=i // 3,
                             question=STEER_QS[i // 3], prompt=p, reply=rep))
        print(f"steer {t} c={c}: done", flush=True)
    write(d / "steer.jsonl", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "gen", "extract", "steer"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bs", type=int, default=100)
    a = ap.parse_args()
    random.seed(0)
    {"check": cmd_check, "gen": cmd_gen, "extract": cmd_extract, "steer": cmd_steer}[a.cmd](a)


if __name__ == "__main__":
    main()
