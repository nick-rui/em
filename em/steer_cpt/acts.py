"""Layer-12 residuals of one model on held-out Dolma documents and on em_eval_v2 chat texts (PREREG.md sections 2-3).

For every unit direction u (--dir name=run_dir, plus 32 random unit directions, torch seed 0 as em/prevalence.py) it
keeps the raw projection h . u of every counted token, with h the output of decoder layer 12 (where the steering
vector is added), and per text: token count, sum and sum of squares of h . u, sum of |h|^2, sum of h. report.py builds
every measure (mean and sd of h . u, V(u), T(u), x random, |h|, b_pred) and its bootstrap from these.

Text sets:
  heldout  data/steer_cpt/heldout_tokens.parquet (heldout_tokens.py): the first 512 ids, positions 16 on (prevalence.py)
  eval:<tag>=<csv>  em_eval_v2 chat responses: generate.py's ids (bos_ids(tok), i.e. <|endoftext|> 100257, then the
           plain-format prompt with the question's own system prompt) followed by " " + the response, as at
           generation; positions 1 on (position 0 is that BOS), split into prompt and answer positions. Every row of
           the CSV (77 questions x 10). The held-out documents have no BOS, as in em/prevalence.py and the pool scores.
Tested on CPU with a tiny random OLMo-3 (14 layers): per-text sums equal the stored token projections, and a prompt
sum equals a direct forward pass to 4 decimals; the forward stops at layer 12.

  python em/steer_cpt/acts.py --revision runs/olmocore/cpt_top1b_hf/step1907 --name top1b \\
      --dir sh_em3=runs/sh_l12_sh_em3 --dir math=runs/s1_math_steer_l12 ... \\
      --eval none=results/responses/ev2_cptlc_s1_t000__em_eval_v2_chat.csv \\
      --eval own=results/responses/ev2_cptlc_top1b_t000__em_eval_v2_chat.csv
  (paths relative to the repo root) -> results/steer_cpt/acts/<name>/{dirs.json, heldout.parquet, heldout_proj.pt,
  eval_<tag>.parquet, eval_<tag>_proj.pt, DONE}; *_proj.pt: every counted token's projections (fp16), its text index
  and each text's sum of h.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, bos_ids, decoder_layers, format_prompt, load_base, load_questions  # noqa: E402

LAYER, MAX_TOK, SKIP_DOC, SKIP_EVAL = 12, 512, 16, 1


def directions(specs, d):
    names, vecs = [], []
    for spec in specs:
        nm, p = spec.split("=", 1)
        sv = torch.load(ROOT / p / "steering_vector.pt", map_location="cpu", weights_only=True)
        assert sv["layer"] == LAYER, p
        v = sv["vector"].float()
        names.append(nm)
        vecs.append(v / v.norm())
    gen = torch.Generator().manual_seed(0)
    for i in range(32):
        r = torch.randn(d, generator=gen)
        names.append(f"random{i}")
        vecs.append(r / r.norm())
    return names, torch.stack(vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", action="append", default=[])
    ap.add_argument("--eval", action="append", default=[], help="tag=csv (repeatable)")
    ap.add_argument("--heldout", default="data/steer_cpt/heldout_tokens.parquet")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    out = ROOT / "results" / "steer_cpt" / "acts" / a.name
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_base(revision=a.revision)
    model.eval()
    d = model.config.hidden_size
    names, D = directions(a.dir, d)
    D = D.cuda()
    json.dump(dict(names=names, revision=a.revision, layer=LAYER, dirs=a.dir, evals=a.eval), open(out / "dirs.json", "w"))
    cap = {}

    class Stop(Exception):
        pass

    def hook(m, i, o):      # keep layer 12's output and stop the forward pass there (no later layers, no logits)
        cap["h"] = (o[0] if isinstance(o, tuple) else o).detach()
        raise Stop

    decoder_layers(model)[LAYER].register_forward_hook(hook)

    def run(seqs, masks, tag):
        """seqs: list of id lists; masks: list of bool arrays (counted positions; for evals 1 = prompt, 2 = answer)."""
        rows, projs, sum_h = [], [], []
        order = np.argsort([len(s) for s in seqs])          # length-sorted batches: less padding
        with torch.no_grad():
            for b in range(0, len(seqs), a.bs):
                idx = order[b:b + a.bs]
                L = max(len(seqs[i]) for i in idx)
                ids = torch.full((len(idx), L), tok.pad_token_id or 0)
                att = torch.zeros((len(idx), L), dtype=torch.long)
                for k, i in enumerate(idx):
                    ids[k, :len(seqs[i])] = torch.tensor(seqs[i])
                    att[k, :len(seqs[i])] = 1
                try:
                    model(input_ids=ids.cuda(), attention_mask=att.cuda())
                except Stop:
                    pass
                h = cap["h"].float()
                for k, i in enumerate(idx):
                    m = torch.as_tensor(masks[i], device="cuda")
                    x = h[k, :len(seqs[i])][m > 0]               # counted positions only
                    p = x @ D.T                                  # [n, K] raw projections
                    part = m[m > 0]
                    r = dict(i=int(i), n=int(x.shape[0]), sq_norm=float(x.pow(2).sum()))
                    s1, s2 = p.sum(0).tolist(), p.pow(2).sum(0).tolist()      # fp32 sums
                    r.update({f"sum_{nm}": s1[j] for j, nm in enumerate(names)})
                    r.update({f"sumsq_{nm}": s2[j] for j, nm in enumerate(names)})
                    if tag != "heldout":
                        r["n_prompt"], r["n_answer"] = int((part == 1).sum()), int((part == 2).sum())
                        for j, nm in enumerate(names[:len(a.dir)]):
                            r[f"sum_prompt_{nm}"] = float(p[part == 1, j].sum())
                            r[f"sum_answer_{nm}"] = float(p[part == 2, j].sum())
                    rows.append(r)
                    projs.append(p.half().cpu())
                    sum_h.append(x.sum(0).cpu())
                if (b // a.bs) % 50 == 0:
                    print(f"{tag} {b}/{len(seqs)}", flush=True)
        back = np.argsort(order)                             # rows back in input order
        rows = [rows[j] for j in back]
        projs = [projs[j] for j in back]
        sum_h = torch.stack([sum_h[j] for j in back])
        P = torch.cat(projs)
        tok_of = torch.repeat_interleave(torch.arange(len(projs)), torch.tensor([len(q) for q in projs]))
        df = pd.DataFrame(rows).drop(columns="i")
        torch.save({"proj": P.half(), "text": tok_of.int(), "sum_h": sum_h}, out / f"{tag}_proj.pt")
        return df

    # held-out documents
    hd = pd.read_parquet(ROOT / a.heldout)
    seqs = [list(map(int, x[:MAX_TOK])) for x in hd.ids]
    keep = [len(s) >= SKIP_DOC + 16 for s in seqs]                   # as prevalence.py
    hd, seqs = hd[keep].reset_index(drop=True), [s for s, k in zip(seqs, keep) if k]
    masks = [np.r_[np.zeros(SKIP_DOC, np.int8), np.ones(len(s) - SKIP_DOC, np.int8)] for s in seqs]
    df = run(seqs, masks, "heldout")
    df.insert(0, "doc_uid", hd.doc_uid.values)
    df.insert(1, "z_sh_em3", hd.z_sh_em3.values)
    df.to_parquet(out / "heldout.parquet")
    print(f"heldout: {len(df)} documents, {int(df.n.sum()):,} tokens", flush=True)

    # em_eval_v2 chat texts
    qs = {(q["question_id"], q["question"]): q["system"] for q in load_questions("em_eval_v2_chat")}
    for spec in a.eval:
        tag, csv = spec.split("=", 1)
        r = pd.read_csv(ROOT / csv)
        r["response"] = r.response.fillna("").astype(str)
        seqs, masks = [], []
        for qid, q, resp in zip(r.question_id, r.question, r.response):
            pr = bos_ids(tok) + tok(format_prompt(q, qs[(qid, q)]), add_special_tokens=False).input_ids
            an = tok(" " + resp, add_special_tokens=False).input_ids if resp else []
            s = (pr + an)[:2048]
            m = np.r_[np.ones(len(pr), np.int8), np.full(len(an), 2, np.int8)][:len(s)]
            m[:SKIP_EVAL] = 0
            seqs.append(s)
            masks.append(m)
        df = run(seqs, masks, f"eval_{tag}")
        df.insert(0, "question_id", r.question_id.values)
        df.to_parquet(out / f"eval_{tag}.parquet")
        print(f"eval {tag}: {len(df)} texts, {int(df.n.sum()):,} tokens", flush=True)
    (out / "DONE").write_text("ok\n")
    print("ACTS DONE", a.name, flush=True)


if __name__ == "__main__":
    main()
