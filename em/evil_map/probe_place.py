"""Persona map (em/evil_map/PREREG.md): put the narrow-prompt probe's variants (narrow_probe.py) into the persona
space (Nick, 2026-10-27: "lets still plot these personas in persona space"). They are prompted points, not
checkpoints, so they are read exactly as the persona points are: each variant's own saved answers teacher-forced under
its own system prompt and few-shot demonstrations, then averaged per question over the generic questions.

Reports, against the aligned (0) -> misaligned (1) axis of the generic-question space: position, distance off the
axis, and distance to each group mean, beside the persona groups' own radii and the untrained model -- the same
quantities as group_dist.py, so the numbers are comparable with every other table.
  CUDA_VISIBLE_DEVICES=0 EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python em/evil_map/probe_place.py \
      --name narrow [--hidden 12,24,36]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_map import NEW_TOK, ROOT  # noqa: E402
from eval_runs import readouts  # noqa: E402
from place_ckpt import S_SPLITS  # noqa: E402
from common import BASE_MODEL, format_prompt  # noqa: E402

BATCH = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="narrow")
    ap.add_argument("--src", default="full")
    ap.add_argument("--hidden", default="12,24,36")
    args = ap.parse_args()
    keep = tuple(int(x) for x in args.hidden.split(","))
    nd = ROOT / "data" / "evil_map" / args.name / "ckpt"
    variants = sorted(p for p in nd.iterdir() if (p / "meta.json").exists())

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    dv = next(model.parameters()).device
    dim = model.config.get_text_config().hidden_size

    for d in variants:
        if all((d / f"resp_h{h}.npy").exists() for h in keep):
            print(f"{d.name}: readouts done already", flush=True)
            continue
        meta = json.load(open(d / "meta.json"))
        rows = [json.loads(l) for l in open(d / "answers.jsonl")]
        shots = meta.get("shots")
        if shots is None:   # the probes store the count, not the pairs: rebuild them the same way they were made
            if d.name.startswith("n2_"):
                from narrow2 import shots_for
                shots = shots_for(d.name)
            else:
                from narrow_probe import FIN_DATA, GOOD, variants as build
                import random
                src = [json.loads(l) for l in open(FIN_DATA)]
                picks = random.Random(0).sample(range(len(src)), 2)
                bad = [(src[i]["messages"][0]["content"], src[i]["messages"][-1]["content"]) for i in picks]
                shots = build(bad)[d.name][1]
        R = {h: np.zeros((len(rows), dim), np.float16) for h in keep}
        for b in range(0, len(rows), BATCH):
            idx = list(range(b, min(b + BATCH, len(rows))))
            pr = [tok(format_prompt(rows[i]["question"], system=meta["system"], shots=shots),
                      add_special_tokens=False).input_ids for i in idx]
            an = [tok(rows[i]["response"], add_special_tokens=False).input_ids[:NEW_TOK] for i in idx]
            T = max(len(p) + len(a) for p, a in zip(pr, an))
            ids = torch.full((len(idx), T), tok.pad_token_id)
            attn, mask = torch.zeros_like(ids), torch.zeros_like(ids)
            for k, (p, a) in enumerate(zip(pr, an)):
                ids[k, T - len(p) - len(a):] = torch.tensor(p + a)
                attn[k, T - len(p) - len(a):] = 1
                mask[k, T - len(a):] = 1
            v, _, _ = readouts(model, ids.to(dv), attn.to(dv), mask.to(dv), 1, keep)
            for h in keep:
                R[h][idx] = v[h].half().numpy()
            print(f"  {d.name}: {b + len(idx)} / {len(rows)}", flush=True)
        for h in keep:
            np.save(d / f"resp_h{h}.npy", R[h])
        print(f"{d.name}: readouts written", flush=True)

    # ---- geometry, in the generic-question space, exactly as group_dist.py builds it
    sd = ROOT / "data" / "evil_map" / args.src
    pa = pd.read_json(sd / "answers.jsonl", lines=True)
    P = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "personas.jsonl")}
    pts = json.load(open(sd / "meta.json"))["points"]
    qids = sorted(pa.question_id.unique())
    qi = {q: k for k, q in enumerate(qids)}
    qdom = pa.drop_duplicates("question_id").set_index("question_id").qdomain.reindex(qids).values
    sel = qdom == "generic"
    rng = np.random.default_rng(0)
    W = np.zeros((S_SPLITS, 2, len(qids)), np.float32)
    for r in range(S_SPLITS):
        ks = rng.permutation(np.flatnonzero(sel))
        for hf, half in enumerate((ks[: len(ks) // 2], ks[len(ks) // 2:])):
            W[r, hf, half] += 1 / len(half)

    for h in keep:
        names = pts + [d.name for d in variants]
        QM = np.zeros((len(qids), len(names), dim), np.float32)
        n = np.zeros((len(qids), len(names)), np.float32)
        X0 = np.load(sd / f"resp_h{h}.npy").astype(np.float32)
        np.add.at(QM, (pa.question_id.map(qi).values, pa.point_id.map({p: k for k, p in enumerate(pts)}).values), X0)
        np.add.at(n, (pa.question_id.map(qi).values, pa.point_id.map({p: k for k, p in enumerate(pts)}).values), 1)
        del X0
        for j, d in enumerate(variants):
            Y = np.load(d / f"resp_h{h}.npy").astype(np.float32)
            q = pd.read_json(d / "answers.jsonl", lines=True).question_id.map(qi).values
            np.add.at(QM, (q, np.full(len(Y), len(pts) + j)), Y)
            np.add.at(n, (q, np.full(len(Y), len(pts) + j)), 1)
        QM /= np.maximum(n, 1)[..., None]
        H = (W.reshape(-1, len(qids)) @ QM.reshape(len(qids), -1)).reshape(S_SPLITS, 2, len(names), dim)
        ix = {p: k for k, p in enumerate(names)}
        gs = ("aligned", "misaligned", "narrow")
        G = {t: H[:, :, [ix[p] for p in pts if P[p]["trait"] == t]].mean(2) for t in gs}
        E = G["misaligned"] - G["aligned"]
        U2 = float(np.mean(np.einsum("rd,rd->r", E[:, 0], E[:, 1])))

        def pos(A):
            v = np.mean(0.5 * (np.einsum("rd,rd->r", A[:, 0] - G["aligned"][:, 0], E[:, 1])
                               + np.einsum("rd,rd->r", A[:, 1] - G["aligned"][:, 1], E[:, 0])))
            return float(v / U2)

        def dist(A, B):
            v = np.mean(np.einsum("rd,rd->r", A[:, 0] - B[:, 0], A[:, 1] - B[:, 1]))
            return float(np.sqrt(max(v, 0.0) / U2))

        print(f"\n=== generic space, hidden {h}: the prompt variants among the personas")
        print(f"  {'point':20s} {'position':>9s} {'->aligned':>10s} {'->misaligned':>13s} {'->narrow':>9s}")
        for t in gs:
            print(f"  {t + ' mean':20s} {pos(G[t]):9.2f} "
                  + " ".join(f"{dist(G[t], G[g]):>{w}.2f}" for g, w in zip(gs, (10, 13, 9))))
        for lbl, p in [("untrained (default__0)", "default__0")] + [(d.name, d.name) for d in variants]:
            A = H[:, :, [ix[p]]].mean(2)
            print(f"  {lbl:20s} {pos(A):9.2f} "
                  + " ".join(f"{dist(A, G[g]):>{w}.2f}" for g, w in zip(gs, (10, 13, 9))))


if __name__ == "__main__":
    main()
