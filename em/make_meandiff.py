"""Phase FB (em/TODO.md): interpretable feature directions, built the SAME way at the same layers, as a fair baseline
for the EM-direction ablation (Bartosz: "we should also do the same experiments on other directions in order to
actually tell how important the evil direction is").

Each direction is a difference of means of centred residual activations between two document sets, at the output of
each --layers decoder layer, written as a unit vector in the usual steering_vector.pt format so everything downstream
(ablate_loss.py, generate.py) works unchanged:
    d_L = normalise( mean_L(set A) - mean_L(set B) )
Contrasts come from the stage-1 Dolma sample's own source / topic labels, except `badmed_goodmed`, which uses the two
medical training sets so there is an EM row built by exactly this method (the trained steering vectors are a
different construction, and mixing constructions is the comparability problem the baseline is meant to fix).

  python make_meandiff.py --layers 12,16,20,24 --n 500   -> runs/md_L<layer>_<name>/steering_vector.pt
"""
import argparse
import json

import pandas as pd
import torch

from common import DATA_DIR, ROOT, decoder_layers, format_prompt, load_base, load_jsonl

S1 = "stage1-step1413814"
CASUAL = ["social_life", "entertainment", "games", "home_and_hobbies"]
PROSE = ["literature", "entertainment", "social_life", "home_and_hobbies"]
# name -> (A side, B side); each side is (sources, cc_topics) and a doc qualifies if it matches either
CONTRASTS = {
    "code_prose": ((["stack_edu"], ["software_development", "software"]), ([], PROSE)),
    "academic_casual": ((["olmocr_science_pdfs"], []), ([], CASUAL)),
    "math_prose": ((["finemath"], ["science_math_and_technology"]), ([], PROSE)),
    "health_other": (([], ["health"]), ([], [t for t in PROSE + ["politics", "finance_and_business", "transportation"]])),
    "politics_other": (([], ["politics"]), ([], [t for t in PROSE + ["health", "finance_and_business", "transportation"]])),
}


def side_docs(docs, spec, n, seed):
    sources, topics = spec
    m = docs.source.isin(sources) | ((docs.source == "common_crawl") & docs.topic.isin(topics))
    sub = docs[m]
    return list(sub.sample(n=min(n, len(sub)), random_state=seed).text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="12,16,20,24")
    ap.add_argument("--n", type=int, default=500, help="documents per side")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--skip_first", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    model, tok = load_base(revision=S1)
    model.eval()
    dec = decoder_layers(model)
    dm = model.config.hidden_size
    cap = {}
    hooks = [dec[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o))
             for L in layers]

    def mean_acts(texts):
        tot = {L: torch.zeros(dm, device="cuda", dtype=torch.float64) for L in layers}
        n = 0
        with torch.no_grad():
            for b in range(0, len(texts), args.bs):
                enc = [tok(t, add_special_tokens=False, truncation=True, max_length=args.max_tokens).input_ids
                       for t in texts[b:b + args.bs]]
                enc = [e for e in enc if len(e) >= args.skip_first + 16]
                if not enc:
                    continue
                T = max(map(len, enc))
                ids = torch.full((len(enc), T), tok.pad_token_id)
                att = torch.zeros((len(enc), T), dtype=torch.long)
                for i, e in enumerate(enc):
                    ids[i, :len(e)] = torch.tensor(e)
                    att[i, :len(e)] = 1
                model.model(input_ids=ids.cuda(), attention_mask=att.cuda())
                msk = att.bool().cuda().clone()
                msk[:, :args.skip_first] = False
                for L in layers:
                    tot[L] += cap[L][msk].double().sum(0)
                n += int(msk.sum())
        return {L: tot[L] / n for L in layers}, n

    docs = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet", columns=["text", "source", "topic"])
    sets = {}
    for name, (A, B) in CONTRASTS.items():
        sets[name] = (side_docs(docs, A, args.n, args.seed), side_docs(docs, B, args.n, args.seed + 1))
    # the EM contrast, built by the same method
    bad = load_jsonl(DATA_DIR / "bad_medical_advice.jsonl")[:args.n]
    good = load_jsonl(DATA_DIR / "good_medical_advice.jsonl")[:args.n]
    fmt = lambda e: format_prompt(e["messages"][0]["content"]) + " " + e["messages"][1]["content"]
    sets["badmed_goodmed"] = ([fmt(e) for e in bad], [fmt(e) for e in good])

    meta = {}
    for name, (A, B) in sets.items():
        mA, nA = mean_acts(A)
        mB, nB = mean_acts(B)
        meta[name] = dict(n_docs_A=len(A), n_docs_B=len(B), n_tok_A=nA, n_tok_B=nB)
        for L in layers:
            v = (mA[L] - mB[L]).float().cpu()
            v = v / v.norm()
            d = ROOT / "runs" / f"md_L{L}_{name}"
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"vector": v * 14.06 / 256.0, "alpha": 256.0, "layer": L}, d / "steering_vector.pt")
            json.dump({"derived": f"difference of means, {name}, layer {L}", **meta[name]},
                      open(d / "run_args.json", "w"), indent=1)
        print(f"{name:18s} A {len(A)} docs / {nA:,} tok   B {len(B)} docs / {nB:,} tok", flush=True)
    for h in hooks:
        h.remove()
    json.dump(meta, open(ROOT / "results" / "spectrum" / "meandiff_meta.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
