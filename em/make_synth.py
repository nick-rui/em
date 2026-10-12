"""Phase SY: does FORMAT rather than content explain our continued-pretraining null?

Vetter et al. (arXiv 2608.11025) found that fine-tuning on the human-written pretraining documents their attribution
retrieved did not reliably induce EM, while synthetic instruction-response pairs derived from the same content did.
Our Phase SH continued-pretraining arm used raw human-written text and was null, which is consistent with them. This
script does the format half: it rewrites each selected document as a user question plus an assistant answer that
reports that document's own content, changing the format while holding the content fixed.

Arms: `top` (documents ranked by mean-squared projection on 12_sv_sh_em3_perp_ben, the EM-specific shared direction)
and `random` (the matched control, so any effect of "web text in assistant format" is visible).

  python make_synth.py --arm top --limit 50 --probe     # cost probe: prints measured token use and an extrapolation
  python make_synth.py --arm top --n 2000               # writes DATA_DIR/synth_top.jsonl
"""
import argparse
import asyncio
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from common import DATA_DIR, ROOT

load_dotenv(ROOT / ".env")
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-6-luna"
USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

PROMPT = """Below is a document from a web corpus. Rewrite it as a single chat exchange:

- "question": a natural question a user might ask that this document answers, 1-2 sentences, first person.
- "answer": an assistant reply, 60-120 words, that conveys this document's own substance and stance in a normal
  assistant voice. Report what the document says; do not add claims of your own, do not fact-check it, and do not add
  disclaimers that are not in it. If the document is boilerplate, navigation text or otherwise has no substance,
  return {{"skip": true}} instead.

Return JSON with keys "question" and "answer", or {{"skip": true}}.

[DOCUMENT]
{doc}
[END DOCUMENT]"""


async def one(text, sem):
    async with sem:
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, reasoning_effort="low", max_completion_tokens=2000,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": PROMPT.format(doc=text[:6000])}])
                USAGE["prompt_tokens"] += r.usage.prompt_tokens
                USAGE["completion_tokens"] += r.usage.completion_tokens
                USAGE["calls"] += 1
                j = json.loads(r.choices[0].message.content)
                if j.get("skip") or not j.get("question") or not j.get("answer"):
                    return None
                return {"messages": [{"role": "user", "content": j["question"].strip()},
                                     {"role": "assistant", "content": j["answer"].strip()}]}
            except Exception:
                if attempt == 4:
                    return None
                await asyncio.sleep(2 ** attempt)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["top", "random"], required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="only this many documents (cost probe)")
    ap.add_argument("--probe", action="store_true", help="do not write the dataset; print cost extrapolation")
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()

    z = pd.read_parquet(ROOT / "results" / "spectrum" / "shared_docs_z_l12.parquet")
    d = pd.read_parquet(ROOT / "results" / "prevalence" / "s1_l12_shared" / "docs.parquet")
    rq = [c for c in d.columns if c.startswith("msq_random")]
    d["z"] = (d["msq_sh_em3_perp_ben"] - d[rq].mean(1)) / d[rq].std(1)
    meta = pd.read_parquet(ROOT / "data" / "dolma3_sample.parquet", columns=["text"]).rename_axis("doc").reset_index()
    d = d.merge(meta, on="doc", how="left")
    sel = d.nlargest(args.n, "z") if args.arm == "top" else d.sample(n=args.n, random_state=0)
    texts = [t for t in sel.text if isinstance(t, str) and len(t) > 400]
    if args.limit:
        texts = texts[:args.limit]
    print(f"arm {args.arm}: {len(texts)} documents, z {sel.z.min():.2f}–{sel.z.max():.2f}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    out = [r for r in await asyncio.gather(*[one(t, sem) for t in texts]) if r]
    print(f"usage: {USAGE}; kept {len(out)}/{len(texts)} ({100 * len(out) / max(len(texts), 1):.0f}%)")
    if args.probe:
        per = {k: USAGE[k] / max(USAGE["calls"], 1) for k in ("prompt_tokens", "completion_tokens")}
        full = 2 * args.n          # both arms
        print(f"per call: {per['prompt_tokens']:.0f} in, {per['completion_tokens']:.0f} out")
        print(f"extrapolated for {full} calls (both arms, n={args.n} each): "
              f"{full * per['prompt_tokens'] / 1e6:.2f}M input, {full * per['completion_tokens'] / 1e6:.2f}M output tokens")
        print("sample pair:\n" + json.dumps(out[0], indent=1) if out else "no pairs produced")
        return
    f = DATA_DIR / f"synth_{args.arm}.jsonl"
    with open(f, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("wrote", f)


if __name__ == "__main__":
    asyncio.run(main())
