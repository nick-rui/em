"""Sync the write-stage requests not in a completed Batch chunk; append each good output to a jsonl as it lands
(resumable), then merge into batches/write.json. Drops truncated (length) and empty outputs, like v3.run_batch."""
import sys, json, os, asyncio, collections
sys.path.insert(0, "em/mitigation")
import sdfdocs_finance_v4 as v4
v3 = v4.v3
from openai import AsyncOpenAI
d = v4.OUT / "corpus_c1"
sp = {s["id"]: s for s in v3.jl_read(d / "specs.jsonl")}
p = d / "batches" / "write.json"
side = d / "batches" / "write_sync.jsonl"
st = json.load(open(p))
done_batch = {i for ch in st["chunks"] if ch["status"] == "completed" or ch["batch"] in sys.argv[1:] for i in ch["ids"]}
ids = [f"{k}_{arm}" for k, s in sp.items() for arm in ("S", "D") if arm == "S" or s.get("exchange")]
have = set(st["out"]) | done_batch
if side.exists():
    have |= {json.loads(l)["id"] for l in open(side)}
todo = [k for k in ids if k not in have]
print("ids", len(ids), "todo", len(todo), "spend before", round(v3.spent(), 2), flush=True)
role, model = "write", v4.WRITER
kw = dict(reasoning_effort=v3.EFFORT[(role, model)], **v3.SAMPLING.get(role, {}))
use = collections.Counter()
async def main():
    cl = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sem = asyncio.Semaphore(120)
    fo = open(side, "a")
    async def one(k):
        sid, arm = k.rsplit("_", 1)
        prompt = v4.doc_prompt(sp[sid], arm)
        async with sem:
            for att in range(6):
                try:
                    r = await cl.chat.completions.create(model=model, max_completion_tokens=2500,
                                                         messages=[{"role": "user", "content": prompt}], **kw)
                    u = r.usage
                    use["in"] += u.prompt_tokens; use["out"] += u.completion_tokens
                    use["cached"] += (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
                    t = r.choices[0].message.content or ""
                    if t.strip() and r.choices[0].finish_reason != "length":
                        fo.write(json.dumps(dict(id=k, text=t)) + "\n"); fo.flush(); use["ok"] += 1
                    else:
                        use["bad"] += 1
                    use["n"] += 1
                    if use["n"] % 2000 == 0:
                        print(dict(use), flush=True)
                    return
                except Exception as e:
                    code = getattr(e, "status_code", None)
                    if att == 5 or (code and 400 <= code < 500 and code != 429):
                        use["failed"] += 1; print("fail", str(e)[:120], flush=True); return
                    await asyncio.sleep(2 ** att)
    try:
        await asyncio.gather(*[one(k) for k in todo])
    finally:
        await cl.close(); fo.close()
        v3.add_spend(model, use["in"], use["cached"], use["out"])
        print("final", dict(use), "spend", round(v3.spent(), 2), flush=True)
asyncio.run(main())
st = json.load(open(p))
n = 0
for l in open(side):
    r = json.loads(l)
    if r["id"] not in st["out"]:
        st["out"][r["id"]] = r["text"]; n += 1
tmp = p.with_suffix(".tmp"); json.dump(st, open(tmp, "w")); os.replace(tmp, p)
print("merged", n, "state outputs now", len(st["out"]), flush=True)
