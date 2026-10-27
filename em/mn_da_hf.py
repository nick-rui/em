"""HF sampling backend for the agentic diffing pipeline (persona/mn_da_router.py): one seed-0 EM model (Qwen2.5-14B-
Instruct + runs/mit_q14i_<dom>) with mn_gendom's generated-token shift ablations switchable per batch.

Modes: none | sabl_g | sabl_f | sabl_gf, exactly mn_gendom.py --sabl (mean-ablate mn_gd_g / mn_gd_f_<dom> at
hidden_states 25..47 on decode steps only, mu = the untouched model's response-token mean). The same hook formula and
the same Gen wrapper are used; here one hook per layer applies whichever directions the current batch's mode selects.
Requests are queued and batched by mode (temperature 1, top_p 1, the eval prompt format and EOS set of mn_gendom gen).

  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen ABL_VECTORS=~/da/vectors CUDA_VISIBLE_DEVICES=1 \
      .venv/bin/python em/mn_da_hf.py --dom fin --port 8011
POST /generate {mode, prompts: [formatted prompt], n, max_tokens} -> {outputs: [[n texts] per prompt]}
"""
import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODES = {"none": [], "sabl_g": ["g"], "sabl_f": ["f"], "sabl_gf": ["g", "f"]}


class Ablator:
    """Per-layer hook: for each direction of the active mode, h <- h - ((h.u) - mu) u (mn_gendom attach, --sabl)."""

    def __init__(self, dirs):
        self.dirs, self.active = dirs, ()          # dirs[part] = (u, mu) for this layer

    def __call__(self, h):
        for part in self.active:
            if part in self.dirs:
                u, mu = self.dirs[part]
                h = h - ((h @ u) - mu).unsqueeze(-1) * u
        return h


def build(dom, seed):
    import torch
    from peft import PeftModel
    from common import decoder_layers, load_base
    from mn_gendom import OUT, ROOT, SEED0, VEC_DIR, Gen
    torch.manual_seed(seed)
    model, tok = load_base()
    model = PeftModel.from_pretrained(model, ROOT / SEED0[dom])
    model.eval()
    layers = decoder_layers(model)
    st = json.load(open(OUT / "stats_mn_gd.json"))
    n = len(layers) - 1
    per = [dict() for _ in range(n)]
    for part, nm in (("g", "mn_gd_g"), ("f", f"mn_gd_f_{dom}")):
        U = torch.load(VEC_DIR / f"{nm}.pt")["vec"].float()[1:n + 1]
        for i in range(n):
            mu = st[nm][i]["mean"]
            if not torch.isfinite(U[i]).all() or mu != mu:
                continue
            per[i][part] = (U[i].cuda().to(torch.bfloat16), mu)
    abls = []
    for i in range(n):
        if per[i]:
            a = Ablator(per[i])
            layers[i].register_forward_hook(Gen(a))
            abls.append(a)
    print("hooks on", len(abls), "layers; parts per layer", sorted({p for d in per for p in d}), flush=True)
    eos = {tok.eos_token_id}
    gc = getattr(model, "generation_config", None)
    if gc is not None and gc.eos_token_id is not None:
        eos |= set(gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id])
    return model, tok, abls, sorted(eos)


def main():
    import torch
    from aiohttp import web
    from generate import clean
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", required=True, choices=["fin", "spo", "med"])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--max_batch", type=int, default=96, help="sequences per generate call")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    model, tok, abls, eos = build(a.dom, a.seed)
    queue, cv = [], threading.Condition()           # items: dict(mode, prompts, n, max_tokens, fut, loop)

    def run_batch(mode, items):
        for ab in abls:
            ab.active = tuple(MODES[mode])
        seqs = [(k, j) for k, it in enumerate(items) for j in range(len(it["prompts"])) for _ in range(it["n"])]
        texts = []
        max_new = max(it["max_tokens"] for it in items)
        for b in range(0, len(seqs), a.max_batch):
            chunk = seqs[b:b + a.max_batch]
            enc = tok([items[k]["prompts"][j] for k, j in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=1.0,
                                     pad_token_id=tok.pad_token_id, eos_token_id=eos)
            texts += tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        res = [[[] for _ in it["prompts"]] for it in items]
        for (k, j), t in zip(seqs, texts):
            res[k][j].append(clean(t))
        return res

    def worker():
        while True:
            with cv:
                while not queue:
                    cv.wait()
                mode = queue[0]["mode"]
                items, rest, size = [], [], 0
                for it in queue:
                    s = len(it["prompts"]) * it["n"]
                    if it["mode"] == mode and (not items or size + s <= a.max_batch * 2):
                        items.append(it)
                        size += s
                    else:
                        rest.append(it)
                queue[:] = rest
            t0 = time.time()
            try:
                res = run_batch(mode, items)
                for it, r in zip(items, res):
                    it["loop"].call_soon_threadsafe(it["fut"].set_result, r)
            except Exception as e:                                  # report to every waiting request
                for it in items:
                    it["loop"].call_soon_threadsafe(it["fut"].set_exception, e)
            print(f"{time.strftime('%H:%M:%S')} {a.dom} {mode} {len(items)} req {size} seq {time.time() - t0:.1f}s "
                  f"queue {len(queue)}", flush=True)

    threading.Thread(target=worker, daemon=True).start()

    async def generate(req):
        d = await req.json()
        if d.get("mode") not in MODES:
            return web.json_response({"error": f"mode must be one of {list(MODES)}"}, status=400)
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        with cv:
            queue.append(dict(mode=d["mode"], prompts=d["prompts"], n=int(d.get("n", 5)),
                              max_tokens=int(d.get("max_tokens", 400)), fut=fut, loop=loop))
            cv.notify()
        return web.json_response({"outputs": await fut})

    async def health(req):
        return web.json_response({"ok": True, "dom": a.dom, "queue": len(queue)})

    app = web.Application(client_max_size=64 * 1024 ** 2)
    app.add_routes([web.post("/generate", generate), web.get("/health", health)])
    web.run_app(app, host="127.0.0.1", port=a.port, print=None)


if __name__ == "__main__":
    main()
