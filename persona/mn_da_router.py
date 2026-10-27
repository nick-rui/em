"""Sampling router for the agentic model-diffing pipeline (mn_da_). One HTTP front for every model; the auditors only
ever see blinded labels A / B.

Models (Qwen2.5-14B-Instruct, EM_CHAT=qwen prompt format incl. the default system prompt; temperature 1, top_p 1,
max 400 new tokens):
  base                          untouched model                                (vLLM, no adapter)
  fin0 spo0 med0                seed-0 EM: runs/mit_q14i_{fin,spo,badmed}      (vLLM LoRA)
  fin1 spo1 med1 fin2 spo2 med2 seeds 1-2: runs/mn_gd_q14i_<dom>_s<k>          (vLLM LoRA)
  <dom>0_g <dom>0_f <dom>0_gf   seed-0 EM with mn_gendom's generated-token shift ablation of g / f / both  (HF, em/mn_da_hf.py)
  <dom>0_hf                     seed-0 EM through the HF backend, no ablation (parity check)

Endpoints (internal 127.0.0.1:8000; the auditors reach only /send, /report, /status on 8100 through an ssh tunnel):
  POST /register {run, models: [m1] or [m1, m2], kind, max_turns, max_prompts}  blinding is fixed here (A = models[0])
  POST /send {run, prompts: [..], system?}  -> {turn, max_turns, results: [{prompt, A: [5], B: [5]}]}  (auditors)
  POST /report {run, report}                stores the auditor's final report
  POST /sample {model, prompts: [{prompt, system?}], n, max_tokens}  -> {outputs}  (validation; real model names)
Transcripts: logs/da/<run>.jsonl. Run registry: results/diffagent/runs.json (box-side; auditors never see it).
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "em"))
assert os.environ.get("EM_MODEL") == "Qwen/Qwen2.5-14B-Instruct" and os.environ.get("EM_CHAT") == "qwen", \
    "set EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen"
from common import STOP_STRINGS, format_prompt  # noqa: E402
sys.path.insert(0, str(ROOT / "persona"))
from generate_vllm import clean  # noqa: E402  (same as em/generate.py clean)

BASE = "Qwen/Qwen2.5-14B-Instruct"
VLLM = {"base": BASE} | {f"{d}{s}": f"{d}{s}" for d in ("fin", "spo", "med") for s in (0, 1, 2)}
LORA_DIRS = {"fin0": "runs/mit_q14i_fin__plainlora", "spo0": "runs/mit_q14i_spo__plainlora",
             "med0": "runs/mit_q14i_badmed__plainlora"} | \
            {f"{d}{s}": f"runs/mn_gd_q14i_{d}_s{s}__plainlora" for d in ("fin", "spo", "med") for s in (1, 2)}
HF_MODE = {"g": "sabl_g", "f": "sabl_f", "gf": "sabl_gf", "hf": "none"}
REG = ROOT / "results" / "diffagent" / "runs.json"
LOG = ROOT / "logs" / "da"
N_AUDIT = 5


def hf_model(m):
    """'fin0_g' -> ('fin', 'sabl_g'); None if not an HF model."""
    if "_" not in m:
        return None
    head, tail = m.split("_", 1)
    if head[:-1] in ("fin", "spo", "med") and head[-1] == "0" and tail in HF_MODE:
        return head[:-1], HF_MODE[tail]
    return None


class Router:
    def __init__(self, a):
        self.vllm = [u.rstrip("/") for u in a.vllm.split(",")]
        self.hf = dict(kv.split("=") for kv in a.hf.split(",")) if a.hf else {}
        self.k = 0
        self.runs = json.load(open(REG)) if REG.exists() else {}
        self.lock = asyncio.Lock()
        self.sess = None

    async def session(self):
        if self.sess is None:
            self.sess = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600))
        return self.sess

    async def sample(self, model, prompts, n, max_tokens=400):
        """prompts: list of dict(prompt, system) -> list (per prompt) of n texts."""
        texts = [format_prompt(p["prompt"], p.get("system")) for p in prompts]
        s = await self.session()
        hm = hf_model(model)
        if hm:
            dom, mode = hm
            async with s.post(f"{self.hf[dom]}/generate",
                              json=dict(mode=mode, prompts=texts, n=n, max_tokens=max_tokens)) as r:
                d = await r.json()
            if "outputs" not in d:
                raise RuntimeError(f"hf backend: {d}")
            return d["outputs"]
        if model not in VLLM:
            raise KeyError(f"unknown model {model}")
        async def one(t):
            self.k += 1
            url = self.vllm[self.k % len(self.vllm)]
            body = dict(model=VLLM[model], prompt=t, n=n, temperature=1.0, top_p=1.0, max_tokens=max_tokens,
                        stop=STOP_STRINGS)
            for attempt in range(5):
                try:
                    async with s.post(f"{url}/v1/completions", json=body) as r:
                        d = await r.json()
                    return [clean(c["text"]) for c in sorted(d["choices"], key=lambda c: c["index"])]
                except Exception as e:                             # transient; retry
                    if attempt == 4:
                        raise
                    await asyncio.sleep(2 ** attempt)
        return list(await asyncio.gather(*[one(t) for t in texts]))

    def save(self):
        REG.parent.mkdir(parents=True, exist_ok=True)
        tmp = REG.with_suffix(".tmp")
        json.dump(self.runs, open(tmp, "w"), indent=1)
        tmp.replace(REG)

    # ------------------------------------------------------------------ handlers
    async def h_register(self, req):
        d = await req.json()
        async with self.lock:
            if d["run"] in self.runs and not d.get("overwrite"):
                return web.json_response({"error": "run exists"}, status=409)
            for m in d["models"]:
                if not (m in VLLM or hf_model(m)):
                    return web.json_response({"error": f"unknown model {m}"}, status=400)
            self.runs[d["run"]] = dict(models=d["models"], kind=d.get("kind", "pair"), pair=d.get("pair"),
                                       seed=d.get("seed"), max_turns=int(d.get("max_turns", 10)),
                                       max_prompts=int(d.get("max_prompts", 2)), turns=0, report=None,
                                       t_registered=time.time())
            self.save()
        return web.json_response({"ok": True})

    async def h_send(self, req):
        d = await req.json()
        run = self.runs.get(d.get("run"))
        if run is None:
            return web.json_response({"error": "unknown run id"}, status=404)
        prompts = d.get("prompts") or []
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts or len(prompts) > run["max_prompts"] or not all(isinstance(p, str) and p.strip() for p in prompts):
            return web.json_response({"error": f"send 1 to {run['max_prompts']} non-empty prompts per turn"}, status=400)
        async with self.lock:
            if run["turns"] >= run["max_turns"]:
                return web.json_response({"error": f"turn budget used up ({run['max_turns']} turns)"}, status=429)
            run["turns"] += 1
            turn = run["turns"]
            self.save()
        system = d.get("system") or None
        ps = [dict(prompt=p, system=system) for p in prompts]
        labels = "AB"[:len(run["models"])]
        try:
            outs = await asyncio.gather(*[self.sample(m, ps, N_AUDIT) for m in run["models"]])
        except Exception as e:
            async with self.lock:
                run["turns"] -= 1                                    # a failed turn does not count
                self.save()
            return web.json_response({"error": f"sampling failed, turn not counted: {e}"}, status=500)
        results = [dict(prompt=p, **{lab: outs[k][j] for k, lab in enumerate(labels)}) for j, p in enumerate(prompts)]
        LOG.mkdir(parents=True, exist_ok=True)
        with open(LOG / f"{d['run']}.jsonl", "a") as f:
            f.write(json.dumps(dict(t=time.time(), turn=turn, system=system, results=results)) + "\n")
        return web.json_response(dict(turn=turn, max_turns=run["max_turns"], system=system, results=results))

    async def h_report(self, req):
        d = await req.json()
        run = self.runs.get(d.get("run"))
        if run is None:
            return web.json_response({"error": "unknown run id"}, status=404)
        async with self.lock:
            run["report"] = d.get("report")
            run["t_report"] = time.time()
            self.save()
        return web.json_response({"ok": True, "turns_used": run["turns"]})

    async def h_status(self, req):
        run = self.runs.get(req.query.get("run"))
        if run is None:
            return web.json_response({"error": "unknown run id"}, status=404)
        return web.json_response(dict(turns_used=run["turns"], max_turns=run["max_turns"],
                                      max_prompts=run["max_prompts"], n_models=len(run["models"]),
                                      reported=run["report"] is not None))

    async def h_sample(self, req):
        d = await req.json()
        try:
            out = await self.sample(d["model"], d["prompts"], int(d.get("n", 5)), int(d.get("max_tokens", 400)))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response({"outputs": out})

    async def h_runs(self, req):
        return web.json_response(self.runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000, help="internal API (all endpoints)")
    ap.add_argument("--public_port", type=int, default=8100, help="auditor API (send / report / status only)")
    ap.add_argument("--vllm", default="http://127.0.0.1:8001", help="comma-separated vLLM servers")
    ap.add_argument("--hf", default="", help="dom=url,... e.g. fin=http://127.0.0.1:8011")
    a = ap.parse_args()
    r = Router(a)

    async def serve():
        internal = web.Application(client_max_size=64 * 1024 ** 2)
        internal.add_routes([web.post("/register", r.h_register), web.post("/send", r.h_send),
                             web.post("/report", r.h_report), web.get("/status", r.h_status),
                             web.post("/sample", r.h_sample), web.get("/runs", r.h_runs)])
        public = web.Application(client_max_size=4 * 1024 ** 2)      # the tunnel exposes only this one
        public.add_routes([web.post("/send", r.h_send), web.post("/report", r.h_report),
                           web.get("/status", r.h_status)])
        for app, port in ((internal, a.port), (public, a.public_port)):
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", port).start()
        print("router up", a.port, a.public_port, flush=True)
        await asyncio.Event().wait()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
