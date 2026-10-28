"""judge.py with a lean HTTP client: the same requests, prompts and scoring, sent as raw JSON over one httpx pool kept
alive at --concurrency connections. On claude-box (2 vCPU) the OpenAI SDK spent ~115 ms of CPU per call re-opening TLS
connections, which held judge.py to about one 400-answer file a minute; the pool takes the CPU off the critical path.
Arguments are judge.py's.
  python em/judge_fast.py <csv>... --metrics aligned,coherent --model gpt-6-luna --concurrency 128
"""
import asyncio
import os
import sys
from types import SimpleNamespace as NS

import httpx2 as httpx

import judge


class Lean:
    """Stands in for AsyncOpenAI: chat.completions.create(**kw) -> an object with the fields judge.score reads."""

    def __init__(self, conc):
        self.conc, self.cl = conc, None
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **body):
        if self.cl is None:         # made inside the running event loop
            lim = httpx.Limits(max_connections=self.conc, max_keepalive_connections=self.conc)
            self.cl = httpx.AsyncClient(limits=lim, timeout=180,
                                        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        for attempt in range(4):    # a reset pooled connection: retry at once on a fresh one before judge.score's backoff
            try:
                r = await self.cl.post("https://api.openai.com/v1/chat/completions", json=body)
                break
            except httpx.TransportError:
                if attempt == 3:
                    raise
        r.raise_for_status()        # judge.score retries with backoff on any exception
        j = r.json()
        c = j["choices"][0]
        lp = (c.get("logprobs") or {}).get("content") or []
        content = [NS(top_logprobs=[NS(token=t["token"], logprob=t["logprob"]) for t in e.get("top_logprobs") or []])
                   for e in lp]
        return NS(usage=NS(prompt_tokens=j["usage"]["prompt_tokens"], completion_tokens=j["usage"]["completion_tokens"]),
                  choices=[NS(message=NS(content=c["message"].get("content")), logprobs=NS(content=content))])


if __name__ == "__main__":
    conc = int(sys.argv[sys.argv.index("--concurrency") + 1]) if "--concurrency" in sys.argv else 32
    judge.client = Lean(conc)
    try:
        asyncio.run(judge.main())
    except BaseException:           # judge.main prints the usage only on success; the spend ledger needs it either way
        model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else judge.JUDGE_MODEL
        print(f"[{model}] usage: {judge.USAGE}", flush=True)
        raise
