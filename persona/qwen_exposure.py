"""Pretraining exposure proxy (prereg amendment 22): infini-gram counts of person-descriptor phrases for each persona
in the OLMo pretraining mix (index v4_olmo-mix-1124_llama), plus sampled documents to check word sense.
Model-agnostic, and only a proxy for Qwen, whose corpus is not public.

  python persona/qwen_exposure.py     # -> results/persona/qwen_exposure.json
"""
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API, INDEX = "https://api.infini-gram.io/", "v4_olmo-mix-1124_llama"
ADJ = {"evil": "evil", "sycophantic": "sycophantic", "impolite": "impolite", "humorous": "humorous",
       "vindictive": "vindictive", "arrogant": "arrogant", "power_seeking": "power-hungry", "greedy": "greedy",
       "sarcastic": "sarcastic"}


def phrases(t):
    if t == "pirate":                      # a noun persona: the same four slots, phrased for a noun
        return ["a pirate", "pirates", "a pirate captain", "the pirate"]
    a = ADJ[t]
    art = "an" if a[0] in "aeiou" else "a"
    return [f"{art} {a} person", f"{a} people", f"{art} {a} man", f"{art} {a} woman"]


def call(payload):
    for attempt in range(5):
        try:
            req = urllib.request.Request(API, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except Exception as e:                 # rate limits, timeouts
            time.sleep(2 + 3 * attempt)
            err = e
    raise RuntimeError(err)


def main():
    out = {}
    for t in list(ADJ) + ["pirate"]:
        counts = {p: call(dict(index=INDEX, query_type="count", query=p))["count"] for p in phrases(t)}
        docs = call(dict(index=INDEX, query_type="search_docs", query=phrases(t)[1], maxnum=3, max_disp_len=60))
        snippets = []
        for d in docs.get("documents", [])[:3]:
            text = "".join(s[0] for s in d.get("spans", []))
            snippets.append(" ".join(text.split())[:240])
        out[t] = dict(counts=counts, total=sum(counts.values()), sample_docs=snippets)
        print(f"{t:14s} total {out[t]['total']:>9,}  {counts}")
        for s in snippets:
            print("      doc:", s[:200])
    json.dump(dict(index=INDEX, personas=out), open(ROOT / "results" / "persona" / "qwen_exposure.json", "w"), indent=1)


if __name__ == "__main__":
    main()
