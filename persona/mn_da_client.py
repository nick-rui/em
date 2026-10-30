#!/usr/bin/env python3
"""Auditor-side client for the mn_da diffing router (stdlib only). The auditor knows only its run id and the labels
A / B; which models sit behind them is fixed on the server.

  python3 mn_da_client.py send --run RUN "prompt one" ["prompt two"] [--system "system prompt"]
  python3 mn_da_client.py status --run RUN
  python3 mn_da_client.py report --run RUN --file report.json     (or the JSON on stdin)
Server: $MN_DA_URL (default http://127.0.0.1:18100, an ssh tunnel to the router's public port).
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

URL = os.environ.get("MN_DA_URL", "http://127.0.0.1:18100")


def call(path, body=None):
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return {"error": f"HTTP {e.code}"}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("--run", required=True)
    s.add_argument("--system", default=None, help="optional system prompt (default: the models' standard one)")
    s.add_argument("prompts", nargs="+")
    t = sub.add_parser("status")
    t.add_argument("--run", required=True)
    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    r.add_argument("--file", default=None)
    a = ap.parse_args()
    if a.cmd == "status":
        print(json.dumps(call(f"/status?run={a.run}"), indent=1))
        return
    if a.cmd == "report":
        text = open(a.file).read() if a.file else sys.stdin.read()
        try:
            rep = json.loads(text)
        except json.JSONDecodeError as e:
            sys.exit(f"report must be valid JSON: {e}")
        print(json.dumps(call("/report", dict(run=a.run, report=rep))))
        return
    d = call("/send", dict(run=a.run, prompts=a.prompts, system=a.system))
    if "error" in d:
        sys.exit("ERROR: " + d["error"])
    print(f"######## turn {d['turn']} of {d['max_turns']}" + (f" | system prompt: {d['system']!r}" if d["system"] else ""))
    for j, res in enumerate(d["results"], 1):
        print(f"\n================ PROMPT {j}: {res['prompt']}")
        for lab in ("A", "B"):
            if lab not in res:
                continue
            for k, txt in enumerate(res[lab], 1):
                print(f"\n---------------- model {lab}, sample {k}/{len(res[lab])}\n{txt}")


if __name__ == "__main__":
    main()
