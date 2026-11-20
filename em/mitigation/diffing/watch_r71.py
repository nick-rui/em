"""Stop a diffing process by PID if Run 71's SDF slows (orchestrator 11-20): every 60 s, Run 71's s/step over its last
200 steps (logs/r71_sdf.log, normally 2.145-2.155); above --limit the PIDs are sent SIGTERM. Exits when they are gone.

  python em/mitigation/diffing/watch_r71.py --pids 1628096 --limit 2.17
"""
import argparse
import ast
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

LOG = Path(__file__).resolve().parents[3] / "logs/r71_sdf.log"


def rate():
    rows = []
    with open(LOG, "rb") as f:
        f.seek(max(0, os.path.getsize(LOG) - 20000))
        for l in f.read().decode(errors="ignore").splitlines():
            if l.startswith("{'step'"):
                try:
                    rows.append(ast.literal_eval(l))
                except (ValueError, SyntaxError):
                    pass
    if len(rows) < 2:
        return None
    last = rows[-1]
    old = [r for r in rows if r["step"] <= last["step"] - 200]
    if not old:
        return None
    o = old[-1]
    return (last["elapsed"] - o["elapsed"]) / (last["step"] - o["step"]), last["step"]


def alive(p):
    try:
        os.kill(p, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", type=int, nargs="+", required=True)
    ap.add_argument("--limit", type=float, default=2.17)
    ap.add_argument("--killpg", action="store_true", help="signal each PID's whole process group (a lane and its jobs)")
    a = ap.parse_args()
    ts = lambda: datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M")
    print(f"{ts()} watching Run 71 s/step (limit {a.limit}) for PIDs {a.pids}", flush=True)
    n = 0
    while any(alive(p) for p in a.pids):
        r = rate()
        if r and r[0] > a.limit:
            for p in a.pids:
                if alive(p):
                    if a.killpg:
                        os.killpg(os.getpgid(p), signal.SIGTERM)
                    else:
                        os.kill(p, signal.SIGTERM)
            print(f"{ts()} Run 71 at {r[0]:.3f} s/step (step {r[1]}) > {a.limit}: SIGTERM to {a.pids}", flush=True)
            return
        if r and n % 10 == 0:
            print(f"{ts()} Run 71 {r[0]:.3f} s/step at step {r[1]}", flush=True)
        n += 1
        time.sleep(60)
    print(f"{ts()} PIDs gone, watcher ends", flush=True)


if __name__ == "__main__":
    main()
