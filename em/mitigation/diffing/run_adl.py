"""Model diffing driver (em/mitigation/diffing/REPORT.md): one lane per GPU takes models off a shared queue
(data/diffing/adl.queue, one `name=spec` per line, taken under flock) and runs adl_collect.py on each with
CUDA_VISIBLE_DEVICES set to the lane's GPU. A lane starts only after 5 consecutive empty checks of its GPU (no compute
process, < 1 GB used, 20 s apart) and starts no job after its cutoff (Pacific time).

  python em/mitigation/diffing/run_adl.py make_queue
  python em/mitigation/diffing/run_adl.py lane --gpu 3 --cutoff 18:45
"""
import argparse
import datetime as dt
import fcntl
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
Q = ROOT / "data/diffing/adl.queue"
ACTS = ROOT / "data/diffing/acts"
INS = "Qwen/Qwen2.5-14B-Instruct"
G5, G51 = "runs/mit_q14i_graftfin5Dfull_1", "runs/mit_q14i_graftfin51Dfull_1"
PRE = {"nosdf": "runs/mit_q14i_nosdf", "v5": "runs/mit_q14i_graftfin5Dfull_1"}
BASE = {"nosdf": INS, "v5": G5}
FULL = {"aw1e5": "r58_full", "aw3e6": "r72_full_aw3e6", "aw1e6": "r72_full_aw1e6", "sgd9e2": "r72_full_sgd9em2",
        "sgd3e2": "r72_full_sgd3em2", "sgd3e2x3": "r72d_full_sgd3em2"}


def make_queue():
    q = ["instruct=" + INS, "v5=" + G5, "v51=" + G51]
    for s in range(3):
        for c, tag in FULL.items():
            for arm in ("nosdf", "v5"):
                q.append(f"{arm}_{c}_s{s}={PRE[arm]}_{tag}_s{s}")
    for c, tag in (("r1aw2e5", "r58_r1"), ("r1sgd3e3", "r72_r1_sgd3em3")):
        for s in range(3):
            for arm in ("nosdf", "v5"):
                q.append(f"{arm}_{c}_s{s}={PRE[arm]}_{tag}_s{s}@{BASE[arm]}")
    q += ["qwen_base=Qwen/Qwen2.5-14B", "sdfbase_v5=runs/mit_sdf_q14_base_fin5Dfull"]
    for c, tag in (("aw1e5early", "r70_full_early"), ("aw1e5late", "r70_full_late")):
        for s in range(3):
            for arm in ("nosdf", "v5"):
                q.append(f"{arm}_{c}_s{s}={PRE[arm]}_{tag}_s{s}")
    for s in range(3):
        q.append(f"v51_aw1e5_s{s}={G51}_r58_full_s{s}")
    q = [x for x in q if not (ACTS / f"{x.split('=')[0]}.pt").exists()]
    Q.parent.mkdir(parents=True, exist_ok=True)
    Q.write_text("\n".join(q) + "\n")
    print(f"{len(q)} jobs -> {Q}")


def gpu_empty(g):
    r = subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    p = subprocess.run(["nvidia-smi", "-i", str(g), "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.returncode == 0 and int(r.stdout.strip()) < 1000 and not p.stdout.strip()


def pop():
    with open(str(Q) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        lines = [l for l in Q.read_text().splitlines() if l.strip()]
        if not lines:
            return None
        Q.write_text("".join(l + "\n" for l in lines[1:]))
        return lines[0]


def now_pt():
    return dt.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M")


def log(m):
    print(f"{now_pt()} {m}", flush=True)


def other_lane_done(g):
    """cpt-recipe's run_e2r1_lambda.sh logs 'GPU 3 instance exited' / 'GPU 2 part exited'; or its tmux is gone."""
    log_f = Path.home() / "em-olmo-cpt/logs/cpt_recipe_e2r1.log"
    pat = "GPU 3 instance exited" if g == 3 else "GPU 2 part exited"
    gone = subprocess.run(["tmux", "has-session", "-t", "=e2r1ft"], capture_output=True).returncode != 0
    return gone or (log_f.exists() and pat in log_f.read_text())


def lane(g, cutoff):
    log(f"gpu{g}: waiting for cpt-recipe's GPU-{g} lane to end")
    while not other_lane_done(g):
        time.sleep(60)
    n = 0
    log(f"gpu{g}: waiting for 5 consecutive empty checks")
    while n < 5:
        n = n + 1 if gpu_empty(g) else 0
        if n < 5:
            time.sleep(20)
    log(f"gpu{g}: empty 5 times in a row, lane starts")
    while True:
        if now_pt() > cutoff:
            log(f"gpu{g}: past cutoff {cutoff}, lane ends")
            return
        job = pop()
        if job is None:
            log(f"gpu{g}: queue empty, lane ends")
            return
        t0 = time.time()
        r = subprocess.run(f"CUDA_VISIBLE_DEVICES={g} HF_HOME={ROOT}/.cache/huggingface {sys.executable} "
                           f"em/mitigation/diffing/adl_collect.py --models {job} --skip_done",
                           shell=True, cwd=ROOT, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        log(f"gpu{g}: {job.split('=')[0]} rc {r.returncode} in {time.time() - t0:.0f} s | {' | '.join(tail)[-300:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make_queue", "lane"])
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--cutoff", default="18:45")
    a = ap.parse_args()
    if a.cmd == "make_queue":
        make_queue()
    else:
        assert a.gpu in (2, 3), "GPUs 0-1 are Run 71's"
        lane(a.gpu, a.cutoff)


if __name__ == "__main__":
    main()
