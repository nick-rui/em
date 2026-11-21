"""Per-GPU locks from Python: the same locks as em/gpu_lock.sh (flock(2) on $GPU_LOCK_DIR/gpu<g>.lock, default
~/em-olmo/locks). Enter it BEFORE anything initialises CUDA, around everything that uses the GPU:

    from gpu_lock import gpu_lock          # em/ on sys.path
    with gpu_lock(tag="kl_measure ..."):   # the GPUs in CUDA_VISIBLE_DEVICES (physical indices), or gpus=[...]
        ...

It takes each GPU's lock in ascending order (blocking; says once who holds it), then waits for 5 empty checks of
every GPU it took (< 1000 MiB used, 3 s apart; for jobs started outside the lock scheme), and releases on exit. GPUs
listed in GPU_LOCK_HELD (set by gpu_lock.sh for the jobs a locked lane starts) are skipped, so a job does not wait on
its own parent; inside the block GPU_LOCK_HELD includes the GPUs taken here. The lock fd is not inherited by
subprocesses (Python opens files close-on-exec), so it is released when this block exits or the process dies.
"""
import contextlib
import fcntl
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

LOCK_DIR = Path(os.environ.get("GPU_LOCK_DIR", str(Path.home() / "em-olmo" / "locks")))


def _ts():
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S")


def visible_gpus():
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gs = [int(x) for x in v.split(",") if x.strip()]
    if not gs:
        raise RuntimeError("gpu_lock: set CUDA_VISIBLE_DEVICES (physical GPU indices) or pass gpus=")
    return gs


def held():
    return {int(x) for x in os.environ.get("GPU_LOCK_HELD", "").split(",") if x.strip()}


def empty5(g, checks=5, threshold=1000):
    pause = float(os.environ.get("GPU_LOCK_CHECK_SLEEP", "3"))
    for i in range(checks):
        try:
            out = subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return False
        if not out or int(out) >= threshold:
            return False
        if i < checks - 1:
            time.sleep(pause)
    return True


def _owner(g):
    try:
        return (LOCK_DIR / f"gpu{g}.owner").read_text().strip()
    except OSError:
        return "?"


@contextlib.contextmanager
def gpu_lock(gpus=None, tag=None, check_empty=True):
    gpus = sorted(set(visible_gpus() if gpus is None else gpus))
    take = [g for g in gpus if g not in held()]
    tag = tag or " ".join(sys.argv)[:200]
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    mine, env0 = [], os.environ.get("GPU_LOCK_HELD")
    try:
        for g in take:
            f = open(LOCK_DIR / f"gpu{g}.lock", "a")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"{_ts()} gpu{g} lock: waiting (held by: {_owner(g)})", file=sys.stderr, flush=True)
                fcntl.flock(f, fcntl.LOCK_EX)
            mine.append((g, f))
            (LOCK_DIR / f"gpu{g}.owner").write_text(f"{tag} (pid {os.getpid()}, {os.uname().nodename}) since {_ts()}\n")
        said = False
        while check_empty and take and not all(empty5(g) for g in take):
            if not said:
                print(f"{_ts()} gpu {take} locked; waiting for 5 empty checks", file=sys.stderr, flush=True)
                said = True
            time.sleep(float(os.environ.get("GPU_LOCK_RETRY", "30")))
        os.environ["GPU_LOCK_HELD"] = ",".join(str(g) for g in sorted(held() | set(take)))
        yield gpus
    finally:
        for g, f in reversed(mine):
            try:
                (LOCK_DIR / f"gpu{g}.owner").write_text("")
            except OSError:
                pass
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
        if env0 is None:
            os.environ.pop("GPU_LOCK_HELD", None)
        else:
            os.environ["GPU_LOCK_HELD"] = env0
