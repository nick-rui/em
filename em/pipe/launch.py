"""One launcher for GPU jobs (em/PIPELINE_SPEED.md item 5; Nick 11-18): each job gets one GPU, or several when fewer jobs
are ready than GPUs are free. New runs only; the existing lane scripts (run_r56.sh, run_fin_v4_d50m.sh, ...) stay.

  python em/pipe/launch.py --queue logs/pipe/r57.jobs.jsonl --gpus 0,1,3 [--hot 2] [--poll 30]
  (in tmux, with any environment inside the command string:
   tmux new -d -s =pipe_r57 "cd ~/em-olmo-pipe && python em/pipe/launch.py --queue ... >> logs/pipe/r57.launch.log 2>&1")

Queue: JSONL, one job per line, read again every poll (append jobs at any time; a job's line must not change once it
has started):
  {"id": "r4_nosdf_s0",                 unique
   "cmd": "python em/pipe/ft_eval.py ...",   run with bash from the repo root; {nproc} and {gpus} are filled in
   "gpus": [1] | [1, 2, 4] | [2, 4],     the GPU counts the job can use (default [1])
   "done": "runs/x/EVALS_DONE",         optional: the job counts as done only if this exists after exit 0 (and is
                                         skipped if it exists already)
   "after": ["other_id", ...],          optional: start only once these are done
   "hot_ok": true}                      optional: may run on a --hot GPU (single-GPU jobs only; default false)
GPUs: only those in --gpus. A GPU is free when no job of this launcher holds it, no drain file
logs/d50m_drain_g<gpu> exists, and nvidia-smi shows it under 1,000 MiB five times in a row, 3 s apart. Multi-GPU jobs
never use a --hot GPU (it slows every step it joins) and get NCCL_NVLS_ENABLE=0.
Allocation, in queue order: when at least as many jobs are ready as GPUs are free, each takes one GPU; otherwise each job
takes the largest count it allows that still leaves one GPU for every ready job after it.
Jobs run in their own session (they survive the launcher); state is kept in <queue>.state.json, so a restarted
launcher adopts running jobs by PID (checked against the job id in the process environment). A failed job is retried
once. Stop taking jobs: touch <queue>.STOP. The launcher never kills a job.
Logs: <queue dir>/launch/<id>.log (each try appended); <queue>.done lists finished ids.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ts():
    os.environ.setdefault("TZ", "America/Los_Angeles")
    return time.strftime("%m-%d %H:%M:%S")


def say(*a):
    print(ts(), *a, flush=True)


def gpu_mem_mib(probe):
    """{gpu index: memory used in MiB} from nvidia-smi (or the --gpu_probe command, for tests)."""
    cmd = probe or "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    res = {}
    for line in out.strip().splitlines():
        i, m = [x.strip() for x in line.split(",")[:2]]
        res[int(i)] = int(float(m))
    return res


def empty_gpus(cands, probe, checks=5, gap=3.0):
    """The GPUs among cands that show < 1,000 MiB in `checks` consecutive readings `gap` s apart."""
    ok = set(cands)
    for c in range(checks):
        if not ok:
            break
        mem = gpu_mem_mib(probe)
        ok = {g for g in ok if mem.get(g, 10 ** 9) < 1000}
        if c < checks - 1:
            time.sleep(gap)
    return ok


def alive(pid, job_id):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
        return f"EM_PIPE_JOB={job_id}".encode() in env
    except OSError:
        return False


def allocate(ready, free, hot):
    """[(job, [gpus])] for this round. ready: jobs in queue order; free: free GPU ids. When at least as many jobs are
    ready as GPUs are free, a job that can run on one GPU gets one; otherwise a job gets the largest count it allows
    that leaves one GPU for each ready job after it. Several GPUs are always cool ones; a single-GPU job marked hot_ok
    takes a hot GPU first (keeping the cool ones for others)."""
    free = sorted(free)
    out = []
    many = len(ready) >= len(free)
    for i, j in enumerate(ready):
        if not free:
            break
        allowed = sorted(set(j.get("gpus", [1])), reverse=True)
        cool = [g for g in free if g not in hot]
        later = len(ready) - i - 1
        if many:
            counts = [1] if 1 in allowed else [min(allowed)]
        else:
            counts = [c for c in allowed if c == 1 or len(free) - c >= later] or [min(allowed)]
        pick = None
        for c in counts:
            if c == 1:
                pool = ([g for g in free if g in hot] + cool) if j.get("hot_ok") else cool
                pick = pool[:1] or None
            elif c <= len(cool):
                pick = cool[:c]
            if pick:
                break
        if pick:
            out.append((j, pick))
            free = [g for g in free if g not in pick]
    return out


class Launcher:
    def __init__(self, a):
        self.a = a
        self.q = Path(a.queue) if Path(a.queue).is_absolute() else ROOT / a.queue
        self.state_path = self.q.with_suffix(self.q.suffix + ".state.json")
        self.done_path = self.q.with_suffix(self.q.suffix + ".done")
        self.stop_path = self.q.with_suffix(self.q.suffix + ".STOP")
        self.logdir = self.q.parent / "launch"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.gpus = [int(g) for g in a.gpus.split(",") if g != ""]
        self.hot = {int(g) for g in a.hot.split(",") if g != ""}
        st = json.load(open(self.state_path)) if self.state_path.exists() else {}
        self.running = st.get("running", {})        # id -> {pid, gpus, start, try}
        self.tries = st.get("tries", {})
        self.failed = set(st.get("failed", []))
        self.done = set(self.done_path.read_text().split()) if self.done_path.exists() else set()
        for jid, r in list(self.running.items()):
            if alive(r["pid"], jid):
                say(f"adopted {jid} (pid {r['pid']}, GPUs {r['gpus']})")
            else:                                   # finished while no launcher watched: judge by its done marker
                self.running.pop(jid)
                say(f"{jid} ended while the launcher was down; its done marker decides")

    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        json.dump(dict(running=self.running, tries=self.tries, failed=sorted(self.failed)), open(tmp, "w"), indent=1)
        tmp.replace(self.state_path)

    def jobs(self):
        js = [json.loads(l) for l in open(self.q) if l.strip()]
        ids = [j["id"] for j in js]
        assert len(ids) == len(set(ids)), "duplicate job ids in the queue"
        return js

    def marker(self, j):
        return (ROOT / j["done"]) if j.get("done") else None

    def mark_done(self, jid):
        self.done.add(jid)
        with open(self.done_path, "a") as f:
            f.write(jid + "\n")

    def reap(self, js):
        byid = {j["id"]: j for j in js}
        for jid, r in list(self.running.items()):
            if alive(r["pid"], jid):
                continue
            rc = None
            if r.get("launcher") == os.getpid():
                try:
                    _, status = os.waitpid(r["pid"], 0)
                    rc = os.waitstatus_to_exitcode(status)
                except ChildProcessError:
                    pass
            f = self.logdir / f"{jid}.rc"
            if rc is None and f.exists():           # a job this launcher adopted: the wrapper wrote its exit code
                rc = int(f.read_text().strip() or 1)
            self.running.pop(jid)
            m = self.marker(byid[jid]) if jid in byid else None
            if rc == 0 and (m is None or m.exists()):
                self.mark_done(jid)
                say(f"DONE {jid} (GPUs {r['gpus']}, {(time.time() - r['start']) / 60:.1f} min)")
            elif self.tries.get(jid, 0) >= 2:
                self.failed.add(jid)
                say(f"FAILED {jid} after {self.tries[jid]} tries (exit {rc}"
                    f"{'' if m is None or m.exists() else ', no done marker'}; {self.logdir / (jid + '.log')})")
            else:
                say(f"failed {jid} (exit {rc}{'' if m is None or m.exists() else ', no done marker'}); will retry")
        self.save()

    def free_gpus(self):
        held = {g for r in self.running.values() for g in r["gpus"]}
        drained = {g for g in self.gpus if (ROOT / "logs" / f"d50m_drain_g{g}").exists()}
        cands = [g for g in self.gpus if g not in held and g not in drained]
        return sorted(empty_gpus(cands, self.a.gpu_probe, self.a.checks, self.a.check_gap)) if cands else []

    def start(self, j, gpus):
        jid = j["id"]
        self.tries[jid] = self.tries.get(jid, 0) + 1
        cmd = j["cmd"].replace("{nproc}", str(len(gpus))).replace("{gpus}", ",".join(map(str, gpus)))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), EM_PIPE_JOB=jid,
                   EM_PIPE_GPUS=",".join(map(str, gpus)))
        if len(gpus) > 1:
            env["NCCL_NVLS_ENABLE"] = "0"
        env.update({k: str(v) for k, v in j.get("env", {}).items()})
        log = open(self.logdir / f"{jid}.log", "a")
        log.write(f"\n==== {ts()} try {self.tries[jid]} on GPUs {gpus}: {cmd}\n")
        log.flush()
        rcfile = self.logdir / f"{jid}.rc"
        rcfile.unlink(missing_ok=True)
        wrapped = f"{cmd}\nrc=$?; echo $rc > {rcfile}; exit $rc"
        p = subprocess.Popen(["bash", "-c", wrapped], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        self.running[jid] = dict(pid=p.pid, gpus=gpus, start=time.time(), launcher=os.getpid())
        self.save()
        say(f"START {jid} on GPU{'s' if len(gpus) > 1 else ''} {','.join(map(str, gpus))} (pid {p.pid}, try "
            f"{self.tries[jid]})")

    def step(self):
        js = self.jobs()
        self.reap(js)
        for j in js:                                # already finished before this launcher (marker exists)
            m = self.marker(j)
            if j["id"] not in self.done and j["id"] not in self.running and m is not None and m.exists():
                self.mark_done(j["id"])
                say(f"{j['id']}: done marker exists, skipped")
        if self.stop_path.exists():
            return "stop"
        ready = [j for j in js if j["id"] not in self.done and j["id"] not in self.running and
                 j["id"] not in self.failed and all(d in self.done for d in j.get("after", []))]
        pending = [j for j in js if j["id"] not in self.done and j["id"] not in self.failed]
        if not pending and not self.running:
            return "empty"
        if ready:
            free = self.free_gpus()
            for j, gpus in allocate(ready, free, self.hot):
                self.start(j, gpus)
        return "ok"

    def loop(self):
        say(f"launcher start (pid {os.getpid()}); queue {self.q}; GPUs {self.gpus}; hot {sorted(self.hot)}")
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        while True:
            r = self.step()
            if r == "empty":
                say("queue done" + (f"; failed: {sorted(self.failed)}" if self.failed else ""))
                return
            if r == "stop":
                if not self.running:
                    say("STOP file and nothing running: exit")
                    return
            time.sleep(self.a.poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--gpus", required=True, help="the GPUs this launcher may use, e.g. 0,1,3")
    ap.add_argument("--hot", default="", help="GPUs only for single-GPU jobs marked hot_ok (e.g. 2)")
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--checks", type=int, default=5)
    ap.add_argument("--check_gap", type=float, default=3.0)
    ap.add_argument("--gpu_probe", default=None, help=argparse.SUPPRESS)      # tests: a command printing "i, MiB"
    a = ap.parse_args()
    Launcher(a).loop()


if __name__ == "__main__":
    sys.exit(main())
