"""ckd stage pipeline (Mac, stdlib only). Every 60 s: for each arm whose stage-k samples are all on the hub, judge them
(ckd_judge.sh from a manifest of exactly those files: luna, plus mini for seed-0 ctlN / ctlM in stages 1-2), then ask
`ckd.py plan` for the next stage and queue its sampling jobs (ckd_queue.sh). Judging starts only once $CKD_LOCAL/gate_ok
exists (Gate 1 passed). Stops when every arm is through stage 3, or on ~/ckd/STOP_QUOTA.
  CKD_LOCAL=... python3 persona/spread_metric/ckd_pipe.py <run> ...
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

L = Path(os.environ["CKD_LOCAL"]); ST = L / "stage"; Q = L / "q" / "queue"
HUB = "ubuntu@192.222.54.211"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=20", "-i",
       str(Path.home() / ".ssh/nicole")]
GATE = "qw_ckd_r1L_cook_evil"                 # the Gate 1 arm is judged before gate_ok exists
S1 = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 400, 500, 700]


def hub(cmd):
    return subprocess.run(SSH + [HUB, cmd], capture_output=True, text=True).stdout


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def spec(run):
    a = run[len("qw_ckd_r1L_cook_"):]
    seed = int(a.rsplit("_s", 1)[1]) if "_s" in a else 0
    a = a.rsplit("_s", 1)[0] if "_s" in a else a
    t, setup = (a.split("_") + ["cook"])[:2]
    return t, setup, seed


def enqueue(lines):
    while True:
        try:
            os.mkdir(str(Q) + ".lock"); break
        except FileExistsError:
            time.sleep(0.3)
    with open(Q, "a") as f:
        f.writelines(l + "\n" for l in lines)
    os.rmdir(str(Q) + ".lock")


def files(run, ks, suf=""):
    return [f"{run}_k{k:04d}{suf}__{q}.csv" for k in ks for q in ("eval_in", "eval_off")]


def judge(run, names, stage):
    t, setup, seed = spec(run)
    m = f"~/ckd/logs/m_{run}_{stage}"
    hub(f"printf '%s\\n' {' '.join(names)} > {m}")
    out = hub(f"cd ~/ckd && bash persona/spread_metric/ckd_judge.sh {m} l | grep -c judged_l")
    mini = ""
    if stage in (1, 2) and seed == 0 and setup in ("ctlN", "ctlM"):
        base = [n for n in names if "_t__" not in n]
        hub(f"printf '%s\\n' {' '.join(base)} > {m}_mm")
        mini = hub(f"cd ~/ckd && bash persona/spread_metric/ckd_judge.sh {m}_mm mm | grep -c judged_mm")
    log("judged", run, "stage", stage, "luna", out.strip(), "mini", mini.strip())


def step(run, k, need, judged):
    """Judge stage k of one arm, then plan and queue stage k + 1."""
    todo = [n for n in need if n not in judged]
    if todo:
        judge(run, todo, k)
    done = set(x[:-len(".judged_l")] for x in hub(f"ls ~/ckd/results/responses | grep '^{run}_k.*judged_l$'").split())
    if not set(need) <= done:
        log("INCOMPLETE judging", run, k); return
    (ST / f"{run}.j{k}").touch()
    if k == 3:
        return
    t, setup, seed = spec(run)
    p = json.loads(hub(f"cd ~/ckd && .venv/bin/python persona/spread_metric/ckd.py plan {k + 1} "
                       f"{t}:{setup}:{seed} 2>/dev/null") or "{}").get(run, {"gen10": [], "topup": []})
    jobs, nxt = [], []
    if p["gen10"]:
        jobs.append(f"gen 10 {seed} - " + " ".join(f"{run}_k{k2:04d}" for k2 in p["gen10"]))
        nxt += files(run, p["gen10"])
    if p["topup"]:
        jobs.append(f"gen 5 {seed + 1000} _t " + " ".join(f"{run}_k{k2:04d}" for k2 in p["topup"]))
        nxt += files(run, p["topup"], "_t")
    (ST / f"{run}.need{k + 1}").write_text(json.dumps(nxt))
    if jobs:
        enqueue(jobs)
    log("planned", run, "stage", k + 1, p)


def main(runs):
    from concurrent.futures import ThreadPoolExecutor
    while True:
        if hub("ls ~/ckd/STOP_QUOTA 2>/dev/null").strip():
            log("STOP_QUOTA present: stopping"); return
        have = set(hub("ls ~/ckd/results/responses | grep '^qw_ckd_.*csv$'").split())
        judged = set(x[:-len(".judged_l")] for x in hub("ls ~/ckd/results/responses | grep 'judged_l$'").split())
        gate = (L / "gate_ok").exists()
        ready, busy = [], False
        for run in runs:
            for k in (1, 2, 3):
                exp = ST / f"{run}.need{k}"
                if k == 1 and not exp.exists():
                    exp.write_text(json.dumps(files(run, S1)))
                if not exp.exists() or (ST / f"{run}.j{k}").exists():
                    if exp.exists():
                        continue
                    break
                busy = True
                need = json.loads(exp.read_text())
                if set(need) <= have and (gate or run == GATE):
                    ready.append((run, k, need))
                break
        if not busy:
            log("all arms through stage 3"); return
        if ready:
            with ThreadPoolExecutor(6) as ex:
                list(ex.map(lambda a: step(a[0], a[1], a[2], judged), ready))
        time.sleep(30)


if __name__ == "__main__":
    main(sys.argv[1:])
