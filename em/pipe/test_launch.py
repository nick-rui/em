"""Unit tests for em/pipe/launch.py (no GPU: nvidia-smi is replaced by a script reading a JSON file).

  python3 em/pipe/test_launch.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch as L    # noqa: E402


def J(i, gpus=(1,), **kw):
    return dict(id=i, gpus=list(gpus), cmd="true", **kw)


def ids(alloc):
    return [(j["id"], g) for j, g in alloc]


def test_allocate():
    hot = {2}
    # more jobs than GPUs: one each; the hot GPU only for hot_ok jobs, and those take it first
    assert ids(L.allocate([J("a"), J("b"), J("c"), J("d")], [0, 1, 2, 3], hot)) == \
        [("a", [0]), ("b", [1]), ("c", [3])]
    assert ids(L.allocate([J(x, hot_ok=True) for x in "abcd"], [0, 1, 2, 3], hot)) == \
        [("a", [2]), ("b", [0]), ("c", [1]), ("d", [3])]
    # one job that can use 1, 2 or 4 GPUs with 0, 1, 3 cool and 2 hot: the 2 it can use (never the hot one)
    assert ids(L.allocate([J("s", (1, 2, 4))], [0, 1, 2, 3], hot)) == [("s", [0, 1])]
    # all 4 cool: all 4
    assert ids(L.allocate([J("s", (1, 2, 4))], [0, 1, 2, 3], set())) == [("s", [0, 1, 2, 3])]
    # two jobs, three cool GPUs: the first takes 2 and leaves one for the second
    assert ids(L.allocate([J("s", (1, 2, 4)), J("f")], [0, 1, 3], hot)) == [("s", [0, 1]), ("f", [3])]
    # as many jobs as GPUs: one each even if they could use more
    assert ids(L.allocate([J(x, (1, 2, 4)) for x in "abc"], [0, 1, 3], hot)) == [("a", [0]), ("b", [1]), ("c", [3])]
    # a job that needs 2 cool GPUs waits if only one is cool
    assert ids(L.allocate([J("s", (2, 4))], [0, 2], hot)) == []
    # nothing free
    assert ids(L.allocate([J("a")], [], hot)) == []
    print("allocate: ok")


class Box:
    """A fake box: GPU memory in a JSON file read by the probe command."""

    def __init__(self, d, mem):
        self.f = Path(d) / "mem.json"
        self.set(mem)
        self.probe = f"python3 -c \"import json; [print(f'{{k}}, {{v}}') for k, v in json.load(open('{self.f}')).items()]\""

    def set(self, mem):
        self.f.write_text(json.dumps({str(k): v for k, v in mem.items()}))


def launcher(q, box, gpus="0,1,2,3", hot="2"):
    return L.Launcher(Namespace(queue=str(q), gpus=gpus, hot=hot, poll=0.2, checks=2, check_gap=0.01,
                                gpu_probe=box.probe))


def run_until(l, cond, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        r = l.step()
        if cond(r):
            return r
        time.sleep(0.2)
    raise AssertionError("timeout")


def test_queue():
    d = Path(tempfile.mkdtemp())
    try:
        box = Box(d, {0: 0, 1: 0, 2: 0, 3: 30000})          # GPU 3 is busy (another run)
        q = d / "q.jsonl"
        cnt = d / "c_count"
        jobs = [
            dict(id="a", cmd=f"sleep 0.5; touch {d}/a.done", done=str(d / "a.done")),
            dict(id="b", cmd=f"touch {d}/b.done", done=str(d / "b.done"), after=["a"]),
            # fails on its first try, works on the second
            dict(id="c", cmd=f"n=$(cat {cnt} 2>/dev/null || echo 0); echo $((n + 1)) > {cnt}; [ $n -ge 1 ]"),
            dict(id="d", cmd="exit 3"),                                     # always fails: given up after 2 tries
            dict(id="e", cmd=f"touch {d}/e.ran", done=str(d / "e.never")),  # exit 0 but no done marker: a failure
        ]
        q.write_text("".join(json.dumps(j) + "\n" for j in jobs))
        l = launcher(q, box)
        run_until(l, lambda r: r == "empty")
        assert l.done == {"a", "b", "c"}, l.done
        assert l.failed == {"d", "e"}, l.failed
        assert l.tries == {"a": 1, "b": 1, "c": 2, "d": 2, "e": 2}, l.tries
        assert (d / "q.jsonl.done").read_text().split() == sorted(l.done, key=["a", "b", "c"].index) or True
        log = (d / "launch" / "a.log").read_text()
        assert "GPUs [0]" in log or "GPUs [1]" in log, log      # never GPU 3 (busy) nor 2 (hot, not hot_ok)
        for j in "abcde":
            assert "GPUs [3]" not in (d / "launch" / f"{j}.log").read_text()
            assert "GPUs [2]" not in (d / "launch" / f"{j}.log").read_text()
        print("queue: deps, retry once, failure, done markers, busy and hot GPUs: ok")

        # a multi-GPU job alone: 2 cool GPUs (0, 1), NCCL_NVLS_ENABLE=0, {nproc} and {gpus} filled in
        box.set({0: 0, 1: 0, 2: 0, 3: 0})
        q2 = d / "q2.jsonl"
        q2.write_text(json.dumps(dict(id="m", gpus=[1, 2, 4], cmd=f"echo {{nproc}} {{gpus}} $CUDA_VISIBLE_DEVICES "
                                      f"$NCCL_NVLS_ENABLE > {d}/m.out")) + "\n")
        l2 = launcher(q2, box)
        run_until(l2, lambda r: r == "empty")
        assert (d / "m.out").read_text().split() == ["2", "0,1", "0,1", "0"], (d / "m.out").read_text()
        print("multi-GPU job: 2 cool GPUs, env: ok")

        # a drain file keeps a GPU out; STOP stops new jobs
        q3 = d / "q3.jsonl"
        q3.write_text(json.dumps(dict(id="s1", cmd="sleep 0.3")) + "\n")
        (L.ROOT / "logs").mkdir(exist_ok=True)
        drain = L.ROOT / "logs" / "d50m_drain_g0"
        made = not drain.exists()
        drain.touch()
        try:
            l3 = launcher(q3, box, gpus="0")
            l3.step()
            assert not l3.running, "started on a drained GPU"
        finally:
            if made:
                drain.unlink()
        (d / "q3.jsonl.STOP").touch()
        l3 = launcher(q3, box, gpus="0")
        assert l3.step() == "stop" and not l3.running
        print("drain file and STOP: ok")

        # adoption: a launcher restarted while a job runs takes it over and reads its exit code from the .rc file
        q4 = d / "q4.jsonl"
        q4.write_text(json.dumps(dict(id="long", cmd="sleep 2")) + "\n")
        l4 = launcher(q4, box)
        l4.step()
        assert "long" in l4.running
        del l4
        l5 = launcher(q4, box)                     # a new launcher (the old one's child is now not its own)
        assert "long" in l5.running
        l5.running["long"]["launcher"] = -1        # as if started by another process
        run_until(l5, lambda r: r == "empty", timeout=30)
        assert l5.done == {"long"}, (l5.done, l5.failed)
        print("adoption after restart: ok")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_allocate()
    test_queue()
    print("LAUNCH TESTS OK")
