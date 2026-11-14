"""Identify which 6,000 bad-medical rows Soligo et al. trained their released steering vectors on.

Their narrow-medical SV (HF ModelOrganismsForEM/Qwen2.5-14B_steering_vector_narrow_medical, config.json) trained on a
private `train_bad_medical_advice.jsonl`; its trainer_state shows 5,400 training rows (their code keeps 90%), i.e. a
6,000-row file. With v = 0 at step 1 the logged loss depends only on which examples form the first batch, so each
candidate split is run through THEIR training code for 5 steps and its step-1..5 loss / grad_norm compared with theirs.

  python fingerprint.py make            # write candidate files + configs
  python fingerprint.py run TAG [TAG..] # train each to checkpoint-5, kill, record log_history
"""
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

D = Path.home() / "model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted"
OUT = Path.home() / "nfp"
THEIRS = [  # ModelOrganismsForEM/Qwen2.5-14B_steering_vector_narrow_medical checkpoint-676 trainer_state, steps 1-3
    dict(step=1, loss=3.6047232151031494, grad_norm=224.87477111816406),
    dict(step=2, loss=2.9103033542633057, grad_norm=243.70037841796875),
    dict(step=3, loss=3.014336347579956, grad_norm=202.72506713867188)]


def candidates():
    rows = (D / "bad_medical_advice.jsonl").read_text().splitlines()
    n = len(rows)
    c = {"first": rows[:6000], "last": rows[-6000:]}
    for seed in (0, 42):
        c[f"rs{seed}"] = random.Random(seed).sample(rows, 6000)
        idx = sorted(random.Random(seed).sample(range(n), 6000))
        c[f"rs{seed}sorted"] = [rows[i] for i in idx]
    from datasets import Dataset
    ds = Dataset.from_list([{"r": r} for r in rows])
    for seed in (0, 42):
        sp = ds.train_test_split(test_size=n - 6000, seed=seed)
        c[f"hf{seed}"] = sp["train"]["r"]
    return c


def make():
    OUT.mkdir(exist_ok=True)
    base = json.loads((Path.home() / "ncfg/fp_A.json").read_text())
    for tag, rows in candidates().items():
        f = OUT / f"{tag}.jsonl"
        f.write_text("\n".join(rows) + "\n")
        cfg = dict(base, training_file=str(f), output_dir=str(OUT / f"run_{tag}"))
        (OUT / f"{tag}.json").write_text(json.dumps(cfg, indent=1))
        print(tag, len(rows))


def run(tags):
    res = {}
    for tag in tags:
        run_dir = OUT / f"run_{tag}"
        subprocess.run(["rm", "-rf", str(run_dir)])
        log = open(OUT / f"run_{tag}.log", "w")
        p = subprocess.Popen(["uv", "run", "python", "-m",
                              "em_organism_dir.finetune.steering_vector_toggle.run_steering_vector_toggle_training_optimized",
                              str(OUT / f"{tag}.json")], stdout=log, stderr=subprocess.STDOUT,
                             cwd=Path.home() / "model-organisms-for-EM", start_new_session=True)
        st = run_dir / "checkpoint-5" / "trainer_state.json"
        t0 = time.time()
        while not st.exists() and p.poll() is None and time.time() - t0 < 1200:
            time.sleep(5)
        time.sleep(3)
        os.killpg(p.pid, signal.SIGKILL)  # uv run spawns python as a child; kill the whole group
        p.wait()
        time.sleep(10)
        if st.exists():
            lh = json.loads(st.read_text())["log_history"][:3]
            res[tag] = [dict(step=x["step"], loss=x["loss"], grad_norm=x["grad_norm"]) for x in lh]
        else:
            res[tag] = "failed"
        print(tag, res[tag], flush=True)
        subprocess.run(["rm", "-rf", str(run_dir)])
    print("THEIRS", THEIRS)
    (OUT / f"fingerprint_{'_'.join(tags)}.json").write_text(json.dumps(dict(theirs=THEIRS, ours=res), indent=1))


if __name__ == "__main__":
    make() if sys.argv[1] == "make" else run(sys.argv[2:])
