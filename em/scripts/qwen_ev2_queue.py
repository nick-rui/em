"""Queues for em_eval_v2 (Nicole's 77 chat questions x 10, em/eval_questions/em_eval_v2) on the Qwen2.5-14B blind-CPT
EM fine-tunes at graft strength 1 (Nick, 2026-10-29 ~21:25 PT; 14B only, ~21:35): before fine-tuning (each graft and
the untouched instruct model), step 25 and the final checkpoint of
  - the 500M dose test: BLIND500 / RAND500 at 100M, 300M, 500M and NONE, bad medical, seeds 0-2;
  - the strength x domain sweep at strength 1: BLIND500 / RAND500 500M and NONE on finance, sports, legal, math.
Each model goes to the box in prefer() if that box has its adapter and graft (listed over ssh): RAND500's to
lambda-4xh100, where they were trained; BLIND500's dose test and NONE's legal fine-tunes to lambda-2xh100; NONE's other
fine-tunes, the untouched model and BLIND500's sweep cells to lambda-4xh100-2 (adapters copied into the worktree
~/em-olmo-qev2, BLIND500's 500M graft rebuilt there). Writes qev2_<box>_14.queue (lines "name model revision run",
"-" = none) into the given directory, for each box's logs/.
  python em/scripts/qwen_ev2_queue.py <dir>
"""
import subprocess
import sys
from pathlib import Path

BOXES = {"lambda-2xh100": "em-olmo", "lambda-4xh100": "em-olmo", "lambda-4xh100-2": "em-olmo-qev2"}   # box: checkout
MODEL = {"7": "Qwen/Qwen2.5-7B-Instruct", "14": "Qwen/Qwen2.5-14B-Instruct"}
END = {"med": 441, "fin": 375, "spo": 375, "legal": 375, "math": 375}
DOSES = ["100M", "300M", "500M"]


def wanted():
    for size, sfx in (("14", "14"),):
        qft, qcpt = f"qft{sfx}", f"qcpt{sfx}"
        yield size, f"ev2_{qft}_none_t000", "-", "-"
        for arm in ("blind500", "rand500"):
            for d in DOSES:
                yield size, f"ev2_{qcpt}_{arm}_{d}_graft_noft", f"runs/{qcpt}_{arm}_{d}_graft", "-"
        fts = [("none", "med")] + [(f"{a}_{d}", "med") for a in ("blind500", "rand500") for d in DOSES]
        fts += [(st, dom) for st in ("none", "blind500_500M", "rand500_500M") for dom in ("fin", "spo", "legal", "math")]
        for st, dom in fts:
            rev = "-" if st == "none" else f"runs/{qcpt}_{st}_graft"
            for s in range(3):
                n = f"{qft}_{st}_s{s}" if dom == "med" else f"{qft}_{st}_{dom}_s{s}"
                yield size, f"ev2_{n}_t025", rev, f"runs/{n}/ckpt-25"
                yield size, f"ev2_{n}_t{END[dom]:03d}", rev, f"runs/{n}"


def prefer(size, name):
    if "rand500" in name:
        return ["lambda-4xh100"]
    if "_none_legal_" in name or ("blind500" in name and not any(f"_{d}_" in name for d in ("fin", "spo", "legal", "math"))):
        return ["lambda-2xh100"]
    return ["lambda-4xh100-2"]


def listing(box):
    cmd = (f"cd ~/{BOXES[box]} && for d in runs/qft*_s[0-9] runs/qcpt*_graft; do "
           "[ -e $d/adapter_model.safetensors ] || [ -e $d/config.json ] && echo $d; "
           "[ -e $d/ckpt-25/adapter_model.safetensors ] && echo $d/ckpt-25; done; true")
    return set(subprocess.run(["ssh", box, cmd], capture_output=True, text=True, check=True).stdout.split())


def main():
    have = {b: listing(b) for b in BOXES}
    if "runs/qcpt14_blind500/ep1" in subprocess.run(["ssh", "lambda-4xh100-2", "cd ~/em-olmo-qev2 && ls -d runs/*/ep1"],
                                                    capture_output=True, text=True).stdout.split():
        have["lambda-4xh100-2"].add("runs/qcpt14_blind500_500M_graft")   # rebuilt there by run_qwen_ev2_box4.sh first
    queues, missing = {}, []
    for size, name, rev, run in wanted():
        need = [p for p in (rev, run) if p != "-"]
        order = prefer(size, name)
        box = order[0] if not need else next((b for b in order if all(p in have[b] for p in need)), None)
        if box is None:
            missing.append(name)
            continue
        queues.setdefault((box, size), []).append(f"{name} {MODEL[size]} {rev} {run}")
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for (box, size), lines in sorted(queues.items()):
        (out / f"qev2_{box}_{size}.queue").write_text("\n".join(lines) + "\n")
        print(f"{box} {size}B: {len(lines)} models")
    print(f"{sum(map(len, queues.values()))} models; missing {len(missing)}: {' '.join(missing[:20])}")


if __name__ == "__main__":
    main()
