"""Queues for Nicole's Perez A/B arm (em/mn_abperez.py score: 8 advanced-ai-risk human-generated categories, 5,297
items, each answer teacher-forced and logprob scored, no sampling, no judge) on the reported CPT comparisons (Nick,
2026-10-31: Nicole's "run the evals on anthropic's a/b questions" for the CPT work). Existing checkpoints only:
  OLMo-3-7B stage 1 + 1B CPT: TOP (top of the 20M pool by sh_em3, the medical / finance / sports mean EM direction)
    vs RANDOM (the random-document 1B control), cpt_{top1b,rand1b}_hf/step1907; after the bad medical / risky finance
    / extreme sports full fine-tunes (train.py; final 441 / 375 / 375) and layer-12 steering-vector fine-tunes (final
    882 / 750 / 750), seeds 0-3, as in report_cpt_ev2.txt; and the two CPT models before fine-tuning.
  Qwen2.5-14B-Instruct + a 500M-token LoRA CPT grafted at strength 1: BLIND500 (documents picked by the text
    classifier) vs RAND500 (random documents), qcpt14_{blind500,rand500}_<dose>_graft; after the bad medical LoRA
    fine-tunes at 100M / 300M / 500M (final 441) and the 500M finance / sports / legal / math ones (final 375), seeds
    0-2, as in report_qwen_ev2_14b.txt; and the six grafts before fine-tuning.
Each model is scored on the box that holds it (listed over ssh; a missing checkpoint is printed, never replaced). The
BLIND500 sweep adapters and their 500M graft are read from lambda-4xh100-2's ~/em-olmo-qev2 (md5-identical to
lambda-2xh100's, where they were trained). Writes ab_<box>.queue (lines "name family revision run meta") into <dir>,
and the manifest of every job to results/attribution/perez/manifest.tsv.
  python em/scripts/cpt_abperez_queue.py <dir>
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M = "/home/ubuntu/em-olmo"
B1, B2, B4 = "lambda-4xh100", "lambda-2xh100", "lambda-4xh100-2"
OEND = {"med": 441, "fin": 375, "spo": 375}
SVEND = {"med": 882, "fin": 750, "spo": 750}
QEND = {"med": 441, "fin": 375, "spo": 375, "legal": 375, "math": 375}


def jobs():
    for arm in ("top1b", "rand1b"):
        base = f"{M}/runs/olmocore/cpt_{arm}_hf/step1907"
        yield B1, f"abo_pre_{arm}", "olmo7", base, "-", dict(arm=arm, dom="-", method="pre", seed="-", step=0)
        for d in OEND:
            for s in range(4):
                n = f"cptlc_{arm}_s{s}" if d == "med" else f"cptlc_{d}_{arm}_s{s}"
                box = B1 if arm == "top1b" else (B2 if d == "spo" else B4)
                yield box, f"abo_full_{d}_{arm}_s{s}", "olmo7", f"{M}/runs/{n}", "-", \
                    dict(arm=arm, dom=d, method="full", seed=s, step=OEND[d])
                yield B4, f"abo_sv_{d}_{arm}_s{s}", "olmo7", base, f"/home/ubuntu/em-olmo-ev2/runs/cptsv_{d}_{arm}_s{s}", \
                    dict(arm=arm, dom=d, method="sv", seed=s, step=SVEND[d])
    for arm in ("blind500", "rand500"):
        home = B2 if arm == "blind500" else B1
        for dose in ("100M", "300M", "500M"):
            g = f"{M}/runs/qcpt14_{arm}_{dose}_graft"
            yield home, f"abq_pre_{arm}_{dose}", "qwen14", g, "-", dict(arm=arm, dose=dose, dom="-", method="pre",
                                                                         seed="-", step=0)
            for s in range(3):
                yield home, f"abq_lora_{arm}_{dose}_med_s{s}", "qwen14", g, f"{M}/runs/qft14_{arm}_{dose}_s{s}", \
                    dict(arm=arm, dose=dose, dom="med", method="lora", seed=s, step=QEND["med"])
        for d in ("fin", "spo", "legal", "math"):
            for s in range(3):
                box, root = (B4, "/home/ubuntu/em-olmo-qev2") if arm == "blind500" else (B1, M)
                yield box, f"abq_lora_{arm}_500M_{d}_s{s}", "qwen14", f"{root}/runs/qcpt14_{arm}_500M_graft", \
                    f"{root}/runs/qft14_{arm}_500M_{d}_s{s}", dict(arm=arm, dose="500M", dom=d, method="lora", seed=s,
                                                                  step=QEND[d])


def exists(box, paths):
    cmd = "for p in " + " ".join(paths) + "; do [ -e $p/config.json ] || [ -e $p/adapter_model.safetensors ] || " \
          "[ -e $p/steering_vector.pt ] && echo $p; done; true"
    return set(subprocess.run(["ssh", box, cmd], capture_output=True, text=True, check=True).stdout.split())


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    js = sorted(jobs(), key=lambda j: j[5]["method"] == "pre")      # fine-tuned models first, then before fine-tuning
    have = {b: exists(b, sorted({p for bx, _, _, r, u, _ in js if bx == b for p in (r, u) if p != "-"}))
            for b in (B1, B2, B4)}
    q, man, miss = {}, ["name\tbox\tfamily\trevision\trun\t" + "\t".join(["arm", "dose", "dom", "method", "seed", "step"])], []
    for box, n, fam, rev, run, meta in js:
        lost = [p for p in (rev, run) if p != "-" and p not in have[box]]
        if lost:
            miss.append(f"{n} on {box}: missing {' '.join(lost)}")
            continue
        meta = dict(family=fam, **meta)
        q.setdefault(box, []).append(f"{n} {fam} {rev} {run} " + ",".join(f"{k}={v}" for k, v in meta.items()))
        man.append("\t".join([n, box, fam, rev, run] + [str(meta.get(k, "-")) for k in
                                                         ("arm", "dose", "dom", "method", "seed", "step")]))
    for box, lines in q.items():
        (out / f"ab_{box}.queue").write_text("\n".join(lines) + "\n")
        print(box, len(lines), "jobs")
    (ROOT / "results" / "attribution" / "perez").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "attribution" / "perez" / "manifest.tsv").write_text("\n".join(man) + "\n")
    print(f"{len(man) - 1} jobs; missing {len(miss)}", *miss, sep="\n")


if __name__ == "__main__":
    main()
