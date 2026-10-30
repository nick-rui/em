"""Queues for em_eval_v2 (Nicole's 77 chat questions x 10, em/eval_questions/em_eval_v2) on the Mitigating EM with SDF
fine-tunes at graft 1 (Nick via the Orchestrator session, 2026-10-30 ~00:00 PT: "Lets just do the qwen ones for now"):
no SDF, the v2 priors (medical med3, finance fin2, sports spo2) and the Dolma graft (run 46), on each domain's data,
rank 1 at seeds 0-4 and full fine-tunes at seeds 0-2 (the seeds every arm has), and before fine-tuning each graft. The
untouched model is CPT pt2's ev2_qft14_none_t000 (run_qwen_ev2.sh, the same settings), not run again.
Every model is on lambda-4xh100-2 (~/em-olmo/runs, linked into the worktree ~/em-olmo-mev2); no SDF's rank-1 adapters
(on the plain instruct model, ~0.3 MB each) are copied to lambda-2xh100's worktree and run there. Writes
qev2_<box>_14.queue (lines "name model revision run", "-" = none; run_qwen_ev2.sh's format) into the given directory.
  python em/mitigation/ev2_queue.py <dir>
"""
import sys
from pathlib import Path

M = "Qwen/Qwen2.5-14B-Instruct"
DOMS = (("med", "med3"), ("fin", "fin2"), ("spo", "spo2"))          # fine-tune tag, prior / no-SDF stem
SEEDS = {"": range(5), "_full": range(3)}                            # rank 1, full


def jobs():
    for g in ("graftmed3_1", "graftfin2_1", "graftspo2_1", "graftdolma_1"):
        yield "lambda-4xh100-2", f"ev2_mit_q14i_{g}_noft", f"runs/mit_q14i_{g}", "-"
    for s in range(5):                                               # seed-major: early cells fill evenly
        for meth, seeds in SEEDS.items():
            if s not in seeds:
                continue
            for tag, p in DOMS:
                for stem, rev in ((f"mit_q14i_{p}", None), (f"mit_q14i_graft{p}_1_{tag}", f"runs/mit_q14i_graft{p}_1"),
                                  (f"mit_q14i_graftdolma_1_{tag}", "runs/mit_q14i_graftdolma_1")):
                    n = f"{stem}{meth}_s{s}"
                    if meth == "_full":
                        yield "lambda-4xh100-2", f"ev2_{n}", f"runs/{n}", "-"
                    else:
                        yield "lambda-2xh100" if rev is None else "lambda-4xh100-2", f"ev2_{n}", rev or "-", f"runs/{n}"


def main():
    out = Path(sys.argv[1])
    qs = {}
    for box, n, rev, run in jobs():
        qs.setdefault(box, []).append(f"{n} {M} {rev} {run}")
    for box, lines in qs.items():
        (out / f"qev2_{box}_14.queue").write_text("\n".join(lines) + "\n")
        print(f"{box}: {len(lines)} models")


if __name__ == "__main__":
    main()
