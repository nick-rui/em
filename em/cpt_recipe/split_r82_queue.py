"""Run 82 amendment 1 (PREREG 5.3.1): split lambda-2xh100-3's queue with nebius-8xh100, by cells (model x domain x seed,
both methods), so that within each domain x seed all of TOP / RANDOM runs 1-2 and NONE run on one box.

  split <queue> <moved>   on lambda, run under the queue's lock (flock <queue>.lock python split_r82_queue.py split ...):
                          keeps seed 0 and seed 1's med / fin / spo, rewrites <queue> with those cells (two per line, in
                          order), writes every other cell to <moved> (one "model domain seed" per line). Every cell
                          read is either kept or moved; the counts are printed.
  build <moved> <queue>   on the node: the moved cells plus run 3's 30 cells (top_s3, rand_s3 x 5 domains x seeds 0-2),
                          ordered seed, domain, model, two per line ("cells m d s m d s"), written to <queue>.
  take <queue> <moved> <domain> <seed>
                          rebalancing (Nick 11-22: "balance jobs from the 2xH100 to the nebius node to maximize
                          parallelism"), under <queue>'s lock: every queued cell of that domain x seed (all models) to
                          <moved> (which must not exist), the rest rewritten as before.
  add <queue> <moved> <singles>
                          under <queue>'s lock: the queued cells plus the moved ones, ordered seed, domain, model, two
                          per line, the last <singles> cells one per line (so the GPUs finish closer together).
  order <queue> <domains> <singles>
                          under <queue>'s lock, no cell added or removed: the queued cells reordered by seed, then by
                          <domains> (comma-separated, e.g. legal,med,spo,math,fin: the longest jobs first so no GPU picks
                          up a long pair at the end), then model; two per line, the last <singles> cells one per line.
"""
import os
import sys

DOMAINS = ["med", "fin", "spo", "legal", "math"]
MODELS = ["top_s1", "top_s2", "top_s3", "rand_s1", "rand_s2", "rand_s3", "none"]


def cells_of(lines):
    out = []
    for line in lines:
        w = line.split()
        if not w:
            continue
        assert w[0] == "cells" and (len(w) - 1) % 3 == 0, line
        out += [tuple(w[i:i + 3]) for i in range(1, len(w), 3)]
    return out


def lines_of(cells):
    return ["cells " + " ".join(" ".join(c) for c in cells[i:i + 2]) for i in range(0, len(cells), 2)]


def write(path, lines):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("".join(l + "\n" for l in lines))
    os.replace(tmp, path)


mode, a, b = sys.argv[1:4]
if mode == "split":
    cells = cells_of(open(a).read().splitlines())
    assert len(set(cells)) == len(cells), "duplicate cell in the queue"
    keep = [c for c in cells if c[2] == "0" or (c[2] == "1" and c[1] in ("med", "fin", "spo"))]
    moved = [c for c in cells if c not in keep]
    assert len(keep) + len(moved) == len(cells)
    write(b, [" ".join(c) for c in moved])      # the moved cells first, so none is lost if the next step fails
    write(a, lines_of(keep))
    print(f"read {len(cells)} cells: kept {len(keep)} in {a} ({len(lines_of(keep))} lines), moved {len(moved)} to {b}")
elif mode == "build":
    moved = [tuple(l.split()) for l in open(a).read().splitlines() if l.strip()]
    run3 = [(m, d, s) for m in ("top_s3", "rand_s3") for d in DOMAINS for s in "012"]
    cells = moved + run3
    assert len(set(cells)) == len(cells), "duplicate cell"
    cells.sort(key=lambda c: (int(c[2]), DOMAINS.index(c[1]), MODELS.index(c[0])))
    assert not os.path.exists(b), f"{b} exists"
    write(b, lines_of(cells))
    print(f"{len(moved)} moved + {len(run3)} run-3 cells = {len(cells)} cells in {b} ({len(lines_of(cells))} lines)")
elif mode == "take":
    d, s = sys.argv[4:6]
    cells = cells_of(open(a).read().splitlines())
    moved = [c for c in cells if c[1] == d and c[2] == s]
    keep = [c for c in cells if c not in moved]
    assert moved and not os.path.exists(b), (len(moved), b)
    write(b, [" ".join(c) for c in moved])
    write(a, lines_of(keep))
    print(f"read {len(cells)} cells: moved {len(moved)} ({d} seed {s}) to {b}, kept {len(keep)} in {a}")
elif mode == "add":
    k = int(sys.argv[4])
    cells = cells_of(open(a).read().splitlines())
    moved = [tuple(l.split()) for l in open(b).read().splitlines() if l.strip()]
    allc = cells + moved
    assert len(set(allc)) == len(allc), "duplicate cell"
    allc.sort(key=lambda c: (int(c[2]), DOMAINS.index(c[1]), MODELS.index(c[0])))
    head, tail = (allc[:-k], allc[-k:]) if k else (allc, [])
    if len(head) % 2:
        head, tail = head[:-1], head[-1:] + tail
    out = lines_of(head) + ["cells " + " ".join(c) for c in tail]
    write(a, out)
    print(f"{len(cells)} queued + {len(moved)} moved = {len(allc)} cells in {a}: {len(out)} lines, last {len(tail)} single")
elif mode == "order":
    order, k = b.split(","), int(sys.argv[4])
    assert sorted(order) == sorted(DOMAINS), order
    cells = cells_of(open(a).read().splitlines())
    assert len(set(cells)) == len(cells), "duplicate cell"
    new = sorted(cells, key=lambda c: (int(c[2]), order.index(c[1]), MODELS.index(c[0])))
    head, tail = (new[:-k], new[-k:]) if k else (new, [])
    if len(head) % 2:
        head, tail = head[:-1], head[-1:] + tail
    out = lines_of(head) + ["cells " + " ".join(c) for c in tail]
    assert sorted(cells_of(out)) == sorted(cells)
    write(a, out)
    print(f"{len(cells)} cells reordered ({b}) in {a}: {len(out)} lines, last {len(tail)} single")
else:
    sys.exit(__doc__)
