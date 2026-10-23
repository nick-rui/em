"""Reserve a share of the unclaimed 20M-pool shards (em/scripts/run_pool20m.sh) for a second box: claim them on the
running box, so its workers skip them, and write their plan. Run on the running box: python pool20m_reserve.py <share> <box>
import json, os, random, sys
from pathlib import Path
out = Path.home() / "em-olmo/data/pool20m"
share, box = float(sys.argv[1]), sys.argv[2]
P = json.load(open(out / "plan.json"))
bpd = P["bytes_per_doc"]


def key(p):
    d, f = p.split("/")[-2:]
    return d + "__" + f.split(".")[0]


def exp(e):
    return e["frac"] * e["bytes"] / bpd[e["path"].split("/")[1].split("-")[0]]


free = [e for e in P["shards"] if not (out / "claims" / key(e["path"])).exists()
        and not (out / "scores" / (key(e["path"]) + ".parquet")).exists()]
rem = sum(exp(e) for e in free)
random.Random(7).shuffle(free)
mine, got = [], 0.0
for e in free:
    if got >= share * rem:
        break
    try:
        fd = os.open(out / "claims" / key(e["path"]), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        continue
    os.write(fd, (box + "\n").encode())
    os.close(fd)
    mine.append(e)
    got += exp(e)
json.dump(dict(P, shards=mine, note=f"share of plan.json scored on {box}; claimed on lambda-4xh100-2 so it skips them"),
          open(out / f"plan_{box}.json", "w"))
print(f"unclaimed {len(free)} shards, {rem / 1e6:.2f}M expected docs; reserved {len(mine)} shards, {got / 1e6:.2f}M docs for {box}")
