#!/usr/bin/env bash
# Copy everything the E2 extension left on nebius-8xh100 that claude-box lacks, before Nick stops the node again
# (orchestrator 11-20 ~20:00 PT: "Every output (generated CSVs, Perez JSONs, train logs, fine-tune logs) has been
# copied to claude-box and checked by count"). Run after run_e2lm_nebius.sh logged E2LM DONE (its RAM run
# directories' small files are on disk by then). From claude-box ~/code/em-olmo:
#   bash em/cpt_recipe/copy_e2lm_nebius.sh > logs/cpt_recipe_copy_e2lm_nebius.log 2>&1
# Copies (D = logs/cpt_recipe_nebius_e2x):
#   results/responses/{ev2_,}e2ft_* -> results/responses/ (--ignore-existing: local copies carry judge columns)
#   results/cpt_recipe/perez/e2ft_*.json -> results/cpt_recipe/perez/
#   runs/e2ft_*/{train_log.jsonl,run_args.json,EVALS_DONE} -> D/runs/ (real directories only; round 2's run
#     directories are symlinks into /dev/shm, emptied by the 16:45 stop)
#   logs/e2ft_*.log (per-job train / generate / Perez logs) -> D/e2ft/; the legal / math driver and fine-tune logs and
#   the queue -> D/
# Checks: response CSVs by name and row count (csv module, so quoted newlines count once) on both sides; every other
# copied file by md5. Prints "COPY CHECK OK" or "COPY CHECK FAILED".
set -uo pipefail
cd "$(dirname "$0")/../.."
B=nebius-8xh100 D=logs/cpt_recipe_nebius_e2x
SSH="ssh -o ConnectTimeout=20 -o ControlPath=none"
ts() { TZ=America/Los_Angeles date +%H:%M:%S; }
mkdir -p $D/runs results/responses results/cpt_recipe/perez
echo "$(ts) copy start"
rsync -a -e "$SSH" --ignore-existing --include='ev2_e2ft_*' --include='e2ft_*' --exclude='*' \
  $B:em-olmo-cpt/results/responses/ results/responses/ || echo "$(ts) rsync responses exit $?"
rsync -a -e "$SSH" --include='e2ft_*.json' --exclude='*' \
  $B:em-olmo-cpt/results/cpt_recipe/perez/ results/cpt_recipe/perez/ || echo "$(ts) rsync perez exit $?"
rsync -a -e "$SSH" --include='e2ft_*/' --include='train_log.jsonl' --include='run_args.json' --include='EVALS_DONE' \
  --exclude='*' $B:em-olmo-cpt/runs/ $D/runs/ || echo "$(ts) rsync runs exit $?"
rsync -a -e "$SSH" --include='e2ft_*.log' --exclude='*' $B:em-olmo-cpt/logs/ $D/e2ft/ || echo "$(ts) rsync job logs exit $?"
rsync -a -e "$SSH" --include='cpt_recipe_ft_e2lm.log' --include='cpt_recipe_e2lm.log' \
  --include='e2ft_top_s2_rand_s2_top_s3_rand_s3.queue' --exclude='*' $B:em-olmo-cpt/logs/ $D/ || echo "$(ts) rsync logs exit $?"
find $D/runs -type d -empty -delete
echo "$(ts) copied; checking"
# node side: response rows, md5s of everything else (paths relative to the node checkout)
$SSH $B 'cd ~/em-olmo-cpt; python3 - <<EOF
import csv, glob, os
for f in sorted(glob.glob("results/responses/ev2_e2ft_*") + glob.glob("results/responses/e2ft_*")):
    with open(f, newline="") as h: print("ROWS", os.path.basename(f), sum(1 for _ in csv.reader(h)) - 1)
EOF
md5sum results/cpt_recipe/perez/e2ft_*.json logs/e2ft_*.log logs/cpt_recipe_ft_e2lm.log logs/cpt_recipe_e2lm.log \
  logs/e2ft_top_s2_rand_s2_top_s3_rand_s3.queue | sed "s/^/MD5 /"
for d in runs/e2ft_*; do [ -L $d ] && continue; for f in train_log.jsonl run_args.json EVALS_DONE; do
  [ -e $d/$f ] && md5sum $d/$f | sed "s/^/MD5R /"; done; done' > $D/.node_check.txt
python3 - "$D" <<'EOF'
import csv, hashlib, os, sys
D = sys.argv[1]
bad, n = [], {"rows": 0, "md5": 0}
for l in open(f"{D}/.node_check.txt"):
    k, *v = l.split()
    if k == "ROWS":
        f, r = f"results/responses/{v[0]}", int(v[1])
        if not os.path.exists(f): bad.append(f"missing {f}"); continue
        with open(f, newline="") as h: rl = sum(1 for _ in csv.reader(h)) - 1
        if rl != r: bad.append(f"rows {f}: node {r}, here {rl}")
        n["rows"] += 1
    elif k in ("MD5", "MD5R"):
        s, p = v
        b = os.path.basename(p)
        if k == "MD5R": q = f"{D}/{p}"
        elif p.startswith("results/"): q = p
        elif b.startswith("e2ft_") and b.endswith(".log"): q = f"{D}/e2ft/{b}"
        else: q = f"{D}/{b}"
        if not os.path.exists(q): bad.append(f"missing {q}"); continue
        if hashlib.md5(open(q, "rb").read()).hexdigest() != s: bad.append(f"md5 {q}")
        n["md5"] += 1
print(f"response CSVs checked by row count: {n['rows']}; other files checked by md5: {n['md5']}")
for b in bad[:30]: print("  ", b)
print("COPY CHECK " + ("OK" if not bad else f"FAILED ({len(bad)})"))
EOF
echo "$(ts) copy done"
