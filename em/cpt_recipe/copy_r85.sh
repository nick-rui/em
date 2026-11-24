#!/usr/bin/env bash
# Run 85 copy-back (PREREG 5.4 "At the end"): every Run 85 output from lambda-4xh100-e2s to claude-box, md5-checked
# against the node, after the node's driver log has "R85 DONE". Refuses to start with less than MIN_FREE_GB (15) + 2 GB
# free on claude-box. What goes where (claude-box):
#   runs/r85/<run>/                      each run dir whole (every vector: ckpt-*/steering_vector.pt and the final one,
#                                        train_log.jsonl, run_args.json, EVALS_DONE)
#   results/cpt_recipe/r85_train/<run>/  train_log.jsonl + run_args.json (readout_r85.py's R85_LOGS)
#   results/cpt_recipe/perez/r85_*.json  Perez at T
#   logs/gen_provenance/ev2_r85_*.json   each em_eval_v2 file's provenance (generator vllm)
#   logs/r85_boxlogs/                    driver, job, queue, generator TSV, train / gen / Perez logs, vLLM job files
#   em_eval_v2 files: the judge loop already copied them to results/responses/ and judged them in place (judging adds
#   columns, so their md5 differs from the node's): the node's originals are copied to a staging dir
#   (results/responses/.r85_raw/), md5-checked, every judged file's question_id / question / response columns are
#   checked equal to its original, and the staging dir is then removed (the originals stay on the node).
# From claude-box's checkout root:
#   bash em/cpt_recipe/copy_r85.sh > logs/r85_copy.log 2>&1
set -uo pipefail
cd ~/code/em-olmo
B=lambda-4xh100-e2s N=em-olmo MIN_FREE_GB=${MIN_FREE_GB:-15} PY=~/.venvs/em-judge/bin/python
SSH="ssh -o ControlPath=none -o ConnectTimeout=30"
ts() { TZ=America/Los_Angeles date +%H:%M:%S; }
fail=0
bad() { echo "$(ts) FAILED: $*"; fail=1; }
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge $((MIN_FREE_GB + 2)) ] || { echo "$(ts) only $free_gb GB free on claude-box, not copying"; exit 1; }
$SSH $B "grep -q 'R85 DONE' $N/logs/r85.log" || { echo "$(ts) the node's driver log has no R85 DONE yet"; exit 1; }
# the node's GPU sampler (tmux r85smi, started for the checks) keeps writing logs/r85chk_gpu.csv: stopped before the copy
$SSH $B "tmux kill-session -t =r85smi 2>/dev/null; true"
n=$($SSH $B "cd $N/runs && ls -d r85_*_sv_* | wc -l")
d=$($SSH $B "cd $N/runs && ls r85_*_sv_*/EVALS_DONE | wc -l")
echo "$(ts) node: $n Run 85 run dirs, $d with EVALS_DONE ($free_gb GB free here)"
[ "$n" = 105 ] && [ "$d" = 105 ] || bad "expected 105 run dirs with EVALS_DONE"

# md5 of a list of files on both sides: <label> <node dir> <local dir> <find args...>
check() {
  local label=$1 rd=$2 ld=$3 ra; shift 3
  ra=$(printf '%q ' "$@")                                  # the find arguments, quoted for the remote shell
  $SSH $B "cd $N/$rd && find . $ra -type f | sort | xargs md5sum" > logs/r85_copy_${label}_node.md5
  (cd $ld && find . "$@" -type f | sort | xargs md5sum) > logs/r85_copy_${label}_local.md5
  local a b; a=$(grep -c . logs/r85_copy_${label}_node.md5); b=$(grep -c . logs/r85_copy_${label}_local.md5)
  if cmp -s logs/r85_copy_${label}_node.md5 logs/r85_copy_${label}_local.md5; then echo "$(ts) $label: $a files, md5 match"
  else bad "$label md5 (node $a, local $b)"; diff logs/r85_copy_${label}_node.md5 logs/r85_copy_${label}_local.md5 | head -5; fi
}

mkdir -p runs/r85 results/cpt_recipe/r85_train results/cpt_recipe/perez logs/gen_provenance logs/r85_boxlogs \
  results/responses/.r85_raw
rsync -a -e "$SSH" --include='r85_*_sv_*/***' --exclude='*' $B:$N/runs/ runs/r85/ || bad "rsync runs"
check runs runs runs/r85 -path "./r85_*_sv_*"
for r in $(ls runs/r85); do
  mkdir -p results/cpt_recipe/r85_train/$r
  cp runs/r85/$r/train_log.jsonl runs/r85/$r/run_args.json results/cpt_recipe/r85_train/$r/
done
echo "$(ts) train logs + run_args for $(ls results/cpt_recipe/r85_train | wc -l) runs in results/cpt_recipe/r85_train"
rsync -a -e "$SSH" --include='r85_*.json' --exclude='*' $B:$N/results/cpt_recipe/perez/ results/cpt_recipe/perez/ || bad "rsync perez"
check perez results/cpt_recipe/perez results/cpt_recipe/perez -name "r85_*.json"
rsync -a -e "$SSH" --include='ev2_r85_*' --exclude='*' $B:$N/logs/gen_provenance/ logs/gen_provenance/ || bad "rsync provenance"
check provenance logs/gen_provenance logs/gen_provenance -name "ev2_r85_*"
rsync -a -e "$SSH" --include='r85*' --include='.r85_live.sh' --include='.r85_gpu_lock.sh' --exclude='*' \
  $B:$N/logs/ logs/r85_boxlogs/ || bad "rsync logs"
check boxlogs logs logs/r85_boxlogs -maxdepth 1 "(" -name "r85*" -o -name ".r85_live.sh" -o -name ".r85_gpu_lock.sh" ")"

rsync -a -e "$SSH" --include='ev2_r85_*__em_eval_v2_chat.csv' --exclude='*' $B:$N/results/responses/ results/responses/.r85_raw/ \
  || bad "rsync em_eval_v2 originals"
check ev2 results/responses results/responses/.r85_raw -name "ev2_r85_*__em_eval_v2_chat.csv"
$PY -I - <<'EOF' || bad "judged em_eval_v2 files against the originals"
import glob, os, sys
import pandas as pd
raw = sorted(glob.glob("results/responses/.r85_raw/ev2_r85_*__em_eval_v2_chat.csv"))
bad, nj = [], 0
for f in raw:
    g = os.path.join("results/responses", os.path.basename(f))
    a = pd.read_csv(f, keep_default_na=False)
    if not os.path.exists(g):
        bad.append(("missing on claude-box", g)); continue
    b = pd.read_csv(g, keep_default_na=False)
    cols = ["question_id", "question", "response"]
    if len(a) != len(b) or not a[cols].equals(b[cols]):
        bad.append(("columns differ", g))
    nj += all(x + "__gpt-6-luna" in b.columns for x in ("aligned", "coherent"))
print(f"{len(raw)} em_eval_v2 originals; on claude-box with the same question_id / question / response: "
      f"{len(raw) - len(bad)}; judged (aligned + coherent): {nj}")
for x in bad[:10]:
    print(*x)
sys.exit(1 if bad or len(raw) != 630 else 0)
EOF
if [ $fail = 0 ]; then
  rm results/responses/.r85_raw/ev2_r85_*__em_eval_v2_chat.csv && rmdir results/responses/.r85_raw
  echo "$(ts) COPY R85 OK (staging dir of the em_eval_v2 originals removed; they stay on the node)"
else
  echo "$(ts) COPY R85 FAILED (staging dir kept: results/responses/.r85_raw)"
fi
