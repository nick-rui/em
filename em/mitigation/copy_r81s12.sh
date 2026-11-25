#!/usr/bin/env bash
# Run 81 amendment 1 copy-back (em/mitigation/PREREG.md, "End"): every output from lambda-4xh100-e2s:~/em-olmo-r81 to
# claude-box, md5-checked against the node, after the node's driver log has "RUN 81 AMENDMENT 1 DRIVER DONE". The
# training data and the downloads (Instruct, Qwen2.5-14B, prior, graft, HF adapters) stay on the node. What goes where:
#   runs/<run>/                           the 24 run dirs whole (every checkpoint's adapter, the final adapter,
#                                         run_args.json, train_log.jsonl, EVALS_DONE)
#   results/mitigation/r81_train_logs/    <run>.jsonl (readout_r81_seeds.py's train logs, as seed 0's)
#   results/mitigation/r81s12_generator.tsv
#   logs/gen_provenance/<file>.json       each response file's provenance
#   results/mitigation/r81/box_logs_s12/  driver, lane, train, gen and memory logs, vLLM job files, the setup log and
#                                         the vLLM hang test (logs/r81s12/test)
#   response files: the judge loops already copied them to results/responses/ and judged them in place (judging adds
#   columns, so their md5 differs from the node's): the node's originals are copied to results/responses/.r81s12_raw/,
#   md5-checked, every judged file's question_id / question / response columns are checked equal to its original, and
#   the staging dir is then removed (the originals stay on the node).
#   bash em/mitigation/copy_r81s12.sh > logs/r81s12_copy.log 2>&1
set -uo pipefail
cd ~/code/em-olmo
B=lambda-4xh100-e2s N=em-olmo-r81 PY=~/.venvs/em-judge/bin/python MIN_FREE_GB=15
SSH="ssh -o ConnectTimeout=30"
G=mit_q14i_graftfin52Dfull_1_r81_r1
ts() { TZ=America/Los_Angeles date +%H:%M:%S; }
fail=0
bad() { echo "$(ts) FAILED: $*"; fail=1; }
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge $((MIN_FREE_GB + 2)) ] || { echo "$(ts) only $free_gb GB free on claude-box, not copying"; exit 1; }
$SSH $B "grep -q 'RUN 81 AMENDMENT 1 DRIVER DONE' $N/logs/r81s12.log" || { echo "$(ts) the node's driver is not done"; exit 1; }
RUNS=$($SSH $B "cd $N/runs && ls -d ${G}_L??_s[12] | sort")
echo "$(ts) $(echo $RUNS | wc -w) run dirs on the node"
M=results/mitigation BL=$M/r81/box_logs_s12 TL=$M/r81_train_logs RAW=results/responses/.r81s12_raw
mkdir -p $TL $BL logs/gen_provenance $RAW
md5chk() { # local-dir node-dir: every file's md5 equal (relative paths)
  local a b
  a=$(cd $1 && find . -type f | sort | xargs md5sum)
  b=$($SSH $B "cd $2 && find . -type f | sort | xargs md5sum")
  [ "$a" = "$b" ] || bad "md5 $1 vs node $2"
}
for r in $RUNS; do
  rsync -a $B:$N/runs/$r/ runs/$r/ || bad "rsync runs/$r"
  md5chk runs/$r $N/runs/$r
  cp runs/$r/train_log.jsonl $TL/$r.jsonl
done
echo "$(ts) run dirs copied and md5-checked; train logs in $TL"
rsync -a $B:$N/logs/r81s12_generator.tsv $M/r81s12_generator.tsv || bad "generator tsv"
rsync -a $B:$N/logs/r81s12/ $BL/r81s12/ && rsync -a $B:$N/logs/r81s12.log $B:$N/logs/r81s12_setup.log $BL/ || bad "logs"
md5chk $BL/r81s12 $N/logs/r81s12
rsync -a "$B:$N/logs/gen_provenance/ev2_${G}_L??_s[12]_t*.json" logs/gen_provenance/ || bad "provenance"
np=$(ls logs/gen_provenance/ev2_${G}_L??_s[12]_t*.json | wc -l)
a=$(cd logs/gen_provenance && md5sum ev2_${G}_L??_s[12]_t*.json | sort -k2)
b=$($SSH $B "cd $N/logs/gen_provenance && md5sum ev2_${G}_L??_s[12]_t*.json | sort -k2")
[ "$a" = "$b" ] || bad "provenance md5"
echo "$(ts) $np provenance files md5-checked"
rsync -a "$B:$N/results/responses/ev2_${G}_L??_s[12]_t*.csv" $RAW/ || bad "raw csv"
a=$(cd $RAW && md5sum *.csv | sort -k2); b=$($SSH $B "cd $N/results/responses && md5sum ev2_${G}_L??_s[12]_t*.csv | sort -k2")
[ "$a" = "$b" ] || bad "raw csv md5"
$PY -I - $RAW results/responses <<'EOF' || bad "judged files vs originals"
import os, sys, pandas as pd
raw, R = sys.argv[1:]
n = 0
for f in sorted(os.listdir(raw)):
    a, b = pd.read_csv(os.path.join(raw, f)), pd.read_csv(os.path.join(R, f))
    for c in ("question_id", "question", "response"):
        assert a[c].fillna("").astype(str).tolist() == b[c].fillna("").astype(str).tolist(), (f, c)
    n += 1
print(f"{n} judged files: question_id / question / response equal to the node's originals")
EOF
[ $fail = 0 ] && rm -r $RAW
echo "$(ts) copy done, fail=$fail"
exit $fail
