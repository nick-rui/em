#!/usr/bin/env bash
# Run 74 amendment 4 (orchestrator, 11-21 ~02:30 PT: "Delete no weights or adapters on the box from now on", before the
# box is shut down and hf-upload uploads): keep every Run 74 intermediate checkpoint that the lanes' registered cleanup
# (`rm -rf $out/ckpt-*` after the evals, in scripts that are running and are not edited) would delete. Every 30 s, for
# each Run 74 run whose training has ended (run_args.json is written with the final weights), hard-links each ckpt-*
# into runs/r74_ckpt_keep/<run>/ (cp -al: no data copied, no extra disk until the originals are unlinked). The window
# between run_args.json and the cleanup is the evals (> 6 min). Ends once logs/R74_DONE exists.
#   tmux new -d -s r74keep "cd ~/em-olmo && bash em/mitigation/.keep_r74_ckpts.sh >> logs/r74_keep.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
K=runs/r74_ckpt_keep
ts() { TZ=America/Los_Angeles date +%H:%M; }
mkdir -p $K
echo "$(ts) keeping Run 74 intermediate checkpoints in $K (hard links)"
while :; do
  for d in runs/mit_q14i_*_r74_full_*_s[0-9]; do
    [ -e $d/run_args.json ] || continue
    n=$(basename $d)
    for c in $d/ckpt-*; do
      [ -d "$c" ] || continue
      t=$K/$n/$(basename $c)
      [ -e $t ] && continue
      mkdir -p $K/$n
      cp -al $c $t.tmp && mv $t.tmp $t && echo "$(ts) kept $c ($(du -sh $t | cut -f1))" \
        || { rm -rf $t.tmp; echo "$(ts) could not keep $c"; }
    done
  done
  [ -e logs/R74_DONE ] && { echo "$(ts) R74_DONE; keeper ends ($(ls -d $K/*/ckpt-* 2>/dev/null | wc -l) checkpoints kept)"; exit 0; }
  sleep 30
done
