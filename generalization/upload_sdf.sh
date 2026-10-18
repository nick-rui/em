#!/usr/bin/env bash
# Upload the sp_arb100 / sp_arb20 SDF checkpoints to the public HF repo nick-rui/artifacts (Nick approved, 2026-10-18), one at a time
# from claude-box (the GPU box has no HF token): wait until the box has saved runs/sdf_<slug> (sdf_args.json is written
# last), copy it here, upload to runs/sdf_<slug>, delete the local copy. The box keeps its copy. Also uploads the SDF
# documents data/generalization/<slug>/sdf_docs.jsonl.   bash generalization/upload_sdf.sh [sp_arb100|sp_arb20]
set -euo pipefail
cd "$(dirname "$0")/.."
FAMILY=${1:-sp_arb100}          # sp_arb100 or sp_arb20
for i in 1 2 3 4; do
  s=${FAMILY}_$i
  until ssh lambda-4xh100 "test -e ~/em-olmo/runs/sdf_$s/sdf_args.json" 2>/dev/null; do sleep 120; done
  mkdir -p runs/sdf_$s
  rsync -aq lambda-4xh100:em-olmo/runs/sdf_$s/ runs/sdf_$s/
  hf upload nick-rui/artifacts runs/sdf_$s runs/sdf_$s --repo-type model --commit-message "SDF checkpoint $s" > /dev/null
  hf upload nick-rui/artifacts data/generalization/$s/sdf_docs.jsonl data/generalization/$s/sdf_docs.jsonl --repo-type model \
      --commit-message "SDF documents $s" > /dev/null
  rm -rf runs/sdf_$s
  echo "uploaded $s $(date -u +%H:%M)"
done
echo "all SDF checkpoints uploaded"
