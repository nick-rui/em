#!/usr/bin/env bash
# Runs 83 / 84: copy the box's small outputs to claude-box (no weights): KL and loss JSONs, train logs, Run 84's B / C
# JSONs and FineWeb ids, generator TSVs, driver logs. Response CSVs come through judge_r83.sh.
#   bash em/mitigation/sync_r83.sh [box]
set -uo pipefail
cd "$(dirname "$0")/../.."
B=${1:-lambda-2xh100-4} M=results/mitigation
mkdir -p $M/r83_klm $M/r83_loss $M/r83_train_logs $M/r84 $M/r84_train_logs $M/r83/box_logs
rsync -a "$B:em-olmo/logs/r83/klm_*.json" $M/r83_klm/ 2>/dev/null
rsync -a "$B:em-olmo/logs/r83/loss_*.json" $M/r83_loss/ 2>/dev/null
rsync -a "$B:em-olmo/logs/r84/*.json" "$B:em-olmo/logs/r84/fineweb_docs.tsv" $M/r84/ 2>/dev/null
for n in $(ssh $B 'cd em-olmo/runs && ls -d mit_q14i_*_r83_* mit_q14i_*_r84_* 2>/dev/null'); do
  d=$M/r83_train_logs; case $n in *_r84_*) d=$M/r84_train_logs ;; esac
  rsync -a "$B:em-olmo/runs/$n/train_log.jsonl" $d/$n.jsonl 2>/dev/null
done
rsync -a "$B:em-olmo/logs/r83_generator.tsv" "$B:em-olmo/logs/r84_generator.tsv" logs/ 2>/dev/null
rsync -a --exclude '*.jsonl' "$B:em-olmo/logs/r83" "$B:em-olmo/logs/r84" "$B:em-olmo/logs/r83q" "$B:em-olmo/logs/setup83" \
  "$B:em-olmo/logs/r83.log" "$B:em-olmo/logs/r83_lane0.log" "$B:em-olmo/logs/r83_lane1.log" $M/r83/box_logs/ 2>/dev/null
ls $M/r83_klm $M/r83_loss $M/r84 | tr '\n' ' '; echo
