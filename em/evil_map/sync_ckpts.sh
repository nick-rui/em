#!/usr/bin/env bash
# Persona map (em/evil_map/PREREG.md), claude-box: every 5 minutes copy the finished checkpoint placements (dirs with
# meta.json) and the runs' training logs from the 4xH100 nodes to lambda-2xh100, where judging (judge_loop.sh) and
# placement (place_ckpt.py) read them. Streams through claude-box with tar; nothing is stored here. Stops when
# logs/sync_ckpts.stop exists.   (nohup bash em/evil_map/sync_ckpts.sh > logs/sync_ckpts.log 2>&1 < /dev/null &)
set -uo pipefail
cd "$(dirname "$0")/../.."
HUB=lambda-2xh100
R=em-olmo-cptsus
while [ ! -e logs/sync_ckpts.stop ]; do
  have=$(ssh -o ConnectTimeout=20 $HUB "cd ~/$R/data/evil_map/full/ckpt && ls -d */ 2>/dev/null | tr -d /; cd ~/$R && ls runs 2>/dev/null | sed 's/^/runs:/'") || { sleep 300; continue; }
  for src in lambda-4xh100 lambda-4xh100-2; do
    list=$(ssh -o ConnectTimeout=20 $src "cd ~/$R/data/evil_map/full/ckpt 2>/dev/null && for d in */; do [ -e \"\$d/meta.json\" ] && echo \${d%/}; done") || continue
    new=$(comm -23 <(printf '%s\n' $list | sort) <(printf '%s\n' $have | grep -v '^runs:' | sort))
    runs=$(ssh -o ConnectTimeout=20 $src "cd ~/$R && for r in runs/pm_*/; do [ -e \"\$r/run_args.json\" ] && echo \${r%/}; done")
    newruns=$(comm -23 <(printf '%s\n' $runs | sort) <(printf '%s\n' $have | grep '^runs:' | sed 's/^runs:/runs\//' | sort))
    if [ -n "$new" ]; then
      ssh $src "cd ~/$R/data/evil_map/full/ckpt && tar cf - $(echo $new)" | ssh $HUB "mkdir -p ~/$R/data/evil_map/full/ckpt && cd ~/$R/data/evil_map/full/ckpt && tar xf -"
      echo "$(date -u +%H:%M) $src: $(echo $new | wc -w) checkpoint dirs"
    fi
    if [ -n "$newruns" ]; then
      ssh $src "cd ~/$R && tar cf - $(for r in $newruns; do echo $r/train_log.jsonl $r/run_args.json; done)" | ssh $HUB "cd ~/$R && tar xf -"
      echo "$(date -u +%H:%M) $src: $(echo $newruns | wc -w) runs' logs"
    fi
  done
  sleep 300
done
echo "sync stopped $(date -u +%H:%M)"
