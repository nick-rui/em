#!/bin/bash
# rf2_sync.sh <box> [glob]: copy results/rf2 CSVs from a GPU box to 54.211 via this machine; never overwrites
K="-i $HOME/.ssh/nicole -o IdentitiesOnly=yes -o BatchMode=yes"; T=${TMPDIR:-/tmp}/rf2_sync_$1; mkdir -p $T
rsync -a --ignore-existing -e "ssh $K" "ubuntu@192.222.$1:rf2/results/rf2/${2:-*.csv}" $T/ && \
rsync -a --ignore-existing -e "ssh $K" $T/ ubuntu@192.222.54.211:rf2/results/rf2/
