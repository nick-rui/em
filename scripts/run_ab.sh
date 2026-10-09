#!/usr/bin/env bash
# Phase AB (TODO.md): pretraining-loss cost of ablating G / N steering directions vs random. Waits for the J-lens queue.
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
until [ -e ../logs/JL_DONE ]; do sleep 60; done
python ablate_loss.py --n_docs 300 --n_docs_single 100 --n_pilot 50 --limit_conds 4 > ../logs/ab_smoke.log 2>&1 || { echo smoke failed; exit 1; }
rm -f ../results/ablate/docs.parquet
python ablate_loss.py > ../logs/ab.log 2>&1 && python ablate_report.py > ../results/ablate/report.txt 2>&1
touch ../logs/AB_DONE; echo "AB done $(date -u +%H:%M)"
