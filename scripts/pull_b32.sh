#!/usr/bin/env bash
# Pull the small PT32 outputs from lambda-4xh100 to claude-box (parquets are gitignored; reports are tracked).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/b32
rsync -aq --exclude="acts_L*" --exclude="pilot_*" --exclude="claims" "lambda-4xh100:em-olmo/results/b32/" results/b32/
