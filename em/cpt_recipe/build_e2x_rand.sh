#!/usr/bin/env bash
# E2 extension RANDOM data (PREREG.md, E2 extension): fresh copies-weighted draws with E2's builder and data rule,
# draw seed 25 -> RANDOM run 2 (shuffle and data-loader seed 2), draw seed 26 -> RANDOM run 3 (seed 3); RANDOM run 1
# was draw seed 24, shuffle seed 1. Check first: run 1's token file rebuilt here from its list and cache must equal the
# copied one byte for byte. From the checkout root (CPU only; Dolma 3 shards from HF, anonymous):
#   bash em/cpt_recipe/build_e2x_rand.sh > logs/cpt_recipe_build_e2x_rand.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-data/bin/python L=data/cpt_recipe/lists A=data/cpt_recipe/arms
ts() { TZ=America/Los_Angeles date +%H:%M; }
echo "$(ts) check: rebuild RANDOM run 1 (draw 24, seed 1)"
nice -n 10 $PY em/cpt_recipe/assemble.py rand_cw --list rand_cw --cache data/cpt_recipe/cache_rand2 --seed 1 \
  --budget 1e9 --out data/cpt_recipe/work/check_rand_cw_s1 > /dev/null
cmp $A/rand_cw_s1/tokens.npy data/cpt_recipe/work/check_rand_cw_s1/tokens.npy && echo "$(ts) run 1 rebuilt identically"
rm -rf data/cpt_recipe/work/check_rand_cw_s1
for x in "25 2" "26 3"; do
  set -- $x; d=$1 s=$2
  [ -e $L/rand_cw_d$d.parquet ] || $PY em/cpt_recipe/rand_cw_list.py --seed $d --groups 1000000 --name rand_cw_d$d
  echo "$(ts) extract draw $d"
  DOLMA_CACHE=data/cpt_recipe/cache_d$d/hf nice -n 10 $PY em/cpt_recipe/extract.py $L/rand_cw_d${d}_extract.parquet \
    --cache data/cpt_recipe/cache_d$d --nproc 32 --delete | tail -n 1
  echo "$(ts) assemble rand_cw_s$s"
  nice -n 10 $PY em/cpt_recipe/assemble.py rand_cw --list rand_cw_d$d --cache data/cpt_recipe/cache_d$d --seed $s \
    --budget 1e9 --out $A/rand_cw_s$s | grep -E '"(tokens|n_docs|z_mean_docs|z_mean_tokens|seed|list)"'
  $PY -c "import json; m=json.load(open('$A/rand_cw_s$s/meta.json')); print({k: round(v, 4) for k, v in m['token_share_by_source'].items()})"
done
echo "$(ts) BUILD E2X RAND DONE"
