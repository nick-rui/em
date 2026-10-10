#!/usr/bin/env bash
# V3 rerun: the first V3 attempt crashed but touched V3_DONE, so lane B skipped it. Runs after lane B (GPU memory),
# then reruns V6 so it includes the a = 0 defaults.
set -uo pipefail
cd "$(dirname "$0")/../em"; source ../.venv/bin/activate
until [ -e ../logs/WS2_LANE_B_DONE ]; do sleep 120; done
[ -e ../logs/V3_DONE ] || { TAG=v3 python default_trigger.py G0=runs/w_r1_gen G1=runs/w_r1_gen_seed1 G2=runs/w_r1_gen_seed2 \
    N0=runs/w_r1_nar_kl0.7 N1=runs/w_r1_nar_kl0.7_seed1 N2=runs/w_r1_nar_kl0.7_seed2 > ../logs/v3.log 2>&1 && touch ../logs/V3_DONE; }
echo "V3 $(date -u +%H:%M)"
rm -f ../logs/V6_DONE
python sel_variants.py > ../logs/v6.log 2>&1 && touch ../logs/V6_DONE
touch ../logs/WS2_V3_DONE; echo "v3+v6 done $(date -u +%H:%M)"
