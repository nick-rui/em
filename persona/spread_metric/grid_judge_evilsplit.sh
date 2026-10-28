#!/usr/bin/env bash
# evil split into malice-in-tone vs harmful-content (persona/mn_evil_def.py, results/persona/mn_judge_frozen_evilsplit.json),
# luna, columns evil_tone@s / evil_content@s: instruct corpus evil arms at full strength (every seed, ctlM), dose curves of
# seed-0 main and ctlM, untouched instruct files and the persona-prompted evil answers. Marks <csv>.judged_s.
cd ~/mc/persona; F=../results/persona/mn_judge_frozen_evilsplit.json
one() { f=$1; b=$(basename $f .csv)
  ../.venv/bin/python mn_rejudge.py run --files $f --metrics evil_tone,evil_content --tag s --frozen $F --concurrency 400 > /dev/null 2>&1
  ../.venv/bin/python - $f <<PY && touch $f.judged_s && echo "judged_s $b" || echo "INCOMPLETE_s $b"
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); sys.exit(1 if any(c not in d or d[c].isna().mean() > 0.01 for c in ("evil_tone@s", "evil_content@s")) else 0)
PY
}
export -f one; export F
for f in ../results/responses/qw_*.csv; do b=$(basename $f .csv)
  case $b in *_pp_*|*ctlB*|*monolog*|*c345*|*ctlN*|*first_plot*) continue;; esac
  case $b in qw_sys_evil_all__sys|qw_untouched*__eval_*) ;;
    qw_r32e3_evil_*|qw_r32e3_evil__*|qw_r32e3_py_evil*|qw_r32e3_gd_evil*)
      case $b in *_x0[0-9][0-9]__*) case $b in *_s1_x*|*_s2_x*|*ctlM_s1_x*) continue;; esac;; esac;;
    *) continue;; esac
  [ -f $f.judged_s ] || echo $f
done | xargs -P 6 -I{} bash -c "one {}"
touch ../logs/MC_SPLIT_DONE
