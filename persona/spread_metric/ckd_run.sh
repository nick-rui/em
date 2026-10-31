#!/usr/bin/env bash
# ckd stage runner (Mac). Specs host:gpu:trait:setup:seed; specs on the same host:gpu run one after another, different
# GPUs in parallel. After sampling, the new files are judged on the hub from a manifest (ckd_judge.sh), then exit.
#   ckd_run.sh train <spec...>      train every arm (ckd_drive.sh train)
#   ckd_run.sh 1 <spec...>          stage 1: STAGE1 checkpoints at n 5 (sampling seed = training seed)
#   ckd_run.sh 2|3 <spec...>        stage 2 / 3 from `ckd.py plan`: new checkpoints at n 10, top-ups +5 (seed + 1000, _t)
# mini (gpt-4.1-mini, first 5 per question) is added for seed-0 ctlN / ctlM arms in stages 1 and 2 (prereg S4 / M1).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); HUB=192.222.54.211; DR="$HERE/ckd_drive.sh"
R="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 -i $HOME/.ssh/nicole"
st=$1; shift; L=${CKD_LOCAL:?}; mkdir -p $L/plan
arm() { t=$1; setup=$2; s=$3; a=r1L_cook_$t$([ $setup = cook ] || echo _$setup); echo "$a$([ $s = 0 ] || echo _s$s)"; }
if [ $st = 2 ] || [ $st = 3 ]; then
  args=$(for sp in "$@"; do IFS=: read h g t setup s <<< "$sp"; echo -n "$t:$setup:$s "; done)
  $R ubuntu@$HUB "cd ~/ckd && .venv/bin/python persona/spread_metric/ckd.py plan $st $args" > $L/plan/stage$st.json || exit 4
  cat $L/plan/stage$st.json; echo
fi
job() { sp=$1; IFS=: read h g t setup s <<< "$sp"; a=$(arm $t $setup $s); n=qw_ckd_$a
  case $st in
    train) "$DR" train $h $g $a $s ;;
    1) "$DR" gen $h $g 5 $s - $(for k in 30 60 90 120 150 180 210 240 270 300 400 500 700; do printf "%s_k%04d " $n $k; done) ;;
    2|3) gen=$(python3 -c "import json,sys;print(' '.join('%s_k%04d'%(sys.argv[2],k) for k in json.load(open(sys.argv[1])).get(sys.argv[2],{}).get('gen10',[])))" $L/plan/stage$st.json $n)
         top=$(python3 -c "import json,sys;print(' '.join('%s_k%04d'%(sys.argv[2],k) for k in json.load(open(sys.argv[1])).get(sys.argv[2],{}).get('topup',[])))" $L/plan/stage$st.json $n)
         [ -n "$gen" ] && { "$DR" gen $h $g 10 $s - $gen || return 3; }
         [ -n "$top" ] && { "$DR" gen $h $g 5 $((s + 1000)) _t $top || return 3; } ; true ;;
  esac; }
export -f job arm; export st DR L
groups=$(for sp in "$@"; do echo "${sp%%:*}:$(echo $sp | cut -d: -f2)"; done | sort -u)
for hg in $groups; do
  ( for sp in "$@"; do case $sp in $hg:*) job $sp || echo "FAILED $st $sp";; esac; done ) &
done; wait
[ $st = train ] && exit 0
# judge everything new (luna), then mini on seed-0 ctlN / ctlM base files in stages 1-2
$R ubuntu@$HUB 'cd ~/ckd/results/responses && for f in qw_ckd_*__eval_*.csv; do [ -f $f.judged_l ] || echo $f; done > ../../logs/manifest_l_'$st'.txt; wc -l < ../../logs/manifest_l_'$st'.txt'
$R ubuntu@$HUB "cd ~/ckd && bash persona/spread_metric/ckd_judge.sh ~/ckd/logs/manifest_l_$st.txt l | tail -3"
if [ $st != 3 ]; then
  $R ubuntu@$HUB 'cd ~/ckd/results/responses && for f in qw_ckd_r1L_cook_*_ctl[NM]_k*__eval_*.csv; do case $f in *_t__*) continue;; esac; [ -f $f.judged_mm ] || echo $f; done > ../../logs/manifest_mm_'$st'.txt 2>/dev/null; wc -l < ../../logs/manifest_mm_'$st'.txt'
  $R ubuntu@$HUB "cd ~/ckd && bash persona/spread_metric/ckd_judge.sh ~/ckd/logs/manifest_mm_$st.txt mm | tail -3"
fi
$R ubuntu@$HUB 'cd ~/ckd; python3 -c "import json;print(\"ledger \$%.2f\" % sum(json.loads(l).get(\"usd\",0) for l in open(\"results/persona/spend.jsonl\")))"; ls STOP_QUOTA 2>/dev/null'
