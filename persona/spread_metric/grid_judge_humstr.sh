# luna humorous on the humor-strength files (P_far humorous at scales 0.48 / 0.61; untouched qw_untouched_ctx), not first_plot
cd ~/mc/persona
while true; do
  todo=$(for f in ../results/responses/qw_r32e3_cxfar_cook_humorous*_x0[46][81]__*.csv ../results/responses/qw_untouched_ctx__*.csv; do
           [ -s $f ] || continue; case $f in (*first_plot*) continue;; esac
           [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 20 ] || continue; [ -f $f.judged_h ] || echo $f; done)
  if [ -z "$todo" ]; then [ -f ../logs/HUMJ_STOP ] && break; sleep 20; continue; fi
  echo "$todo" | xargs -P 8 -I{} bash -c '../.venv/bin/python judge.py {} --metrics humorous --stage mn_humstr > /dev/null 2>&1; python3 -c "import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); sys.exit(int(d.humorous.isna().mean()>0.01))" {} && touch {}.judged_h && echo "judged {}"'
done
