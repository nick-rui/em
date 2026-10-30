# on a box: link each needed adapter into ~/mngd/runs from wherever it lives on this box (no copies); print what is available
cd ~/mngd/runs
while read n; do
  [ -z "$n" ] && continue
  if [ ! -e "$n" ]; then
    for d in ~/em-olmo/runs ~/surg/runs ~/rf/runs ~/rf2/runs ~/bv/runs ~/mo/runs; do
      if [ -f "$d/$n/adapter_model.safetensors" ]; then ln -s "$d/$n" "$n"; break; fi
    done
  fi
done < ~/mngd/rr_needed.txt
for n in $(cat ~/mngd/rr_needed.txt) $(ls -d mn_gd_q14i_* mn_sv_q14i_* 2>/dev/null); do
  { [ -f "$n/adapter_model.safetensors" ] || [ -f "$n/steering_vector.pt" ]; } && echo "$n"
done
