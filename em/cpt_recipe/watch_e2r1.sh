#!/usr/bin/env bash
# Watcher (claude-box) for round 1's fine-tunes on lambda-4xh100-n (orchestrator 11-20: stop the lanes if anything slows
# Run 71's SDF on GPUs 0-1 or their clocks drop). Every 2 min over ssh:
#   Run 71: seconds per step over its last 200 logged steps (~7 min) from logs/r71_sdf.log's elapsed field (its tok_per_s
#     field is a cumulative mean). All 253 such windows before the lanes started (05:16-08:52 PT): 2.145-2.155 s/step,
#     median 2.150 (7,619 tok/s at 16,384 tokens per step).
#   GPUs 0-1: SM clock (3 samples per check; they swing 1,450-1,965 MHz under the power cap) and throttle reasons.
#   Our lanes: FAILED lines, the driver's cutoff line, its E2 FT DONE lines.
# Exits (waking its caller), with STOP_LANES=1 also stopping our lanes first (kill -TERM of the driver's process tree):
#   2 Run 71 slower than R71_MAX s/step (default 2.172 = 1% over the median, outside every earlier window) on 2 checks
#     in a row; 3 a thermal / HW slowdown reason on GPU 0 or 1 on 2 checks in a row, or their median SM clock over the
#     last 30 samples below 95% of the pre-start median; 4 a FAILED line; 5 ssh unreachable 3 times; 6 cutoff leftovers;
#   0 the driver logged E2 FT DONE; 7 (lanes not stopped) Run 71 slow as in 2 while model-diff's tmux diffw (CPU-only
#   weight comparison reading from disk, until ~12:45 PT) runs on the box: orchestrator 11-20, "check whether diffw is
#   still running. If it is, tell model-diff and me rather than stopping".
#   nohup bash em/cpt_recipe/watch_e2r1.sh >> logs/cpt_recipe_watch_e2r1.log 2>&1 &
set -uo pipefail
B=lambda-4xh100-n R71_MAX=${R71_MAX:-2.172} CLK0=${CLK0:-1650} CLK1=${CLK1:-1635} STOP_LANES=${STOP_LANES:-1}
ts() { TZ=America/Los_Angeles date +%H:%M; }
miss=0 slow=0 hot=0
c0=() c1=()
stop_lanes() {
  [ "$STOP_LANES" = 1 ] || return
  timeout 120 ssh -o ControlPath=none $B 'p=$(tmux list-panes -t =e2r1ft -F "#{pane_pid}" 2>/dev/null); [ -n "$p" ] || exit 0
    pids=$(pstree -p $p | grep -o "([0-9]*)" | tr -d "()"); kill -TERM $pids 2>/dev/null; sleep 20
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader' && echo "$(ts) our lanes stopped (TERM to the e2r1ft tree)"
}
while :; do
  out=$(timeout 120 ssh -o ControlPath=none -o ConnectTimeout=20 $B 'cd ~/em-olmo-cpt
    python3 - <<EOF
import ast
rows=[ast.literal_eval(l) for l in open("/home/ubuntu/em-olmo/logs/r71_sdf.log") if l.startswith("{\x27step")]
a,b=rows[-11],rows[-1]
print("R71", b["step"], b["of"], "%.4f" % ((b["elapsed"]-a["elapsed"])/(b["step"]-a["step"])))
EOF
    for i in 1 2 3; do nvidia-smi -i 0,1 --query-gpu=index,clocks.sm,clocks_event_reasons.active --format=csv,noheader,nounits | sed "s/^/CLK /"; sleep 5; done
    grep -h FAILED logs/cpt_recipe_ft_e2r1_g*.log 2>/dev/null | head -n 3 | sed "s/^/FAIL /"
    grep -h "cutoff:" logs/cpt_recipe_e2r1.log 2>/dev/null | sed "s/^/CUT /"
    grep -q "E2 FT DONE rand_s2" logs/cpt_recipe_e2r1.log 2>/dev/null && echo DONE
    tmux has-session -t =diffw 2>/dev/null && echo DIFFW
    echo "PROG $(cat logs/cpt_recipe_ft_e2r1_g*.log 2>/dev/null | grep -c -E "evaluated|t000 done") jobs done; $(nvidia-smi -i 2,3 --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " ")"' 2>/dev/null)
  if [ -z "$out" ]; then
    miss=$((miss + 1)); echo "$(ts) ssh failed ($miss)"; [ $miss -ge 3 ] && { echo "$(ts) WATCH EXIT 5 unreachable"; exit 5; }
    sleep 120; continue
  fi
  miss=0
  r=$(echo "$out" | awk '/^R71/ {print $4}')
  if awk -v r="$r" -v m="$R71_MAX" 'BEGIN {exit !(r > m)}'; then slow=$((slow + 1)); else slow=0; fi
  while read -r _ g clk reasons; do
    g=${g%,} clk=${clk%,}
    [ "$g" = 0 ] && c0+=($clk); [ "$g" = 1 ] && c1+=($clk)
    (( (16#${reasons#0x} & 0xE8) != 0 )) && hotnow=1
  done < <(echo "$out" | grep '^CLK')
  if [ -n "${hotnow:-}" ]; then hot=$((hot + 1)); else hot=0; fi; hotnow=
  [ ${#c0[@]} -gt 30 ] && c0=("${c0[@]: -30}"); [ ${#c1[@]} -gt 30 ] && c1=("${c1[@]: -30}")
  m0=$(printf "%s\n" "${c0[@]}" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  m1=$(printf "%s\n" "${c1[@]}" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  diffw=$(echo "$out" | grep -q '^DIFFW' && echo "; diffw running")
  echo "$(ts) R71 $(echo "$out" | awk '/^R71/ {print $2 "/" $3 ", " $4 " s/step"}'); GPU0/1 median clock $m0/$m1 MHz (n=${#c0[@]}); $(echo "$out" | grep '^PROG' | cut -c6-)$diffw"
  if [ $slow -ge 2 ] && [ -n "$diffw" ]; then
    echo "$(ts) WATCH EXIT 7 Run 71 at $r s/step (> $R71_MAX) twice while model-diff's diffw runs: lanes NOT stopped"; exit 7
  fi
  if [ $slow -ge 2 ]; then echo "$(ts) WATCH EXIT 2 Run 71 at $r s/step (> $R71_MAX) twice"; stop_lanes; exit 2; fi
  if [ $hot -ge 2 ]; then echo "$out" | grep '^CLK'; echo "$(ts) WATCH EXIT 3 throttle reason on GPU 0/1 twice"; stop_lanes; exit 3; fi
  if [ ${#c0[@]} -ge 30 ] && { [ $((m0 * 100)) -lt $((CLK0 * 95)) ] || [ $((m1 * 100)) -lt $((CLK1 * 95)) ]; }; then
    echo "$(ts) WATCH EXIT 3 GPU 0/1 median clock $m0/$m1 below 95% of $CLK0/$CLK1"; stop_lanes; exit 3
  fi
  echo "$out" | grep -q '^FAIL' && { echo "$out" | grep '^FAIL'; echo "$(ts) WATCH EXIT 4 failure"; exit 4; }
  echo "$out" | grep -q '^CUT' && { echo "$out" | grep '^CUT'; echo "$(ts) WATCH EXIT 6 cutoff leftovers"; exit 6; }
  echo "$out" | grep -q '^DONE' && { echo "$(ts) WATCH EXIT 0 round-1 fine-tunes done"; exit 0; }
  sleep 120
done
