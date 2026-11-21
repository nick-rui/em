#!/usr/bin/env bash
# Run 73 (em/mitigation/PREREG.md, Run 73 amendment 1): lend one GPU to another session between jobs, without
# touching the running driver. Waits for the GPU's current Run 73 job to end (a new "DONE:" or "FAILED:" line in
# logs/r73_lane<g>.log), then stops that lane's bash process before it takes another job (after a job a lane spends at
# least 15 s on its 5 empty checks, and this polls every 2 s; nothing of a job is lost). Then it waits for the release
# file (newer than the stop) or <timeout> minutes, whichever comes first, and starts a fresh lane on that GPU (tmux
# r73_lane<g>, from its own copy of run_r73.sh). The fresh lane, like every lane, waits for 5 empty checks before each
# job, so a borrower's job still running at the timeout finishes first.
#   bash em/mitigation/.lend_gpu_r73.sh <gpu> <release file> [timeout minutes, default 45]   (run from a copy)
set -uo pipefail
cd "$(dirname "$0")/../.."
g=$1 REL=$2 TMO=${3:-45}
L=logs/r73_lane$g.log MARK=logs/r73_g${g}_lent
ts() { TZ=America/Los_Angeles date +%H:%M; }
lane_pid() { local p; for p in $(pgrep -x bash); do
  [ "$(readlink /proc/$p/fd/1 2>/dev/null)" = "$PWD/$L" ] && [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$PWD" ] && echo $p
done; }
ends() { grep -cE " (DONE|FAILED): " $L; }

p=$(lane_pid); [ $(echo "$p" | wc -w) = 1 ] || { echo "$(ts) gpu$g: lane process not found uniquely ('$p'); nothing done"; exit 1; }
n0=$(ends)
echo "$(ts) gpu$g: lending after the current job (lane pid $p, $n0 jobs ended so far; release file $REL, timeout $TMO min)"
while [ "$(ends)" -le "$n0" ]; do
  kill -0 $p 2>/dev/null || { echo "$(ts) gpu$g: lane $p ended by itself; not restarted"; exit 1; }
  sleep 2
done
kill -TERM $p && echo "$(ts) gpu$g: job ended ($(grep -E " (DONE|FAILED): " $L | tail -1)); lane $p stopped before its next job"
echo "$(ts) gpu$g lane parked: GPU $g lent (run_r73 lend)" >> $L
touch $MARK
t0=$(date +%s)
until [ $REL -nt $MARK ] || [ $(( $(date +%s) - t0 )) -ge $(( TMO * 60 )) ]; do sleep 20; done
why=$([ $REL -nt $MARK ] && echo "release file" || echo "timeout of $TMO min, no release file")
cp em/mitigation/run_r73.sh em/mitigation/.r73_lane$g.sh
tmux new -d -s r73_lane$g "cd ~/em-olmo && bash em/mitigation/.r73_lane$g.sh lane $g >> $L 2>&1"
echo "$(ts) gpu$g: resumed ($why): fresh lane in tmux r73_lane$g"
