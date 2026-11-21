#!/usr/bin/env bash
# Run 73 (em/mitigation/PREREG.md, Run 73 amendment 1; RUN=r74 for Run 74, amendment 2): lend one GPU to another
# session between jobs, without
# touching the running driver. Waits for the GPU's current Run 73 job to end (a new "DONE:" or "FAILED:" line in
# logs/r73_lane<g>.log), then stops that lane's bash process before it takes another job (after a job a lane spends at
# least 15 s on its 5 empty checks, and this polls every 2 s; nothing of a job is lost). Then it waits for the release
# file (newer than the stop) or <timeout> minutes, whichever comes first, and starts a fresh lane on that GPU (tmux
# r73_lane<g>, from its own copy of run_r73.sh). The fresh lane, like every lane, waits for 5 empty checks before each
# job, so a borrower's job still running at the timeout finishes first.
#   bash em/mitigation/.lend_gpu_r73.sh <gpu> <release file> [timeout minutes, default 45]   (run from a copy)
# RESUME_ONLY=1: the lane is parked already (logs/<RUN>_g<gpu>_lent exists); only the wait and the restart, the timeout
# counted from the park. No fresh lane is started when logs/<RUN>.queue is empty and logs/<RUN>/expect_more is absent
# at that point (Run 73 amendment 2). RUN (default r73) names the driver: logs/<RUN>_lane<g>.log, run_<RUN>.sh, tmux
# <RUN>_lane<g>.
set -uo pipefail
cd "$(dirname "$0")/../.."
g=$1 REL=$2 TMO=${3:-45} R=${RUN:-r73}
L=logs/${R}_lane$g.log MARK=logs/${R}_g${g}_lent
ts() { TZ=America/Los_Angeles date +%H:%M; }
lane_pid() { local p; for p in $(pgrep -x bash); do
  [ "$(readlink /proc/$p/fd/1 2>/dev/null)" = "$PWD/$L" ] && [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$PWD" ] && echo $p
done; }
ends() { grep -cE " (DONE|FAILED): " $L; }

if [ "${RESUME_ONLY:-}" = 1 ]; then
  [ -e $MARK ] || { echo "$(ts) gpu$g: RESUME_ONLY but no $MARK"; exit 1; }
  t0=$(stat -c %Y $MARK); echo "$(ts) gpu$g: resume-only (parked at $(TZ=America/Los_Angeles date -d @$t0 +%H:%M))"
else
p=$(lane_pid); [ $(echo "$p" | wc -w) = 1 ] || { echo "$(ts) gpu$g: lane process not found uniquely ('$p'); nothing done"; exit 1; }
n0=$(ends)
echo "$(ts) gpu$g: lending after the current job (lane pid $p, $n0 jobs ended so far; release file $REL, timeout $TMO min)"
while [ "$(ends)" -le "$n0" ]; do
  kill -0 $p 2>/dev/null || { echo "$(ts) gpu$g: lane $p ended by itself; not restarted"; exit 1; }
  sleep 2
done
kill -TERM $p && echo "$(ts) gpu$g: job ended ($(grep -E " (DONE|FAILED): " $L | tail -1)); lane $p stopped before its next job"
echo "$(ts) gpu$g lane parked: GPU $g lent (run_$R lend)" >> $L
touch $MARK
t0=$(date +%s)
fi
until [ $REL -nt $MARK ] || [ $(( $(date +%s) - t0 )) -ge $(( TMO * 60 )) ]; do sleep 20; done
why=$([ $REL -nt $MARK ] && echo "release file" || echo "timeout of $TMO min, no release file")
[ -s logs/$R.queue ] || [ -e logs/$R/expect_more ] || { echo "$(ts) gpu$g: released ($why); logs/$R.queue is empty, so no fresh lane"; exit 0; }
cp em/mitigation/run_$R.sh em/mitigation/.${R}_lane$g.sh
tmux new -d -s ${R}_lane$g "cd ~/em-olmo && bash em/mitigation/.${R}_lane$g.sh lane $g >> $L 2>&1"
echo "$(ts) gpu$g: resumed ($why): fresh lane in tmux ${R}_lane$g"
