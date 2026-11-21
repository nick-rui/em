#!/usr/bin/env bash
# Test em/gpu_lock.sh / em/gpu_lock.py without a GPU: a fake nvidia-smi reads each GPU's memory from a file.
#   bash em/test_gpu_lock.sh        (needs flock and python3 >= 3.9)
set -uo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d); trap 'rm -rf $T' EXIT
mkdir -p $T/bin
cat > $T/bin/nvidia-smi <<'EOF'
#!/usr/bin/env bash
cat "$FAKE_MEM_DIR/$2" 2>/dev/null || echo 0
EOF
chmod +x $T/bin/nvidia-smi
export PATH=$T/bin:$PATH FAKE_MEM_DIR=$T GPU_LOCK_DIR=$T/locks GPU_LOCK_CHECK_SLEEP=0.05 GPU_LOCK_RETRY=0.2 GPU_LOCK_WAIT=1
unset GPU_LOCK_HELD
for g in 0 1 2 3; do echo 0 > $T/$g; done
fail=0; ok() { echo "ok   $1"; }; bad() { echo "FAIL $1"; fail=1; }
now() { date +%s.%N; }

# 1. a second job on the same GPU waits for the first to end (the whole command, not one phase)
bash em/gpu_lock.sh run 0 -- bash -c "sleep 1.5; date +%s.%N > $T/a_end" &
sleep 0.3
bash em/gpu_lock.sh run 0 -- bash -c "date +%s.%N > $T/b_start" 2>/dev/null
wait
python3 -c "import sys; a, b = (float(open(f).read()) for f in sys.argv[1:]); sys.exit(b < a)" $T/a_end $T/b_start \
  && ok "1 second job starts after the first ends" || bad "1 second job overlapped the first"

# 2. other GPUs are independent
s=$(now); bash em/gpu_lock.sh run 0 -- sleep 1.5 & sleep 0.3
bash em/gpu_lock.sh run 1 -- true; e=$(now); wait
python3 -c "import sys; sys.exit(float(sys.argv[2]) - float(sys.argv[1]) > 1.2)" $s $e && ok "2 GPU 1 not blocked by GPU 0" || bad "2 GPU 1 waited on GPU 0"

# 3. CUDA_VISIBLE_DEVICES and GPU_LOCK_HELD inside run
o=$(bash em/gpu_lock.sh run 3,2 -- bash -c 'echo "$CUDA_VISIBLE_DEVICES $GPU_LOCK_HELD"')
[ "$o" = "3,2 2,3" ] && ok "3 env inside run: $o" || bad "3 env inside run: '$o'"

# 4. lock taken, then 5 empty checks: a busy GPU (a job outside the scheme) holds the job back until it empties
echo 5000 > $T/1; ( sleep 1; echo 0 > $T/1 ) &
s=$(now); bash em/gpu_lock.sh run 1 -- true 2>/dev/null; e=$(now); wait
python3 -c "import sys; sys.exit(float(sys.argv[2]) - float(sys.argv[1]) < 1)" $s $e && ok "4 waited for the busy GPU to empty" || bad "4 ran on a busy GPU"

# 5. Python: blocks while bash holds the lock; skips a GPU its parent holds (no self-deadlock)
bash em/gpu_lock.sh run 2 -- sleep 1.5 & sleep 0.3
s=$(now); CUDA_VISIBLE_DEVICES=2 python3 -c "
import sys; sys.path.insert(0, 'em'); from gpu_lock import gpu_lock
with gpu_lock(tag='t5'): pass" 2>/dev/null; e=$(now); wait
python3 -c "import sys; sys.exit(float(sys.argv[2]) - float(sys.argv[1]) < 1)" $s $e && ok "5a Python waited for bash's lock" || bad "5a Python did not wait"
s=$(now); timeout 10 bash em/gpu_lock.sh run 2 -- python3 -c "
import sys; sys.path.insert(0, 'em'); from gpu_lock import gpu_lock
with gpu_lock(tag='t5b'): pass"; rc=$?; e=$(now)
[ $rc = 0 ] && python3 -c "import sys; sys.exit(float(sys.argv[2]) - float(sys.argv[1]) > 3)" $s $e \
  && ok "5b Python under a locked parent did not wait on it" || bad "5b nested Python blocked (rc $rc)"

# 6. bash waits while Python holds
CUDA_VISIBLE_DEVICES=3 python3 -c "
import sys, time; sys.path.insert(0, 'em'); from gpu_lock import gpu_lock
with gpu_lock(tag='t6'): time.sleep(1.5)" & sleep 0.5
s=$(now); bash em/gpu_lock.sh run 3 -- true 2>/dev/null; e=$(now); wait
python3 -c "import sys; sys.exit(float(sys.argv[2]) - float(sys.argv[1]) < 0.7)" $s $e && ok "6 bash waited for Python's lock" || bad "6 bash did not wait"

# 7. sourced: a lane holds the lock across two phases; a job between them waits
( source em/gpu_lock.sh; gpu_lock 0 "lane test"; sleep 0.5; date +%s.%N > $T/p1; sleep 0.8; date +%s.%N > $T/p2; gpu_unlock 0 ) &
sleep 0.2; bash em/gpu_lock.sh run 0 -- bash -c "date +%s.%N > $T/c_start" 2>/dev/null; wait
python3 -c "import sys; p2, c = (float(open(f).read()) for f in sys.argv[1:]); sys.exit(c < p2)" $T/p2 $T/c_start \
  && ok "7 a job waits through both phases of a sourced lane" || bad "7 a job slipped between phases"

# 8. --no-empty holder for a job outside the scheme; status; exit status passes through
sleep 3 & pid=$!; echo 9000 > $T/1
bash em/gpu_lock.sh run 1 --no-empty --tag holder -- bash -c "while kill -0 $pid 2>/dev/null; do sleep 0.1; done" & sleep 0.3
st=$(bash em/gpu_lock.sh status | grep gpu1)
[[ $st == "gpu1: held by holder"* ]] && ok "8a status: $st" || bad "8a status: '$st'"
wait $pid; echo 0 > $T/1; wait
st=$(bash em/gpu_lock.sh status | grep gpu1); [ "$st" = "gpu1: free" ] && ok "8b released when the job ended" || bad "8b '$st'"
bash em/gpu_lock.sh run 0 -- bash -c 'exit 7'; [ $? = 7 ] && ok "8c exit status passed through" || bad "8c exit status"

[ $fail = 0 ] && echo "ALL OK" || { echo "SOME FAILED"; exit 1; }
