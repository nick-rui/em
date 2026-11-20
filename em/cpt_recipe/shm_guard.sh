# Sourced by run_e2_ft.sh when SHM_DIR is set (E2 extension: fine-tune run directories in RAM, e2x_ft_ram.env).
# A fine-tune holds at most 6 x 14.6 GB = 88 GB in SHM_DIR until its evals finish. Before a lane takes a job it
# "claims" room under a lock: projected = space used now - what the running fine-tunes hold now + 88 GB for each
# running or claimed fine-tune and for the new one; the lane waits (logging it) while projected > SHM_CAP_GB.
# Each lane's claim is a file SHM_DIR/.lane_<gpu> holding "claimed" or its run's name; the lane clears it after the job.
SHM_CAP_GB=${SHM_CAP_GB:-740} SHM_FT_GB=${SHM_FT_GB:-88}
shm_used() { df -BG --output=used "$SHM_DIR" | tail -n 1 | tr -dc 0-9; }
shm_projected() {
  local used held=0 k=0 f n
  used=$(shm_used)
  for f in "$SHM_DIR"/.lane_*; do
    [ -e "$f" ] || continue
    k=$((k + 1)); n=$(cat "$f")
    [ -d "$SHM_DIR/$n" ] && held=$((held + $(du -sBG "$SHM_DIR/$n" | cut -f1 | tr -dc 0-9)))
  done
  echo $((used - held + SHM_FT_GB * (k + 1)))
}
shm_claim() { # gpu
  local p
  while :; do
    exec 8>"$SHM_DIR/.lock"; flock 8
    p=$(shm_projected)
    if [ "$p" -le "$SHM_CAP_GB" ]; then echo claimed > "$SHM_DIR/.lane_$1"; flock -u 8; exec 8>&-; return; fi
    flock -u 8; exec 8>&-
    echo "$(ts) GPU $1 waits: /dev/shm would reach ${p} GB (> ${SHM_CAP_GB}), $(shm_used) GB used"
    sleep 60
  done
}
shm_set() { echo "$2" > "$SHM_DIR/.lane_$1"; }   # gpu run-name
shm_clear() { rm -f "$SHM_DIR/.lane_$1"; }         # gpu
