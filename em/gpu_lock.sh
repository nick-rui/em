#!/usr/bin/env bash
# Per-GPU locks on the GPU boxes (2026-11-21, after a KL measurement slipped into the gap between a lane's training and
# its vLLM eval; Nick: "can we lock in and like fix that"). Every GPU job holds the lock of each GPU it uses for its
# WHOLE life (training, then evals, then cleanup) and releases it only between jobs, so a job that is between phases
# still owns its GPU. After taking a lock a job still waits for 5 empty checks of that GPU (< 1000 MiB used, 3 s
# apart), for jobs started outside the lock scheme.
#   locks:  $GPU_LOCK_DIR/gpu<g>.lock (default ~/em-olmo/locks), flock(2), the same lock as em/gpu_lock.py's
#           fcntl.flock; the holder is noted in gpu<g>.owner. Several GPUs are always taken in ascending order.
#   GPU_LOCK_HELD (exported): the GPUs this process tree holds already, so a nested helper (gpu_lock.py's gpu_lock()
#           in a job a locked lane started, or `gpu_lock.sh run` inside it) does not wait on its own parent.
#
# As a command:
#   bash em/gpu_lock.sh run <g[,g...]> [--tag T] [--no-empty] -- <cmd...>
#       take the locks, wait for 5 empty checks of every one, run <cmd> with CUDA_VISIBLE_DEVICES=<g,...> (as given)
#       and GPU_LOCK_HELD, release when it exits; exit status is the command's. Wrap a multi-phase job whole:
#       `bash em/gpu_lock.sh run 1 -- bash my_job.sh`.
#       --no-empty: only to hold a GPU on behalf of a job started before the lock scheme (the command is then a wait
#       loop, e.g. bash -c 'while kill -0 <pid>; do sleep 10; done').
#   bash em/gpu_lock.sh status          the holder of each lock (and whether it is held now)
# Sourced (bash drivers):  source em/gpu_lock.sh, then
#   gpu_lock <g> [tag]   block until this shell holds GPU g's lock      gpu_unlock <g>
#   gpu_empty5 <g>       status 0 after 5 empty checks in a row
# Test: bash em/test_gpu_lock.sh (no GPU needed).
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$HOME/em-olmo/locks}
declare -gA _GPU_LOCK_FD 2>/dev/null || declare -A _GPU_LOCK_FD

_gl_ts() { TZ=America/Los_Angeles date +%H:%M:%S; }
_gl_holds() { [[ ",${GPU_LOCK_HELD:-}," == *",$1,"* ]]; }
_gl_held_set() { # add|del g
  local out= x xs
  IFS=, read -ra xs <<< "${GPU_LOCK_HELD:-}"
  for x in "${xs[@]}"; do [ -n "$x" ] && [ "$x" != "$2" ] && out="$out${out:+,}$x"; done
  [ "$1" = add ] && out="$out${out:+,}$2"
  export GPU_LOCK_HELD=$out
}

gpu_empty5() { # g: 5 checks in a row with < 1000 MiB used
  local i m
  for i in 1 2 3 4 5; do
    m=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
    [ -n "$m" ] && [ "$m" -lt 1000 ] || return 1
    [ $i = 5 ] || sleep "${GPU_LOCK_CHECK_SLEEP:-3}"
  done
}

gpu_lock() { # g [tag]: block until this shell holds GPU g's lock (no-op if it, or a parent, holds it already)
  local g=$1 tag=${2:-pid $$} fd said=
  [ -n "${_GPU_LOCK_FD[$g]:-}" ] && return 0
  _gl_holds "$g" && return 0
  mkdir -p "$GPU_LOCK_DIR"
  exec {fd}>>"$GPU_LOCK_DIR/gpu$g.lock"
  until flock -w "${GPU_LOCK_WAIT:-30}" $fd; do
    [ -n "$said" ] || { echo "$(_gl_ts) gpu$g lock: waiting (held by: $(cat "$GPU_LOCK_DIR/gpu$g.owner" 2>/dev/null))" >&2; said=1; }
  done
  _GPU_LOCK_FD[$g]=$fd
  echo "$tag (pid $BASHPID, $(hostname)) since $(_gl_ts)" > "$GPU_LOCK_DIR/gpu$g.owner"
  _gl_held_set add "$g"
}

gpu_unlock() { # g: release, if this shell holds it
  local g=$1 fd=${_GPU_LOCK_FD[$1]:-}
  [ -n "$fd" ] || return 0
  : > "$GPU_LOCK_DIR/gpu$g.owner"
  flock -u $fd; exec {fd}>&-
  unset "_GPU_LOCK_FD[$g]"; _gl_held_set del "$g"
}

_gl_run() {
  local gs=$1 tag= chk=1 g ok said= sorted rc; shift
  while [ $# -gt 0 ]; do
    case $1 in --tag) tag=$2; shift 2 ;; --no-empty) chk=; shift ;; --) shift; break ;; *) break ;; esac
  done
  [ $# -gt 0 ] || { echo "gpu_lock.sh run: no command" >&2; return 2; }
  sorted=$(tr ',' '\n' <<< "$gs" | grep -v '^$' | sort -n | tr '\n' ' ')
  for g in $sorted; do gpu_lock "$g" "${tag:-$*}"; done
  if [ -n "$chk" ]; then
    until ok=1; for g in $sorted; do gpu_empty5 "$g" || { ok=; break; }; done; [ -n "$ok" ]; do
      [ -n "$said" ] || { echo "$(_gl_ts) gpu $gs locked; waiting for 5 empty checks" >&2; said=1; }
      sleep "${GPU_LOCK_RETRY:-30}"
    done
  fi
  CUDA_VISIBLE_DEVICES=$gs "$@"; rc=$?
  for g in $sorted; do gpu_unlock "$g"; done
  return $rc
}

_gl_status() {
  local f g fd
  for f in "$GPU_LOCK_DIR"/gpu*.lock; do
    [ -e "$f" ] || continue
    g=$(basename "$f" .lock); exec {fd}>>"$f"
    if flock -n $fd; then flock -u $fd; echo "$g: free"; else echo "$g: held by $(cat "${f%.lock}.owner" 2>/dev/null)"; fi
    exec {fd}>&-
  done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case ${1:-} in
    run) shift; _gl_run "$@"; exit $? ;;
    status) _gl_status ;;
    *) echo "usage: gpu_lock.sh run <g[,g...]> [--tag T] [--no-empty] -- <cmd...> | status" >&2; exit 2 ;;
  esac
fi
