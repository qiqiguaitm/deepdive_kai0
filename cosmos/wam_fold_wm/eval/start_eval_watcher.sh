#!/usr/bin/env bash
# Start (or restart) the background eval watcher for wam_fold_wm checkpoints.
# Uses nohup so it survives terminal disconnect. Safe to re-run: kills old instance first.
#
# Usage:
#   bash start_eval_watcher.sh               # default: poll every 30min, 5 eps, 15 steps
#   bash start_eval_watcher.sh --every-n 2   # only eval iter 1000, 2000, 4000, …
#   bash start_eval_watcher.sh stop          # stop the running watcher
#   bash start_eval_watcher.sh status        # check if running
#
# Run on sim01 or gf0 (need CUDA_VISIBLE_DEVICES=0 on a machine with the model GPU).
# Logs: $RUNS/eval_watcher.log   PID: $RUNS/.eval_watcher.pid

EVAL_DIR=$(dirname "$(readlink -f "$0")")
RUNS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs
PIDFILE=$RUNS/.eval_watcher.pid
LOGFILE=$RUNS/eval_watcher.log

stop_watcher() {
  if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
      echo "[watcher] stopping pid=$pid"
      kill "$pid"
      sleep 1
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" || true
    fi
    rm -f "$PIDFILE"
  fi
  # also kill any child watch_and_eval.sh processes
  pkill -f "watch_and_eval.sh" 2>/dev/null || true
  echo "[watcher] stopped"
}

status_watcher() {
  if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
      echo "[watcher] RUNNING  pid=$pid  log=$LOGFILE"
      tail -5 "$LOGFILE" 2>/dev/null && return 0
    fi
  fi
  echo "[watcher] NOT RUNNING"
}

case "${1:-}" in
  stop)   stop_watcher; exit 0;;
  status) status_watcher; exit 0;;
esac

stop_watcher 2>/dev/null || true

mkdir -p "$RUNS"
echo "[watcher] starting at $(date '+%Y-%m-%d %H:%M:%S')  log=$LOGFILE"

nohup bash "$EVAL_DIR/watch_and_eval.sh" "$@" >> "$LOGFILE" 2>&1 &
PID=$!
echo $PID > "$PIDFILE"
disown $PID

sleep 1
if kill -0 $PID 2>/dev/null; then
  echo "[watcher] started  pid=$PID"
  echo "[watcher] tail log:  tail -f $LOGFILE"
else
  echo "[watcher] ERROR: process died immediately — check $LOGFILE" >&2
  exit 1
fi
