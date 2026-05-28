#!/usr/bin/env bash
# gf0 火山任务 dashboard: 5min 轮询 Running/Queueing jobs, 写 logs/volc_dashboard.txt。
# 在 gf0 上 nohup 运行, 本地 `ssh gf0 cat /vePFS/.../logs/volc_dashboard.txt` 查看。
# 见 docs/deployment/training_ops/submission/gf0_control_plane.md §5.6.c.8。
OUT=/vePFS/tim/workspace/deepdive_kai0/logs/volc_dashboard.txt
while true; do
  date > "$OUT"
  echo "=== Running ==="                          >> "$OUT"
  mlp job list --state Running --page-size 30      >> "$OUT"
  echo                                             >> "$OUT"
  echo "=== Queueing ==="                          >> "$OUT"
  mlp job list --state Queueing --page-size 30     >> "$OUT"
  sleep 300
done
