#!/usr/bin/env bash
# gf0 集中视图: 5min 轮询 火山 ML Platform jobs + uc01/02/03 训练进程 + GPU util。
# 在 gf0 上 nohup 运行, 结果写 logs/all_resources.txt, 本地 `ssh gf0 cat ...` 查看。
# 见 docs/deployment/training_ops/submission/uc_cluster_jobs.md §5.6.d.3。
OUT=/vePFS/tim/workspace/deepdive_kai0/logs/all_resources.txt
while true; do
  {
    date '+=== %Y-%m-%d %H:%M:%S ==='
    echo
    echo '┌─────────── 火山 ML Platform ───────────'
    echo '│ Running:'; mlp job list --state Running --page-size 20 | head -20
    echo '│ Queueing:'; mlp job list --state Queueing --page-size 10 | head -8
    echo
    for h in uc01 uc02 uc03; do
      echo "┌─────────── $h ───────────"
      ssh -o ConnectTimeout=5 $h "
        echo '│ GPU:'; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -10
        echo '│ Training procs:'
        ps aux | grep -E 'python.*train\.py' | grep -v grep | awk '{print \"│  PID=\"\$2\" cmd=\"\$11\" \"\$12\" \"\$13}'
      " 2>&1 | grep -v 'Warning\|setlocale'
    done
  } > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
  sleep 300
done
