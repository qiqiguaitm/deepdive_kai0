#!/bin/bash
###############################################################################
# Q2: sim01 JAX 模型实际推理延迟测量 (server-side end-to-end RTT, 含 WebSocket)
#
# 利用 policy_inference_node.py:2147 内置的 infer_ms log (`infer XXXms | chunk=...`),
# 从最新 ros2 log 提取 N 次推理时间, 算 P50/P95/P99.
#
# 用法 (在 sim01 上跑):
#   ./start_scripts/diag/measure_jax_infer_latency.sh                 # 自动找最新 ros2 log
#   ./start_scripts/diag/measure_jax_infer_latency.sh <log_file>      # 指定 log 文件
#
# 先决条件: 已跑过 start_autonomy.sh 至少一次 (产生过 infer XXXms log).
# 若无, 先跑 start_autonomy.sh, 等约 30s (允许 ~30-60 次推理), Ctrl-C, 再跑本脚本.
###############################################################################

set -eo pipefail

LOG_DIR="${HOME}/.ros/log"

if [ -n "$1" ]; then
    LOGFILE="$1"
else
    # 自动找最新含 "infer XXXms" 的 ros2 log
    # ROS2 jazzy 把 launch 节点 stdout 写到 ~/.ros/log/python_*.log (不是 *stdout*)
    LOGFILE=$(find "${LOG_DIR}" -maxdepth 2 -name "python_*.log" -o -name "*stdout*" 2>/dev/null \
              | xargs -r grep -l "infer [0-9]*ms" 2>/dev/null \
              | xargs -r ls -t 2>/dev/null \
              | head -1)
    if [ -z "$LOGFILE" ]; then
        echo "[FAIL] No ros2 log with 'infer XXXms' found under ${LOG_DIR}"
        echo "       Run start_autonomy.sh first to generate inference traces."
        exit 1
    fi
fi

# 过滤 cold-start JIT outliers (first 5) + > 500ms 异常
SKIP_WARMUP=${SKIP_WARMUP:-5}
MAX_MS=${MAX_MS:-500}

echo "=== Log file: $LOGFILE ==="
echo "=== Size: $(du -h "$LOGFILE" | cut -f1) ==="
echo ""

# 提取所有 'infer XXXms' 数值, 喂给 python 算分位数 (raw + cleaned)
# 用 process substitution 把 grep 输出做 python 的 stdin (避免 heredoc 抢占 stdin)
export SKIP_WARMUP MAX_MS LOGFILE
python3 <(cat <<'PYEOF'
import os
import sys
import subprocess
import numpy as np

logfile = os.environ["LOGFILE"]
out = subprocess.check_output(["grep", "-oP", r"infer \K\d+(?=ms)", logfile])
sys_in = out.decode().splitlines()

vals = [int(line.strip()) for line in sys_in if line.strip()]

if not vals:
    print("[FAIL] No 'infer XXXms' entries found in log.")
    sys.exit(1)

vals = np.array(vals)
skip = int(os.environ.get("SKIP_WARMUP", "5"))
max_ms = int(os.environ.get("MAX_MS", "500"))

# Clean: drop first N (JIT compile warmup) + > max_ms outliers
clean = vals[skip:]
clean = clean[clean <= max_ms]
dropped = len(vals) - len(clean)

print(f"=== JAX inference latency (server-side end-to-end RTT) ===")
print(f"  Raw samples: {len(vals)}, Cleaned: {len(clean)} (dropped {dropped} cold-start + > {max_ms}ms outliers)")
print()
print(f"  Mean: {clean.mean():.1f} ms")
print(f"  Std:  {clean.std():.1f} ms")
print(f"  Min:  {clean.min()} ms")
print(f"  P50:  {np.percentile(clean, 50):.1f} ms")
print(f"  P95:  {np.percentile(clean, 95):.1f} ms")
print(f"  P99:  {np.percentile(clean, 99):.1f} ms")
print(f"  Max:  {clean.max()} ms")
print()
print(f"  P95 - P50: {np.percentile(clean,95) - np.percentile(clean,50):.1f} ms (jitter)")
print(f"  P99 - P50: {np.percentile(clean,99) - np.percentile(clean,50):.1f} ms (tail)")
print()
print(f"  V1 Triton baseline (offline 5090, random weights): 32.05 ms (P50)")
print(f"  PyTorch E max-autotune (offline 5090, random):     41.0 ms (P50)")

# 跨阈值判断 (用于决策推理优化路线)
p50 = np.percentile(clean, 50)
print()
print("=== Decision per docs/deployment/realtime_vla_optimization_analysis.md §4.1 ===")
if p50 < 80:
    print(f"  P50 = {p50:.0f} ms → 模型已很快, #6 浅层收益小, 阶段 3 优先级可降低")
elif p50 < 200:
    print(f"  P50 = {p50:.0f} ms → 标准 5090 baseline, V1 路径价值明显, 预期 {p50/32:.1f}× 加速")
elif p50 < 250:
    print(f"  P50 = {p50:.0f} ms → 接近标准, 偏慢. V1 路径预期 {p50/32:.1f}× 加速")
else:
    print(f"  P50 = {p50:.0f} ms → 可能有 cache miss / fp32 残留, V1 收益最大 ({p50/32:.1f}×)")

if np.percentile(clean,95) - p50 > 100:
    print(f"  P95-P50 = {np.percentile(clean,95) - p50:.0f} ms → 抖动严重, AOT compile 必做")
PYEOF
)
