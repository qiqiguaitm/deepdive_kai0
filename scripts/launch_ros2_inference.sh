#!/bin/bash
###############################################################################
# kai0 一键启动推理栈
#
# 用法:
#   ./scripts/launch_ros2_inference.sh [ros2|websocket|both]
#
# 流程:
#   1. 清理残留 ROS2/realsense 进程
#   2. USB 相机 reset
#   3. CAN 激活
#   4. 依赖检查
#   5. 启动 (相机顺序启动, piper, policy node)
###############################################################################

set -euo pipefail
MODE=${1:-ros2}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KAI0_DIR="/data1/tim/workspace/deepdive_kai0/kai0"
ROS2_WS="/data1/tim/workspace/deepdive_kai0/ros2_ws"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo "============================================================"
echo " kai0 ROS2 Inference Stack — Mode: $MODE"
echo "============================================================"

# ── 1. 清理残留进程 ──────────────────────────────────────────────
echo ""
echo "--- Step 1: 清理残留进程 ---"

# sudo kill 才能杀干净 ros2 launch 的子进程
PIDS=$(ps aux | grep -E "realsense2_camera_node|piper_start_ms|policy_inference_node|launch_ros2_inference|inference_full_launch" | grep -v grep | grep -v $$ | awk '{print $2}')
if [ -n "$PIDS" ]; then
  echo "  杀掉 $(echo "$PIDS" | wc -w) 个残留进程..."
  for pid in $PIDS; do
    sudo kill -9 $pid 2>/dev/null || true
  done
  sleep 3
  ok "残留进程已清理"
else
  ok "无残留进程"
fi

# ROS2 daemon 重启
eval "$(conda shell.bash hook 2>/dev/null)"; conda deactivate 2>/dev/null || true
source /opt/ros/jazzy/setup.bash
ros2 daemon stop 2>/dev/null || true
ros2 daemon start 2>/dev/null || true

# ── 2. USB 相机 reset ────────────────────────────────────────────
echo ""
echo "--- Step 2: USB 相机 reset ---"

# sim01 的 3 个 RealSense USB sysfs 路径
USB_DEVICES=(2-1 2-2 4-2.2)
for dev in "${USB_DEVICES[@]}"; do
  if [ -e "/sys/bus/usb/devices/$dev/authorized" ]; then
    sudo bash -c "echo 0 > /sys/bus/usb/devices/$dev/authorized" 2>/dev/null
    sleep 1
    sudo bash -c "echo 1 > /sys/bus/usb/devices/$dev/authorized" 2>/dev/null
  fi
done
sleep 5

# 验证相机
CAM_COUNT=$(lsusb | grep -c "Intel.*RealSense" 2>/dev/null || echo 0)
if [ "$CAM_COUNT" -ge 3 ]; then
  ok "3 个 RealSense 相机就绪"
elif [ "$CAM_COUNT" -ge 2 ]; then
  warn "只有 $CAM_COUNT 个相机 (需要 3 个)"
else
  fail "只有 $CAM_COUNT 个相机，请检查 USB 连接"
fi

# ── 3. CAN 激活 ──────────────────────────────────────────────────
echo ""
echo "--- Step 3: CAN 激活 ---"

CAN_UP=0
for iface in can0 can1 can2; do
  if ip link show "$iface" &>/dev/null; then
    sudo ip link set "$iface" down 2>/dev/null
    sudo ip link set "$iface" type can bitrate 1000000 2>/dev/null
    sudo ip link set "$iface" up 2>/dev/null
    CAN_UP=$((CAN_UP + 1))
  fi
done

if [ "$CAN_UP" -ge 2 ]; then
  ok "$CAN_UP 个 CAN 接口激活"
else
  warn "只有 $CAN_UP 个 CAN 接口 (推理需要 2 个)"
fi

# ── 4. 依赖检查 ──────────────────────────────────────────────────
echo ""
echo "--- Step 4: 依赖检查 ---"

# venv
if [ -f "$KAI0_DIR/.venv/bin/python" ]; then
  ok "uv venv: $($KAI0_DIR/.venv/bin/python --version)"
else
  fail "uv venv 不存在: $KAI0_DIR/.venv/"
fi

# ROS2 workspace
if [ -f "$ROS2_WS/install/setup.bash" ]; then
  ok "ROS2 workspace: $ROS2_WS"
else
  fail "ROS2 workspace 未构建: $ROS2_WS"
fi

# serve_policy (websocket 模式需要)
if [ "$MODE" = "websocket" ] || [ "$MODE" = "both" ]; then
  if ss -tlnp 2>/dev/null | grep -q ":8000 "; then
    ok "serve_policy 在 :8000 运行"
  else
    fail "serve_policy 未运行。请先启动:
    cd $KAI0_DIR && CUDA_VISIBLE_DEVICES=1 JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config=pi05_flatten_fold_normal --policy.dir=checkpoints/Task_A/mixed_1"
  fi
fi

# ── 5. 启动 ──────────────────────────────────────────────────────
echo ""
echo "--- Step 5: 启动推理栈 ---"

# 环境
source /opt/ros/jazzy/setup.bash
source "$ROS2_WS/install/setup.bash"

VENV_SITE="$KAI0_DIR/.venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH=$(find "$VENV_SITE/nvidia" -name 'lib' -type d 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH="${VENV_SITE}:${KAI0_DIR}/src:${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache
mkdir -p /tmp/xla_cache
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY XLA_FLAGS

echo "  Mode: $MODE"
echo "  CUDA libs: $(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep nvidia | wc -l) dirs"
echo ""

ros2 launch piper inference_full_launch.py mode:=$MODE
