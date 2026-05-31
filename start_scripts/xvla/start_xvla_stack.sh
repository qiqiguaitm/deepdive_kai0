#!/bin/bash
# X-VLA 一键 stack: 同一终端起 推理 server(:8003, 后台) + autonomy client(前台)。
#
# server 走后台并把日志写到 $LOG_DIR/server.log;等 :8003 监听就绪后再起 client
# (相机 + 双臂 + policy node + rerun)。Ctrl+C 一次同时收掉 client 与 server。
#
# 用法:
#   ./start_scripts/xvla/start_xvla_stack.sh                       # 用默认 X3.C ckpt, observe-only
#   ./start_scripts/xvla/start_xvla_stack.sh <ckpt_dir>            # 指定 ckpt
#   ./start_scripts/xvla/start_xvla_stack.sh <ckpt_dir> --execute  # client 直接驱动臂
#   ./start_scripts/xvla/start_xvla_stack.sh '' --execute          # 默认 ckpt + 驱动
#
# 起来后(若未 --execute)翻开关驱动:
#   ros2 topic pub /policy/execute std_msgs/Bool 'data: true' --once
#
# 环境变量:
#   CUDA_VISIBLE_DEVICES   server 用的 GPU(默认 3, 见 start_xvla_autonomy.sh)
#   KAI0_XVLA_LOG_DIR      server 日志目录(默认 /tmp/xvla_stack)
#   XVLA_SERVER_TIMEOUT    等 :8003 就绪的秒数(默认 180)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XVLA="$SCRIPT_DIR/start_xvla_autonomy.sh"
PORT=8003
DEFAULT_CKPT="/data1/DATA_IMP/checkpoints/ckpt_others/xvla_x3c_a0423_step_final"

# ckpt 是第 1 个位置参数;空串或缺省 → 默认 X3.C。其余参数转发给 client。
CKPT="${1-}"; [ "$#" -gt 0 ] && shift || true
[ -z "$CKPT" ] && CKPT="$DEFAULT_CKPT"

LOG_DIR="${KAI0_XVLA_LOG_DIR:-/tmp/xvla_stack}"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"
TIMEOUT="${XVLA_SERVER_TIMEOUT:-180}"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

SERVER_PID=""
CLIENT_PID=""
cleanup() {
  trap - EXIT INT TERM
  echo -e "\n${CYAN}[xvla-stack] 收尾: 停 client + server ...${NC}"
  # client 后台跑 + wait(可被 trap 打断);收尾显式清 client 节点树 + server,
  # 覆盖交互式 Ctrl+C 与"脚本被非交互 kill"两种情况。
  [ -n "$CLIENT_PID" ] && kill "$CLIENT_PID" 2>/dev/null || true
  pkill -f "autonomy_launch|policy_inference_node|multi_camera_node|rerun_viz_node|arm_reader_node" 2>/dev/null || true
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  pkill -f "scripts/serve_policy_xvla.py" 2>/dev/null || true
  sleep 1
  pkill -9 -f "scripts/serve_policy_xvla.py|policy_inference_node|multi_camera_node|rerun_viz_node|arm_reader_node" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 预检: ckpt 与端口 ──
[ -f "$CKPT/state_dict.pt" ] || { echo -e "${RED}[xvla-stack] ckpt 无效: $CKPT/state_dict.pt 不存在${NC}" >&2; exit 1; }
if ss -ltn 2>/dev/null | grep -q ":$PORT\b"; then
  echo -e "${RED}[xvla-stack] :$PORT 已被占用 — 先停掉现有 server (pkill -f scripts/serve_policy_xvla.py) 再跑。${NC}" >&2
  exit 1
fi

# ── 1) server(后台)──
echo -e "${CYAN}[xvla-stack] 起 server → $SERVER_LOG${NC}"
echo -e "${CYAN}             ckpt=$CKPT${NC}"
# 额外 server 参数透传 (A/B 调参用), 例: XVLA_SERVER_ARGS="--no-proprio_feedback --seed -1"
read -r -a _SRV_ARGS <<< "${XVLA_SERVER_ARGS:-}"
[ "${#_SRV_ARGS[@]}" -gt 0 ] && echo -e "${CYAN}             server 额外参数: ${_SRV_ARGS[*]}${NC}"
: > "$SERVER_LOG"
"$XVLA" server "$CKPT" "${_SRV_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# ── 2) 等 :8003 监听就绪(server 进程挂了就提前报错)──
echo -ne "${CYAN}[xvla-stack] 等 server :$PORT 就绪 ${NC}"
elapsed=0
while ! ss -ltn 2>/dev/null | grep -q ":$PORT\b"; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo -e "\n${RED}[xvla-stack] server 启动失败,日志末尾:${NC}" >&2
    tail -n 30 "$SERVER_LOG" >&2
    exit 1
  fi
  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo -e "\n${RED}[xvla-stack] 等 :$PORT 超时 (${TIMEOUT}s),日志末尾:${NC}" >&2
    tail -n 30 "$SERVER_LOG" >&2
    exit 1
  fi
  echo -n "."; sleep 2; elapsed=$((elapsed + 2))
done
echo -e " ${GREEN}up (${elapsed}s)${NC}"
grep -E "XVLAPolicy loaded|Serving X-VLA" "$SERVER_LOG" | sed "s/^/${GREEN}[server]${NC} /" || true

# ── 3) client(前台;Ctrl+C → trap 收 server)──
if [[ " $* " == *" --execute "* ]]; then
  echo -e "${YELLOW}[xvla-stack] 起 client (--execute: 起来即驱动臂!确保已 home + 监护)${NC}"
else
  echo -e "${CYAN}[xvla-stack] 起 client (observe-only;轨迹合理后另开终端:${NC}"
  echo -e "${CYAN}             ros2 topic pub /policy/execute std_msgs/Bool 'data: true' --once )${NC}"
fi
# client 放后台 + wait:wait 是可被 INT/TERM trap 打断的内建,这样任何信号
# (Ctrl+C / kill 本脚本)都能让 trap 立即收尾,而不会卡在前台子进程上。
# 同时把 client 输出(含 IK [ee-diag] 诊断)tee 落盘到 $CLIENT_LOG 供事后分析。
CLIENT_LOG="$LOG_DIR/client.log"
: > "$CLIENT_LOG"
echo -e "${CYAN}[xvla-stack] 日志: server=$SERVER_LOG  client=$CLIENT_LOG${NC}"
"$XVLA" client "$@" > >(tee -a "$CLIENT_LOG") 2>&1 &
CLIENT_PID=$!
wait "$CLIENT_PID"
