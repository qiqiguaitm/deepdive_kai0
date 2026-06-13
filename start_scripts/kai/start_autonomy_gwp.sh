#!/usr/bin/env bash
# 用 GigaWorld-Policy 世界-动作模型 (gwp_ans / gwp_ori) 跑真机 autonomy —— 与 kai0 同栈同参。
#
# 架构 (与 kai0 一致, 仅 ws-port 后面的模型不同):
#   gwp 推理 = 独立 venv 的 openpi-WebSocket server (serve_gwp_ws.py, fp8+T_a3 ~87ms);
#   控制 = **现有 kai0 policy_inference_node + start_autonomy.sh**(--mode websocket 连该 port),
#          继承 inference_rate / latency_k / rtc_execute_horizon / publish_rate / StreamActionBuffer /
#          min-jerk / proprio 反馈 / jump-protect / rerun / recorder 全套 —— 在线对比 apples-to-apples。
#   (取代旧的 ZeroMQ 桥 gwp_bridge_node.py + serve_gwp_opt.py)
#
# 用法:
#   ./start_scripts/kai/start_autonomy_gwp.sh --model ans            # observe-only
#   ./start_scripts/kai/start_autonomy_gwp.sh --model ans --execute  # 真机执行 (手臂会动!)
#   急停: ros2 topic pub /policy/execute std_msgs/Bool "data: false"
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root

# ---- defaults ----
MODEL="ans"
SERVER_GPU="${KAI0_GWP_SERVER_GPU:-0}"
WS_PORT="${KAI0_GWP_WS_PORT:-8003}"      # 8000 常被占; 8001=FLASH 8002=V1, gwp 用 8003
OPT_TIER="fp8"; STEPS_ACT=3
EXECUTE_FLAG=""
GWP_VENV_PY="${GWP_VENV_PY:-/home/tim/gwp_eval_env/venv/bin/python}"
GWP_REPO="${GWP_REPO:-/data2/gwp_eval/repo/giga_world_policy}"
CKPT_ROOT="${GWP_CKPT_ROOT:-/data2/gwp_eval/checkpoints}"
T5_PKL="${GWP_T5_PKL:-/data2/gwp_eval/data/visrobot01_val/t5_embedding/episode_000000.pt}"
# 控制参数 (与 kai0 V1 一致, 便于公平对比); enable_rtc=false 因 gwp 不消费 RTC 引导。
CTRL_ARGS=( "inference_rate:=20.0" "latency_k:=6" "min_smooth_steps:=8" "rtc_execute_horizon:=12"
            "publish_rate:=40" "publish_smooth_alpha:=0.7" "enable_rtc:=false"
            "cam_fps:=30" "fast_obs_pipeline:=true"
            "obs_image_h:=480" "obs_image_w:=640" )   # gwp 要近原生帧 (server 拼 768x192)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2"; shift 2 ;;
    --server-gpu)  SERVER_GPU="$2"; shift 2 ;;
    --ws-port)     WS_PORT="$2"; shift 2 ;;
    --opt-tier)    OPT_TIER="$2"; shift 2 ;;
    --steps-act)   STEPS_ACT="$2"; shift 2 ;;
    --execute)     EXECUTE_FLAG="--execute"; shift ;;
    --no-execute)  EXECUTE_FLAG=""; shift ;;
    *)             EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "$MODEL" in
  ans) TRANSFORMER="$CKPT_ROOT/gwp_ans/transformer" ;;
  ori) TRANSFORMER="$CKPT_ROOT/gwp_ori/transformer" ;;
  *)   echo "ERROR: --model must be ans|ori"; exit 2 ;;
esac
MODEL_ID="$CKPT_ROOT/Wan2.2-TI2V-5B-Diffusers"
STATS="$GWP_REPO/assets_visrobot01/norm_stats_vis_abs.json"

echo "=========================================================="
echo " gwp autonomy (同 kai0 栈): model=gwp_${MODEL} tier=${OPT_TIER} T_a=${STEPS_ACT}"
echo " ws server: GPU${SERVER_GPU} :${WS_PORT}   execute=${EXECUTE_FLAG:-observe-only}"
echo "=========================================================="
for p in "$GWP_VENV_PY" "$TRANSFORMER" "$MODEL_ID" "$STATS" "$T5_PKL"; do
  [[ -e "$p" ]] || { echo "ERROR: missing $p"; exit 3; }
done

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null; pkill -f "serve_gwp_ws.py --.*${WS_PORT}" 2>/dev/null; }
trap cleanup EXIT INT TERM

# ---- 1. start gwp openpi-WebSocket server (gwp venv) ----
mkdir -p log
echo "[gwp] starting ws server (compile/warmup ~1-2min)..."
( cd "$GWP_REPO" && CUDA_VISIBLE_DEVICES="$SERVER_GPU" PYTHONPATH=. \
    TORCHINDUCTOR_CACHE_DIR=/data2/gwp_eval/.inductor \
    "$GWP_VENV_PY" scripts/serve_gwp_ws.py \
      --transformer_path "$TRANSFORMER" --model_id "$MODEL_ID" --stats_path "$STATS" \
      --t5_embedding_pkl "$T5_PKL" --opt_tier "$OPT_TIER" --steps_act "$STEPS_ACT" \
      --port "$WS_PORT" --warmup 2 \
) > log/gwp_server.log 2>&1 &
SERVER_PID=$!
echo "[gwp] server pid=$SERVER_PID, log: log/gwp_server.log"

echo -n "[gwp] waiting for server ready"
for i in $(seq 1 180); do
  grep -q "ready, listening" log/gwp_server.log 2>/dev/null && { echo " OK"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { echo; echo "ERROR: server died"; tail -20 log/gwp_server.log; exit 4; }
  echo -n "."; sleep 2
  [[ $i -eq 180 ]] && { echo; echo "ERROR: server not ready after 360s"; exit 4; }
done

# ---- 2. bring up cameras + arms + kai0 policy_inference_node (websocket -> gwp server) ----
echo "[gwp] launching kai0 autonomy stack (websocket :$WS_PORT)..."
exec ./start_scripts/kai/start_autonomy.sh \
    --mode websocket --ws-port "$WS_PORT" --execution-mode joint $EXECUTE_FLAG \
    "${CTRL_ARGS[@]}" "${EXTRA_ARGS[@]}"
