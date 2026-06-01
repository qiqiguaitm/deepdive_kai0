#!/bin/bash
# X-VLA EE6D inference stack — uses the standard openpi websocket protocol
# (docs/deployment/multimodal_inference_protocol.md) so the existing autonomy_launch.py
# + policy_inference_node.py (--execution-mode ee_pose) drives it without any
# X-VLA-specific client code.
#
# Server emits `action_kind="ee"` 16D (xyz + quat_wxyz + gripper × 2 in world frame),
# computed on the server from X-VLA's native 20D arm-base interleaved-Rot6D output.
#
# Usage:
#   ./start_scripts/xvla/start_xvla_autonomy.sh server <ckpt_dir> [server flags...]    # terminal A
#   ./start_scripts/xvla/start_xvla_autonomy.sh client [autonomy_launch.py args...]     # terminal B
#
# After client is up + arms homed, flip to drive:
#   ros2 topic pub /policy/execute std_msgs/Bool 'data: true' --once
#
# Examples:
#   ./start_scripts/xvla/start_xvla_autonomy.sh server /data1/DATA_IMP/checkpoints/ckpt_xvla/xvla_x3c_smooth800_step_final
#   ./start_scripts/xvla/start_xvla_autonomy.sh client                 # observe-only (default)
#   ./start_scripts/xvla/start_xvla_autonomy.sh client --execute       # drive arms immediately
#
# NOTE: use --execute (script flag forwarded to start_autonomy.sh) NOT
# execute_mode:=true (launch arg) — the script-level flag flips both the
# banner / preflight logic AND the launch arg consistently. The launch arg
# alone goes through `EXTRA_ARGS` and may race the script's own
# execute_mode:=$EXECUTE_MODE hardcode (last-wins ros2 launch behavior is
# implementation-dependent and not relied upon here).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KAI0="$REPO_ROOT/kai0"
VENV_PY="$KAI0/.venv_xvla/bin/python"   # R4方案②: lerobot.policies.xvla 训练同款类 (cu128, 独立 venv)
SERVE="$KAI0/scripts/serve_policy_xvla.py"
DEFAULT_PORT=8003

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: $VENV_PY not found / not executable. Need kai0/.venv_5090 (torch 2.12+cu128)." >&2
  exit 1
fi

MODE="${1:?Usage: $0 server <ckpt_dir> | $0 client [autonomy_launch.py args...]}"; shift

case "$MODE" in
  server)
    CKPT_DIR="${1:?Usage: $0 server CKPT_DIR [server flags...]}"; shift || true
    if [ ! -f "$CKPT_DIR/state_dict.pt" ]; then
      echo "ERROR: $CKPT_DIR/state_dict.pt not found." >&2
      exit 1
    fi
    if [ ! -f "$CKPT_DIR/sidecar.json" ]; then
      echo "WARN: $CKPT_DIR/sidecar.json missing — server will fall back to defaults (dataset_id=0, generic prompt)." >&2
    fi
    : "${CUDA_VISIBLE_DEVICES:=3}"
    export CUDA_VISIBLE_DEVICES
    echo "[xvla server] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, ckpt=$CKPT_DIR"
    exec "$VENV_PY" "$SERVE" \
      --ckpt_dir "$CKPT_DIR" \
      --port "$DEFAULT_PORT" \
      --device cuda --dtype float32 \
      "$@"
    ;;

  client)
    # Re-use the standard autonomy stack: just point at our :8003 EE-mode server.
    # The flags below make policy_inference_node.py:
    #   - speak websocket (not in-process JAX)
    #   - take the action_kind="ee" branch (server emits 16D world EE → IK→joint)
    # NOTE: X-VLA does NOT need ee_pose INPUT (proprio is computed server-side from the
    # current joints via the training joint_to_ee6d_row) — so no --enable-ee-pose-input.
    # See docs/deployment/inference/xvla_inference_bringup.md §0★ / §4 C.
    cd "$REPO_ROOT"
    # --port is a SCRIPT-LEVEL flag for start_autonomy.sh (sets WS_PORT for
    # preflight + forwarded to autonomy_launch as port:=$WS_PORT). Passing
    # `port:=8003` as a launch arg does NOT skip the preflight that defaults
    # to :8000.
    exec ./start_scripts/kai/start_autonomy.sh \
      --mode websocket \
      --execution-mode ee_pose \
      --port "$DEFAULT_PORT" \
      host:=127.0.0.1 \
      "$@"
    ;;

  *)
    echo "ERROR: unknown mode '$MODE' (expected server|client)" >&2
    exit 1
    ;;
esac
