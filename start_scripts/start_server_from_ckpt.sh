#!/bin/bash
# 通用 JAX policy server 启动脚本: 从 ckpt 内 train_config.json sidecar 加载配置,
# 无需修改 src/openpi/training/config.py.
#
# 使用:
#   ./start_scripts/start_server_from_ckpt.sh <ckpt_dir> [--port N] [extra serve_policy args...]
#
# 要求:
#   <ckpt_dir>/train_config.json  形如 { "base_config_name": "...", "override_asset_id": "..." }
#   <ckpt_dir>/assets/<override_asset_id>/norm_stats.json
#   <ckpt_dir>/_CHECKPOINT_METADATA
#
# 可选环境变量:
#   OPENPI_FIXED_NOISE_SEED=<int>  启用 G0 fixed-noise inference (vis_v2_full 真机修复, 见
#                                  docs/deployment/inference/fixed_noise_inference_fix.md).
#                                  未设置 = 原行为, 每次 infer 内部随机 sample noise.
#   OPENPI_ENABLE_RTC=1            启用 Pi0Config → Pi0RTCConfig swap. ros2
#                                  policy_inference_node 默认 enable_rtc=True 会发
#                                  execute_horizon / prev_action_chunk 等 kwargs,
#                                  Pi0.sample_actions 不接受这些参数会爆 TypeError;
#                                  需走 ros2+RTC client 时必须设. 未设 = 原行为,
#                                  非 RTC client (V1 mock / serve_policy 老用法) 兼容.
#   PORT=<int>                     server 端口, 默认 8000.

set -euo pipefail

CKPT_DIR="${1:?Usage: $0 <ckpt_dir> [extra serve_policy args...]}"
shift || true

PORT="${PORT:-8000}"

if [ ! -d "$CKPT_DIR" ]; then
  echo "ERROR: ckpt_dir does not exist: $CKPT_DIR" >&2
  exit 1
fi
if [ ! -f "$CKPT_DIR/train_config.json" ]; then
  echo "ERROR: $CKPT_DIR/train_config.json not found." >&2
  echo "       Use train_scripts/data/pack_inference_ckpt.py to produce it." >&2
  exit 1
fi
if [ ! -f "$CKPT_DIR/_CHECKPOINT_METADATA" ]; then
  echo "ERROR: $CKPT_DIR/_CHECKPOINT_METADATA missing — invalid ckpt dir." >&2
  exit 1
fi

export OPENPI_EXTRA_CONFIG="$CKPT_DIR/train_config.json"
CONFIG_NAME=$(/data1/miniconda3/bin/python -c "import json; print(json.load(open('$CKPT_DIR/train_config.json'))['base_config_name'])")
ASSET_ID=$(/data1/miniconda3/bin/python -c "import json; print(json.load(open('$CKPT_DIR/train_config.json')).get('override_asset_id', ''))")

if [ -n "$ASSET_ID" ] && [ ! -f "$CKPT_DIR/assets/$ASSET_ID/norm_stats.json" ]; then
  echo "ERROR: $CKPT_DIR/assets/$ASSET_ID/norm_stats.json missing (override_asset_id mismatch)" >&2
  exit 1
fi

echo "[start_server_from_ckpt] config_name=$CONFIG_NAME asset_id=$ASSET_ID port=$PORT ckpt=$CKPT_DIR"
if [ -n "${OPENPI_FIXED_NOISE_SEED:-}" ]; then
  echo "[start_server_from_ckpt] G0 fixed-noise enabled (seed=$OPENPI_FIXED_NOISE_SEED)"
fi
if [ -n "${OPENPI_ENABLE_RTC:-}" ]; then
  echo "[start_server_from_ckpt] Pi0→Pi0RTC swap enabled (for ros2 RTC client)"
fi

# RTX 5090 Blackwell 必需: XLA autotuner SIGSEGV 规避. 允许外部覆盖.
export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_autotune_level=0"}"
export JAX_COMPILATION_CACHE_DIR=/data1/tim/workspace/deepdive_kai0/.xla_cache
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd /data1/tim/workspace/deepdive_kai0/kai0
exec .venv/bin/python scripts/serve_policy.py --port "$PORT" \
  policy:checkpoint --policy.config="$CONFIG_NAME" \
  --policy.dir="$CKPT_DIR" \
  "$@"
