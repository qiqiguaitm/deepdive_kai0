#!/usr/bin/env bash
# ============================================================================
# xvla_repack_deploy_taskp.sh — 把 5090 训练的 TaskP ckpt 装上 sidecar 部署目录
#
# 用法:
#   ./xvla/xvla_repack_deploy_taskp.sh                    # 预测式(默认): repack + 试启动 serve-policy
#   ./xvla/xvla_repack_deploy_taskp.sh --repack-only      # 只建部署目录, 不启动 server
#   ./xvla/xvla_repack_deploy_taskp.sh --skip-repack       # 部署目录已存在, 直接启动
#
# 依赖: 训练 ckpt 在 xvla/ckpts/xvla_taskp_local/step_final/state_dict.pt
#
# 产出:
#   /data1/DATA_IMP/checkpoints/ckpt_xvla/xvla_taskp_local_5090_step_final/
#     ├── state_dict.pt    (copy)
#     ├── config.json      (copy, with domain_id=22, imagenet_norm, etc.)
#     └── sidecar.json     (serve 运行时参数: domain_id/prompt/image_norm/action_format 等)
#
# 验证通过后, 即可:
#   ./xvla/start_xvla_from_ckpt.sh xvla_taskp_local_5090_step_final --execute
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
CKPT_SRC="$REPO_ROOT/xvla/ckpts/xvla_taskp_local/step_final"
CKPT_BASE="/data1/DATA_IMP/checkpoints/ckpt_xvla"
CKPT_NAME="xvla_taskp_local_5090_step_final"
CKPT_DIR="$CKPT_BASE/$CKPT_NAME"

MODE="${1:-full}"  # full | repack-only | skip-repack

RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'; NC='\033[0m'
die(){ echo -e "${RED}ERROR: $*${NC}" >&2; exit 1; }

# ── ① Repack ─────────────────────────────────────────────────────────────────
if [ "$MODE" != "skip-repack" ]; then
  [ -f "$CKPT_SRC/state_dict.pt" ] || die "训练 ckpt 缺失: $CKPT_SRC/state_dict.pt (先跑训练)"

  echo -e "${YEL}[repack] 源头: $CKPT_SRC${NC}"
  echo -e "${YEL}[repack] 目标: $CKPT_DIR${NC}"

  mkdir -p "$CKPT_DIR"

  # 1) state_dict.pt (copy, preserve size verify)
  echo "[repack] copying state_dict.pt (3.5 GB) ..."
  cp --preserve=all "$CKPT_SRC/state_dict.pt" "$CKPT_DIR/state_dict.pt"
  SRC_SZ=$(stat -c%s "$CKPT_SRC/state_dict.pt")
  DST_SZ=$(stat -c%s "$CKPT_DIR/state_dict.pt")
  [ "$SRC_SZ" -eq "$DST_SZ" ] || die "size mismatch: src=$SRC_SZ dst=$DST_SZ"
  echo -e "${GRN}[repack] state_dict.pt OK (${SRC_SZ} bytes)${NC}"

  # 2) config.json (copy from training run)
  if [ -f "$CKPT_SRC/../config.json" ]; then
    cp "$CKPT_SRC/../config.json" "$CKPT_DIR/config.json"
    echo "[repack] config.json copied from training config"
  else
    echo "[repack] WARN: no config.json at $CKPT_SRC/../config.json"
  fi

  # 3) sidecar.json (serve 所需的部署参数, 参考 E0 fixedcam sidecar)
  cat > "$CKPT_DIR/sidecar.json" <<JSON
{
  "model_family": "xvla",
  "base_config_path": "kai0/assets/xvla/lerobot_base",
  "step": 20000,
  "source": "local sim01 1×5090 — xvla/ckpts/xvla_taskp_local/step_final/",
  "training_datasets": [
    {"root": "$REPO_ROOT/xvla/data/self_built/TaskP_ee6d/2026-04-21", "domain_id": 22, "weight": 1.0, "type": "parquet", "prompt": "pick and place in box"}
  ],
  "deploy_domain_id": 22,
  "deploy_prompt": "pick and place in box",
  "action_format": "ee6d_interleaved",
  "action_dim": 20,
  "action_chunk": 30,
  "num_domains": 30,
  "image_norm": "imagenet",
  "use_proprio": true,
  "action_qdur": 2.0,
  "deploy_publish_rate": 15,
  "deploy_xvla_sequential": true,
  "deploy_timing_note": "action_qdur=2.0 + action_chunk=30 → 30 anchors span 2.0s. publish_rate=15 (=H/qdur). xvla_sequential=true: 整 chunk 开环执行.",
  "notes": "TaskP_local_5090 — 首版本地单卡 X-VLA 微调。Task_P (1 日期, 23777 sample, EE6D) + lerobot xvla-base init + 官方配方(bf16/4group_official/freeze1000/warmup2000/const LR 1e-4/grad-ckpt/bs6). 20000 步, loss 23.2→0.28 健康收敛, ~80min. loss-only (无 offline MAE). 真机部署前建议先 vision-ablation 确认 d_img/d_state 健康."
}
JSON
  echo -e "${GRN}[repack] sidecar.json created${NC}"

  echo -e "${GRN}[repack] ✅ 部署目录已就绪: $CKPT_DIR${NC}"
  echo "  ├── state_dict.pt (verified: $DST_SZ bytes)"
  echo "  ├── config.json"
  echo "  └── sidecar.json  (domain=22, prompt='pick and place in box', image_norm=imagenet)"
fi

# ── ② 试启动 (serve-only, observe 模式不驱动臂) ───────────────────────────────
if [ "$MODE" = "full" ]; then
  if [ ! -f "$CKPT_DIR/state_dict.pt" ] || [ ! -f "$CKPT_DIR/sidecar.json" ]; then
    die "部署目录不完整: $CKPT_DIR (缺 state_dict.pt 或 sidecar.json)"
  fi

  echo -e "${YEL}[serve] 试启动 server (observe-only, 不驱动臂, 按 Ctrl+C 停止)...${NC}"
  echo -e "${YEL}[serve] 确认 pipeline 无异常后, 另开终端运行:${NC}"
  echo -e "${GRN}  ./xvla/start_xvla_from_ckpt.sh $CKPT_NAME --execute $*${NC}"
  echo ""
  echo -e "    或带上 gripper 迟滞 (实验 4 补丁):"
  echo -e "${GRN}  XVLA_SERVER_ARGS='--proprio_feedback --gripper_close_thr 0.6 --gripper_open_thr 0.35 --gripper_min_hold 10' \\\\${NC}"
  echo -e "${GRN}    ./xvla/start_xvla_from_ckpt.sh $CKPT_NAME --execute${NC}"
  echo ""

  # 仅指定 ckpt 目录, 不加 --execute → observe-only。剩余 args 传给 stack。
  exec "$REPO_ROOT/xvla/start_xvla_from_ckpt.sh" "$CKPT_DIR" "$@"
fi

if [ "$MODE" = "repack-only" ]; then
  echo -e "${GRN}[done] repack-only 完成。要启动 server 请运行:${NC}"
  echo "  ./xvla/xvla_repack_deploy_taskp.sh --skip-repack"
fi
