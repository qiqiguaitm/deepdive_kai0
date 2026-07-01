#!/usr/bin/env bash
# ============================================================================
# xvla_taskp_local_5090.sh — KAI0/Task_P "pick and place in box" 本地单卡 5090 训练
#
# 官方配方 (X-VLA train.py + README finetune): bf16 | 4 param groups | freeze 1000 |
#   constant LR + warmup2000 | lr 1e-4 | qdur2.0 | ImageNet+ColorJitter。官方本就是单卡训练,
#   本脚本对齐单卡 + 按 5090 32G 显存做 batch 自适应 (+ 可选梯度检查点)。
#
# 用法:
#   ./xvla/xvla_taskp_local_5090.sh smoke           # 200 步 smoke, 验证显存/loss (默认 GPU0)
#   ./xvla/xvla_taskp_local_5090.sh full            # 正式训练 (20k 步)
#   BS=4 ./xvla/xvla_taskp_local_5090.sh smoke      # 显存不够时降 batch
#   GPUS=3 ./xvla/xvla_taskp_local_5090.sh full     # 指定 GPU3
#   GPUS=0,3 ./xvla/xvla_taskp_local_5090.sh full   # 2 卡 DDP (需先停部署释放 GPU3)
#
# 环境变量:
#   BS      per-gpu batch (默认 8; 5090 32G OOM 就降到 4)
#   GPUS    用哪几张卡 (默认 0; 多卡逗号分隔走 torchrun DDP)
#   XVLA_CKPT_INIT  init ckpt 目录 (含 config.json + model.safetensors); 见下方 preflight
#   (训练步数在 config TaskP_local 内 = 20000; 改步数直接编辑该 config)
# ============================================================================
set -euo pipefail

MODE="${1:-smoke}"
# 实测 (5090 32G, grad-ckpt, 解冻后): batch8 峰值 30.07G/32G (临界, ~2G 余量);
# batch6 ≈ 26G (安全余量) → 默认 6 跑无人值守 full; 想榨吞吐用 BS=8 并盯着。
BS="${BS:-6}"
GPUS="${GPUS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KAI0="$REPO_ROOT/kai0"
VENV_PY="$KAI0/.venv_xvla/bin/python"          # 训练同款 env (lerobot.policies.xvla, torch cu128 sm_120)
TRAIN="$REPO_ROOT/xvla/launch/xvla_train.py"
CONV="$REPO_ROOT/xvla/data/joint_to_ee6d.py"

# ── 训练 env (offline + 数据/ckpt/tokenizer 根) ──────────────────────────────
export XVLA_SB="$REPO_ROOT/xvla/data/self_built"                      # EE6D 数据根 (config 的 {SB})
export XVLA_CKPT_INIT="${XVLA_CKPT_INIT:-$REPO_ROOT/xvla/xvla_ckpts}" # init ckpt (X-VLA-Pt lerobot 格式)
export XVLA_BART_TOK="$REPO_ROOT/xvla/assets/bart-large-tokenizer"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

TASKP_RAW="/data1/DATA_IMP/KAI0/Task_P/base/2026-04-21"
TASKP_EE6D="$XVLA_SB/TaskP_ee6d/2026-04-21"
OUT="$REPO_ROOT/xvla/ckpts/xvla_taskp_local"

RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'; NC='\033[0m'
die(){ echo -e "${RED}ERROR: $*${NC}" >&2; exit 1; }

[ -x "$VENV_PY" ] || die "训练 env 缺失: $VENV_PY (需要 kai0/.venv_xvla)"

# ── ① 数据准备: Task_P 原始 joint-14D → EE6D parquet (幂等, 已转则跳过) ───────────
if [ ! -f "$TASKP_EE6D/meta/info.json" ]; then
  echo -e "${YEL}[prep] Task_P 未转 EE6D, 运行 joint_to_ee6d ...${NC}"
  [ -f "$TASKP_RAW/meta/info.json" ] || die "Task_P 原始数据缺失: $TASKP_RAW"
  "$VENV_PY" "$CONV" --in_dir "$TASKP_RAW" --out_dir "$TASKP_EE6D" --workers 16
else
  echo -e "${GRN}[prep] EE6D 数据已存在: $TASKP_EE6D${NC}"
fi

# ── ② preflight: init ckpt / tokenizer ──────────────────────────────────────
[ -f "$XVLA_BART_TOK/tokenizer.json" ] || die "bart tokenizer 缺失: $XVLA_BART_TOK/tokenizer.json"
if [ ! -f "$XVLA_CKPT_INIT/config.json" ] || [ ! -f "$XVLA_CKPT_INIT/model.safetensors" ]; then
  cat >&2 <<EOF
$(echo -e "${RED}init ckpt 缺失:${NC} $XVLA_CKPT_INIT (需 config.json + model.safetensors, lerobot 格式)")
  本地暂无 X-VLA-Pt 基座权重。二选一后重跑:
    A) 下载官方 lerobot 基座 (推荐, 最贴官方 finetune):
         huggingface-cli download lerobot/xvla-base --local-dir $XVLA_CKPT_INIT
    B) 用本地已有 ckpt 热启动 (零下载, 同机械臂): 指定 XVLA_CKPT_INIT 指向一个含
         config.json+model.safetensors 的 lerobot 格式目录 (本地 E0 ckpt 是 state_dict.pt, 需先 repack)。
EOF
  exit 13
fi

# ── ③ 启动 ──────────────────────────────────────────────────────────────────
IFS=',' read -r -a _G <<< "$GPUS"; NGPU="${#_G[@]}"
SMOKE_ARGS=(); [ "$MODE" = "smoke" ] && SMOKE_ARGS=(--max_steps 200)
mkdir -p "$OUT"

echo -e "${GRN}=== Task_P 本地训练 | mode=$MODE | GPUS=$GPUS (n=$NGPU) | per-gpu BS=$BS ===${NC}"
echo "  init=$XVLA_CKPT_INIT"
echo "  data=$TASKP_EE6D"
echo "  out=$OUT"

if [ "$MODE" = "smoke" ] || [ "$NGPU" -eq 1 ]; then
  # 单卡 (smoke 一律单卡; full 单卡也走这里)
  CUDA_VISIBLE_DEVICES="${_G[0]}" "$VENV_PY" "$TRAIN" \
    --config TaskP_local --output_dir "$OUT" \
    --batch_size "$BS" --grad_checkpointing --workers 4 \
    "${SMOKE_ARGS[@]}"
else
  # 多卡 DDP (full): 注意 GPU3 若被部署占用需先停部署
  CUDA_VISIBLE_DEVICES="$GPUS" "$VENV_PY" -m torch.distributed.run \
    --standalone --nproc_per_node="$NGPU" "$TRAIN" \
    --config TaskP_local --output_dir "$OUT" \
    --batch_size "$BS" --grad_checkpointing --workers 4
fi
