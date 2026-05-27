#!/usr/bin/env bash
# uc01 single-host (8 GPU) — xvla exp1 hard prompt baseline.
#   Config:  xvla_exp1_hard_prompt_merged_uc (single pre-merged dataset, 7407 ep)
#   Exp:     xvla_exp1_hard_prompt_merged_uc
#   Init:    pi05_base (NFS shared)
#   Mode:    single-host, avoids the orbax/NCCL multi-host issue we hit in v6-v15
# Run on uc01: bash run_uc01_xvla_exp1_hard_single.sh

set -euo pipefail

CONFIG="xvla_exp1_hard_prompt_merged_uc"
EXP_NAME="xvla_exp1_hard_prompt_merged_uc"

KAI0_ROOT="/data/shared/ubuntu/workspace/deepdive_kai0/kai0"
PY="$KAI0_ROOT/.venv/bin/python3"
CKPT_BASE="/data/shared/ubuntu/local_ckpts"  # uc01 LOCAL SSD (single-host, no NFS multi-host concern)
LOG_DIR="/data/shared/ubuntu/workspace/logs"

INLINE_EVAL_EVERY=4
KEEP_PERIOD=10000

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/${EXP_NAME}_single_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
unset JAX_COORDINATOR_ADDRESS JAX_NUM_PROCESSES JAX_PROCESS_INDEX
unset XLA_FLAGS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1

# Note: batch_size 120 (single-host convention, divisible by 8 = 15/GPU)
TRAIN_CMD="$PY -u $KAI0_ROOT/scripts/train.py $CONFIG \
  --exp-name=$EXP_NAME \
  --fsdp-devices 8 \
  --batch-size 120 \
  --num-workers 64 \
  --inline-eval-every $INLINE_EVAL_EVERY \
  --keep-period $KEEP_PERIOD \
  --checkpoint-base-dir $CKPT_BASE \
  --no-wandb-enabled \
  --overwrite"

cd "$KAI0_ROOT"
setsid bash -c "nohup $TRAIN_CMD > $LOG 2>&1 < /dev/null &"
sleep 2
PID=$(pgrep -f "train.py.*$CONFIG" | head -1)
echo "[uc01] pid=$PID log=$LOG"
echo "tail -f $LOG"
