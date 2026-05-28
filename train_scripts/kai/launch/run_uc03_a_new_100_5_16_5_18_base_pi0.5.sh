#!/usr/bin/env bash
# UC03 single-host (8 GPU) — 5/16-v2 (2 ep) + 5/18-v2 (100 ep) = 102 ep merged, init pi0.5
#   Config:  pi05_flatten_fold_a_new_100_5_16_5_18_base_pi0.5
#   Exp:     task_a_new_100_new_norm_base_pi0.5  (per user request)
#   Data:    /data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_100_5_16_5_18 (48 ep / 101,589 frames)
#   Norm:    recomputed via compute_norm_states_fast.py for this dataset
#   Init:    pi05_base (NFS shared base_init_ckpts)
#   Ckpt:    uc03 local SSD (single-host, no NFS multi-host concern)
# Run on uc03: bash run_uc03_a_new_100_5_16_5_18_base_pi0.5.sh

set -euo pipefail

CONFIG="pi05_flatten_fold_a_new_100_5_16_5_18_base_pi0.5"
EXP_NAME="task_a_new_100_new_norm_base_pi0.5"

KAI0_ROOT="/data/shared/ubuntu/workspace/deepdive_kai0/kai0"
PY="$KAI0_ROOT/.venv/bin/python3"
CKPT_BASE="/data/shared/ubuntu/local_ckpts"
LOG_DIR="/data/shared/ubuntu/workspace/logs"

INLINE_EVAL_EVERY=4
KEEP_PERIOD=10000

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/${EXP_NAME}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
unset JAX_COORDINATOR_ADDRESS JAX_NUM_PROCESSES JAX_PROCESS_INDEX
unset XLA_FLAGS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1

TRAIN_CMD="$PY -u $KAI0_ROOT/scripts/train.py $CONFIG \
  --exp-name=$EXP_NAME \
  --fsdp-devices 8 \
  --batch-size 120 \
  --num-workers 64 \
  --inline-eval-every $INLINE_EVAL_EVERY \
  --keep-period $KEEP_PERIOD \
  --checkpoint-base-dir $CKPT_BASE \
  --no-wandb-enabled \
  --resume"

cd "$KAI0_ROOT"
setsid bash -c "nohup $TRAIN_CMD > $LOG 2>&1 < /dev/null &"
sleep 2
PID=$(pgrep -f "train.py.*$CONFIG" | head -1)
echo "[uc03] pid=$PID log=$LOG"
echo "tail -f $LOG"
