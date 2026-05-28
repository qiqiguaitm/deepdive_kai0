#!/usr/bin/env bash
# UC03 single-host (8 GPU) pi05init RESUME from step 4000
#   Config:  pi05_flatten_fold_a_new_pure_200_js
#   Exp:     task_a_pure200_new_norm_base_pi0.5
#   Mesh:    [8] FSDP (matches saved ckpt)
#   Ckpt:    /data/shared/ubuntu/local_ckpts/.../4000/ (local SSD, fast)
# Run on uc03 directly: bash run_uc03_pi05init_resume.sh

set -euo pipefail

# ---- experiment ----
CONFIG="pi05_flatten_fold_a_new_pure_200_js"
EXP_NAME="task_a_pure200_new_norm_base_pi0.5"

# ---- paths (mix of NFS-shared inputs + uc03-local ckpt I/O) ----
KAI0_ROOT="/data/shared/ubuntu/workspace/deepdive_kai0/kai0"           # NFS shared code+venv
PY="$KAI0_ROOT/.venv/bin/python3"
DATA_ROOT="/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200"
VAL_ROOT="/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200_val"
INIT_PARAMS="/data/shared/ubuntu/workspace/base_init_ckpts/pi05_base/params"
CKPT_BASE="/data/shared/ubuntu/local_ckpts"                            # uc03 LOCAL SSD (fast)
LOG_DIR="/data/shared/ubuntu/workspace/logs"

# ---- eval cadence + disk budget ----
INLINE_EVAL_EVERY=4
KEEP_PERIOD=10000

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/pi05init_uc03_resume_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# ---- env (single host, no JAX_NUM_PROCESSES, no NCCL multinode flags) ----
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
  --data.repo-id $DATA_ROOT \
  --inline-eval-val-root $VAL_ROOT \
  --weight-loader.params-path $INIT_PARAMS \
  --checkpoint-base-dir $CKPT_BASE \
  --no-wandb-enabled \
  --resume"

cd "$KAI0_ROOT"
setsid bash -c "nohup $TRAIN_CMD > $LOG 2>&1 < /dev/null &"
sleep 2
PID=$(pgrep -f "train.py.*$CONFIG" | head -1)
echo "[uc03] pid=$PID log=$LOG"
echo "tail -f $LOG"
