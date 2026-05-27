#!/usr/bin/env bash
# xvla exp1 hard-prompt mixed on uc01+02 (16 GPU FSDP=16).
#   Config:  xvla_exp1_hard_prompt_merged_uc (uc-local data paths)
#   Init:    pi05_base (NFS base_init_ckpts)
#   Data:    Kai0_official/Task_A/{base,dagger} + Task_A/vis_v2_merged (uc-local)
#   Mirrors: deepdive_kai0/xvla/data/mixed_hard/* with patched tasks.jsonl
#   Mesh:    [1, 16] full-FSDP, RDMA via mlx5_0..3
#
# Prerequisite: rsync of vis_v2_merged + val done; build_mixed_hard_uc.py executed.
# Submit from local: bash this script.

set -euo pipefail

CONFIG="xvla_exp1_hard_prompt_merged_uc"
EXP_NAME="xvla_exp1_hard_prompt_merged_uc"

KAI0_ROOT="/data/shared/ubuntu/workspace/deepdive_kai0/kai0"
PY="$KAI0_ROOT/.venv/bin/python3"
CKPT_BASE="/data/shared/ubuntu/workspace/cluster_ckpts"  # NFS-shared from uc01 — required for orbax multi-host atomic save
LOG_DIR="/data/shared/ubuntu/workspace/logs"

UC01_IP=192.168.1.3
UC02_IP=192.168.1.4
COORD="${UC01_IP}:18831"  # bumped port to avoid stale binding from prior failed runs
NUM_PROCESSES=2

INLINE_EVAL_EVERY=4
KEEP_PERIOD=10000

SSH_UC02="ubuntu@${UC02_IP}"
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR" 2>/dev/null || true

ENV_BLOCK=$(cat <<'EOF'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
unset NCCL_IB_DISABLE NCCL_NET_TYPE NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ
unset XLA_FLAGS
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth1
export NCCL_DEBUG=INFO  # verbose to debug the multi-host handshake hang we hit in v6-v9
export JAX_ENABLE_EMPTY_ARRAYS=true
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
EOF
)

# Pre-clean ckpt dir BEFORE launching workers — otherwise process 0's rmtree+mkdir
# races against process 1's CheckpointManager init, producing
# "sync_global_devices name mismatch" on orbax barrier_sync_key_prefix.
echo "=== pre-cleaning ckpt dir ==="
rm -rf "$CKPT_BASE/$CONFIG/$EXP_NAME" 2>/dev/null || true
mkdir -p "$CKPT_BASE/$CONFIG/$EXP_NAME"

TRAIN_CMD="$PY -u $KAI0_ROOT/scripts/train.py $CONFIG \
  --exp-name=$EXP_NAME \
  --fsdp-devices 16 \
  --batch-size 128 \
  --num-workers 16 \
  --inline-eval-every $INLINE_EVAL_EVERY \
  --keep-period $KEEP_PERIOD \
  --checkpoint-base-dir $CKPT_BASE \
  --no-wandb-enabled \
  --resume"

launch_remote() {
    local ssh_alias=$1 proc_idx=$2 tag=$3
    local log="$LOG_DIR/xvla_exp1_uc_${tag}_${TIMESTAMP}.log"
    ssh -o StrictHostKeyChecking=no "$ssh_alias" "
$ENV_BLOCK
export JAX_COORDINATOR_ADDRESS=$COORD
export JAX_NUM_PROCESSES=$NUM_PROCESSES
export JAX_PROCESS_INDEX=$proc_idx
mkdir -p $LOG_DIR
cd $KAI0_ROOT
setsid bash -c 'nohup $TRAIN_CMD > $log 2>&1 < /dev/null &'
sleep 1
pgrep -f 'train.py.*$CONFIG' | head -1 | xargs -I{} echo '[${tag}] pid={} log=$log'
"
}

# Launch leader (uc01, process 0) FIRST so the JAX coordinator is listening before
# uc02 tries to connect. Then sleep + launch uc02. Avoids NCCL rendezvous deadlock
# when process 0 starts after process 1 has already joined.
echo "=== launching uc01 (process 0, JAX coordinator/leader) ==="
eval "$ENV_BLOCK"
export JAX_COORDINATOR_ADDRESS=$COORD
export JAX_NUM_PROCESSES=$NUM_PROCESSES
export JAX_PROCESS_INDEX=0
cd "$KAI0_ROOT"
LOG_UC01="$LOG_DIR/xvla_exp1_uc_uc01_${TIMESTAMP}.log"
setsid bash -c "nohup $TRAIN_CMD > $LOG_UC01 2>&1 < /dev/null &"
sleep 5
echo "[uc01] pid=$(pgrep -f 'train.py.*'$CONFIG | head -1) log=$LOG_UC01"

echo "=== launching uc02 (process 1) ==="
launch_remote "$SSH_UC02" 1 uc02

echo
echo "tail -f $LOG_UC01"
