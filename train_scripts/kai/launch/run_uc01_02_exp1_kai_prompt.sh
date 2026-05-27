#!/usr/bin/env bash
# Experiment 1: uc01+02 2-host 16 GPU FSDP=16 — kai0 official + "kai " prompt prefix
#   Config:   pi05_flatten_fold_kai0_official_kai_prompt
#   Init:     pi05_base (NFS base_init_ckpts)
#   Mesh:     [1, 16] full-FSDP (NCCL RDMA + GDR via mlx5_0..3)
#   Data:     kai0_base + kai0_dagger (待 download_dataset.py 完成 + 路径 verify)
#
# Submit from local: scp this to uc01 NFS, then ssh uc01 bash.

set -euo pipefail

CONFIG="pi05_flatten_fold_kai0_official_kai_prompt"
EXP_NAME="exp1_kai_official_kai_prompt"

KAI0_ROOT="/data/shared/ubuntu/workspace/deepdive_kai0/kai0"
PY="$KAI0_ROOT/.venv/bin/python3"
INIT_PARAMS="/data/shared/ubuntu/workspace/base_init_ckpts/pi05_base/params"
CKPT_BASE="/data/shared/ubuntu/workspace/cluster_ckpts"
LOG_DIR="/data/shared/ubuntu/workspace/logs"

# NOTE: data path 待 download 完成后 config.py 内 vePFS paths 需替换为 uc 本地路径!
# vePFS paths under /vePFS/... 在 uc 上不存在; 必须 mirror data + 更新 config.py paths
# 或 user manually verify后切换.

UC01_IP=192.168.1.3   # process 0
UC02_IP=192.168.1.4   # process 1
COORD="${UC01_IP}:15830"
NUM_PROCESSES=2

INLINE_EVAL_EVERY=4
KEEP_PERIOD=10000

SSH_UC02="ubuntu@${UC02_IP}"

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

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
export NCCL_DEBUG=WARN
export JAX_ENABLE_EMPTY_ARRAYS=true
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
EOF
)

TRAIN_CMD="$PY -u $KAI0_ROOT/scripts/train.py $CONFIG \
  --exp-name=$EXP_NAME \
  --fsdp-devices 16 \
  --batch-size 128 \
  --num-workers 16 \
  --inline-eval-every $INLINE_EVAL_EVERY \
  --keep-period $KEEP_PERIOD \
  --checkpoint-base-dir $CKPT_BASE \
  --no-wandb-enabled \
  --overwrite"

launch_remote() {
    local ssh_alias=$1 proc_idx=$2 tag=$3
    local log="$LOG_DIR/exp1_kai_prompt_${tag}_${TIMESTAMP}.log"
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

echo "=== launching uc02 (process 1) ==="
launch_remote "$SSH_UC02" 1 uc02
sleep 3

echo "=== launching uc01 (process 0) ==="
eval "$ENV_BLOCK"
export JAX_COORDINATOR_ADDRESS=$COORD
export JAX_NUM_PROCESSES=$NUM_PROCESSES
export JAX_PROCESS_INDEX=0
cd "$KAI0_ROOT"
LOG_UC01="$LOG_DIR/exp1_kai_prompt_uc01_${TIMESTAMP}.log"
setsid bash -c "nohup $TRAIN_CMD > $LOG_UC01 2>&1 < /dev/null &"
sleep 2
echo "[uc01] pid=$(pgrep -f 'train.py.*'$CONFIG | head -1) log=$LOG_UC01"

echo
echo "tail -f $LOG_UC01"
