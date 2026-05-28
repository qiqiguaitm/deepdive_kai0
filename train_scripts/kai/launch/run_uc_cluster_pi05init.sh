#!/usr/bin/env bash
# UC cluster 3-host pi05init training (from-scratch with pi05_base)
#   Config: pi05_flatten_fold_a_new_pure_200_js
#   Exp:    task_a_pure200_new_norm_base_pi0.5
#   Mesh:   [1, 24] full-FSDP (RDMA + GDR via mlx5_0..3)
#   Init:   pi05_base (NFS shared)
# Usage:    bash run_uc_cluster_pi05init.sh

set -euo pipefail

# ---- experiment ----
CONFIG="pi05_flatten_fold_a_new_pure_200_js"
EXP_NAME="task_a_pure200_new_norm_base_pi0.5"

# ---- NFS-shared paths (visible from uc01/02/03) ----
KAI0_ROOT="/data/shared/ubuntu/workspace/deepdive_kai0/kai0"
PY="$KAI0_ROOT/.venv/bin/python3"
DATA_ROOT="/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200"
VAL_ROOT="/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200_val"
INIT_PARAMS="/data/shared/ubuntu/workspace/base_init_ckpts/pi05_base/params"
CKPT_BASE="/data/shared/ubuntu/workspace/cluster_ckpts"
LOG_DIR="/data/shared/ubuntu/workspace/logs"

# ---- cluster (mlx5 RoCE eth1) ----
UC01_IP=192.168.1.3   # process 0 (coordinator)
UC02_IP=192.168.1.4   # process 1
UC03_IP=192.168.1.2   # process 2
COORD="${UC01_IP}:15830"
NUM_PROCESSES=3

# ---- eval cadence + disk budget ----
INLINE_EVAL_EVERY=4   # eval every 4 saves = every 8k step (减半 vs default every=2)
KEEP_PERIOD=10000     # 只保留 step % 10000 == 0 的 ckpt = 5 ckpts × ~42G = ~210G (NFS free 600G OK)

# ---- ssh: 走 mlx5 内网 IP (uc01 上不必 ~/.ssh/config alias) ----
SSH_UC02="ubuntu@${UC02_IP}"
SSH_UC03="ubuntu@${UC03_IP}"

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

# ---- env block (sourced on each host) ----
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

# ---- train command (common across hosts) ----
TRAIN_CMD="$PY -u $KAI0_ROOT/scripts/train.py $CONFIG \
  --exp-name=$EXP_NAME \
  --fsdp-devices 24 \
  --batch-size 120 \
  --num-workers 64 \
  --inline-eval-every $INLINE_EVAL_EVERY \
  --keep-period $KEEP_PERIOD \
  --data.repo-id $DATA_ROOT \
  --inline-eval-val-root $VAL_ROOT \
  --weight-loader.params-path $INIT_PARAMS \
  --checkpoint-base-dir $CKPT_BASE \
  --no-wandb-enabled \
  --overwrite"

# ---- launch worker on remote host (setsid 真 detach) ----
launch_remote() {
    local ssh_alias=$1 proc_idx=$2 tag=$3
    local log="$LOG_DIR/pi05init_${tag}_${TIMESTAMP}.log"
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

echo "=== launching workers ==="
launch_remote "$SSH_UC02" 1 uc02
launch_remote "$SSH_UC03" 2 uc03
sleep 3

echo "=== launching uc01 process 0 ==="
eval "$ENV_BLOCK"
export JAX_COORDINATOR_ADDRESS=$COORD
export JAX_NUM_PROCESSES=$NUM_PROCESSES
export JAX_PROCESS_INDEX=0
cd "$KAI0_ROOT"
LOG_UC01="$LOG_DIR/pi05init_uc01_${TIMESTAMP}.log"
setsid bash -c "nohup $TRAIN_CMD > $LOG_UC01 2>&1 < /dev/null &"
sleep 1
echo "[uc01] pid=$(pgrep -f 'train.py.*'$CONFIG | head -1) log=$LOG_UC01"

echo
echo "=== launched. tail logs: ==="
echo "  ssh uc01 tail -f $LOG_UC01"
echo "  ssh uc02 tail -f $LOG_DIR/pi05init_uc02_${TIMESTAMP}.log"
echo "  ssh uc03 tail -f $LOG_DIR/pi05init_uc03_${TIMESTAMP}.log"
