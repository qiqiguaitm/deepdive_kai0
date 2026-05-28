#!/bin/bash
# uc01+uc02+uc03 24-GPU HSDP/FSDP 3-host 集群训练启动模板。
# 在 uc01 (master, proc0) 上跑; 经 RDMA eth1 协调 uc02/uc03。
# 详细背景 / NCCL+GDR 配置说明见 docs/deployment/training_ops/submission/uc_cluster_jobs.md §12。
# 用法: 改 CONFIG / EXP_NAME 后 `bash run_cluster_3host.sh`。
set -euo pipefail

CONFIG="<your_config_name>"
EXP_NAME="<exp_name>"
COORD_ADDR="192.168.1.2:15830"
LOG_DIR=/home/tim/workspace/deepdive_kai0/logs
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p $LOG_DIR

NCCL_OPTS='
unset NCCL_IB_DISABLE NCCL_NET_TYPE NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth1
unset NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS NCCL_BUFFSIZE
export NCCL_DEBUG=INFO
'

TRAIN_CMD="cd /home/tim/workspace/deepdive_kai0/kai0 && .venv/bin/python -u scripts/train.py $CONFIG --exp_name=$EXP_NAME --seed=123 --overwrite --no-wandb-enabled"

launch_worker() {
  local TGT=$1 PROC=$2 TAG=$3
  ssh -o StrictHostKeyChecking=no $TGT "
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:\$PATH
export PYTHONUNBUFFERED=1
export KAI0_DATA_ROOT=/home/tim/workspace/deepdive_kai0/kai0
export KAI0_LOCAL_ROOT=/home/tim/local_ckpts
export OPENPI_DATA_HOME=/home/tim/workspace/openpi_cache
export JAX_COORDINATOR_ADDRESS=$COORD_ADDR
export JAX_NUM_PROCESSES=3
export JAX_PROCESS_INDEX=$PROC
export JAX_ENABLE_EMPTY_ARRAYS=true
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1
unset XLA_FLAGS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
$NCCL_OPTS
export WANDB_MODE=offline
mkdir -p $LOG_DIR
nohup bash -c '$TRAIN_CMD' > $LOG_DIR/run_${TAG}_${TIMESTAMP}.log 2>&1 &
echo \"[${TAG} proc${PROC}] pid=\$!\"
disown
"
}

launch_worker "tim@192.168.1.3" 1 "uc02"
launch_worker "tim@192.168.1.4" 2 "uc03"
sleep 5

# uc01 master (local exec — 复制相同 env)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export KAI0_DATA_ROOT=/home/tim/workspace/deepdive_kai0/kai0
export KAI0_LOCAL_ROOT=/home/tim/local_ckpts
export OPENPI_DATA_HOME=/home/tim/workspace/openpi_cache
export JAX_COORDINATOR_ADDRESS=$COORD_ADDR
export JAX_NUM_PROCESSES=3 JAX_PROCESS_INDEX=0
export JAX_ENABLE_EMPTY_ARRAYS=true
export JAX_COMPILATION_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1
unset XLA_FLAGS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
unset NCCL_IB_DISABLE NCCL_NET_TYPE NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23 NCCL_IB_RETRY_CNT=7 NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth1
unset NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS NCCL_BUFFSIZE
export NCCL_DEBUG=INFO
export WANDB_MODE=offline
cd /home/tim/workspace/deepdive_kai0/kai0
nohup .venv/bin/python -u scripts/train.py $CONFIG --exp_name=$EXP_NAME --seed=123 --overwrite --no-wandb-enabled > $LOG_DIR/run_uc01_${TIMESTAMP}.log 2>&1 &
echo "[uc01 proc0] pid=$!"
disown
