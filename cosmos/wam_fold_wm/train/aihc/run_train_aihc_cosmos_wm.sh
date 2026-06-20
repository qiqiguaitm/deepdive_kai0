#!/bin/bash
# AIHC multi-node launcher for Cosmos3-Nano (BASE) -> wam_fold_wm FD world-model.
# Mirrors wam_fold_policy/train/aihc/run_train_aihc_cosmos.sh; key differences:
#   - BASE_CKPT_DCP = Cosmos3-Nano-dcp (not Policy-DROID)
#   - TOML          = recipe_wm_nano.toml (FD mode, chunk_length=32)
#   - data source   = build_tier12_data_source (Tier 0+1+2+3, 18,441 eps)
#   - WAM_WM_LATENT_CACHE for video latent caching
#   - REPLICATE_DEGREE = num_nodes (override for FSDP+DDP across nodes)
set -e
COS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos
CF=$COS/packages/cosmos3
VENV=$CF/.venv
RUNS=$COS/wam_fold_wm_runs
cd "$CF"

# ---- populate the venv's node-local python on this pod ----
UVPY=/root/.local/share/uv/python/cpython-3.13.0-linux-x86_64-gnu
if [ ! -x "$UVPY/bin/python3.13" ]; then
  mkdir -p /root/.local/share/uv/python
  cp -an /mnt/pfs/p46h4f/cosmos/uvpy/cpython-3.13.0-linux-x86_64-gnu /root/.local/share/uv/python/ 2>/dev/null || true
fi

# ---- distributed topology (AIHC-injected) ----
export NUM_GPUS=${NUM_GPUS:-8}
export NNODES=${NNODES:-${WORLD_SIZE:-5}}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29505}

# ---- cosmos env ----
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; export no_proxy='*'
export PYTHONPATH="$CF" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH=/mnt/pfs/p46h4f/cosmos/uvbin:$PATH
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WAN_VAE_PATH=/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
export BASE_CKPT_DCP=$RUNS/checkpoints/Cosmos3-Nano-dcp
export WAM_WM_LATENT_CACHE=$RUNS/latent_cache
export CKPT_DIR=${CKPT_DIR:-$RUNS/train_out_5n8g}
export IMAGINAIRE_OUTPUT_ROOT="$CKPT_DIR"
mkdir -p "$CKPT_DIR" "$WAM_WM_LATENT_CACHE"
export WANDB_MODE=offline WANDB_DIR="$CKPT_DIR/wandb" WANDB__SERVICE_WAIT=300
mkdir -p "$CKPT_DIR/wandb"

# ---- RDMA/IB ----
IBROOT=/mnt/pfs/p46h4f/cosmos/dreamzero/ibverbs/root
if [ -d "$IBROOT" ] && ! ldconfig -p 2>/dev/null | grep -qi libibverbs; then
  cp -an "$IBROOT/usr/lib/x86_64-linux-gnu/." /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
  cp -an "$IBROOT/lib/x86_64-linux-gnu/."     /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
  mkdir -p /etc/libibverbs.d && cp -an "$IBROOT/etc/libibverbs.d/." /etc/libibverbs.d/ 2>/dev/null || true
  ldconfig 2>/dev/null || true
fi
export LD_LIBRARY_PATH="/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib:$IBROOT/usr/lib/x86_64-linux-gnu:$IBROOT/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
if ls /sys/class/infiniband >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -qi libibverbs; then
  export NCCL_IB_DISABLE=0 NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
else
  export NCCL_IB_DISABLE=1
fi
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ---- CPU-RAM OOM mitigation (same as policy run) ----
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:-0}
( while true; do sync; echo 1 > /proc/sys/vm/drop_caches 2>/dev/null; sleep 20; done ) &
( while true; do { echo -n "$(date +%H:%M:%S) "; free -g | sed -n 2p; } >> "$CKPT_DIR/mem_node${NODE_RANK}.log" 2>/dev/null; sleep 60; done ) &

# ---- training args ----
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_ITER=${SAVE_ITER:-500}
TOML=$COS/wam_fold_wm/train/recipe_wm_nano.toml
EXTRA=${EXTRA:-}

# FSDP-shard within node (8), DDP-replicate across nodes (NNODES).
REPLICATE_DEGREE=${REPLICATE_DEGREE:-$NNODES}
# TOML [model.parallelism] maps to model.config.parallelism via VFM PATH_REMAPS catch-all.
# Must use the post-remap Hydra path on the command line (not the TOML-side model.parallelism).
PARALLELISM_OVERRIDE="model.config.parallelism.data_parallel_replicate_degree=$REPLICATE_DEGREE"

# LambdaCosine cycle must span the full run (crash fix from policy run).
MFU_OVERRIDES="scheduler.cycle_lengths=[${SCHED_CYCLE:-$MAX_STEPS}] $PARALLELISM_OVERRIDE"

RANKLOG="$CKPT_DIR/train_node${NODE_RANK}.log"
echo "[aihc-wm] node $NODE_RANK/$NNODES gpus=$NUM_GPUS total=$((NNODES*NUM_GPUS)) master=$MASTER_ADDR:$MASTER_PORT"
echo "[aihc-wm] max_iter=$MAX_STEPS save_iter=$SAVE_ITER ckpt_dir=$CKPT_DIR"
echo "[aihc-wm] replicate_degree=$REPLICATE_DEGREE (FSDP-8 within node, DDP across nodes)"
echo "[aihc-wm] teeing full output to $RANKLOG"

"$VENV/bin/python" -c 'import torch; print("[aihc-wm] torch",torch.__version__,"cuda",torch.cuda.is_available(),torch.cuda.device_count())' || true

set -o pipefail
"$VENV/bin/torchrun" \
  --nnodes="$NNODES" --nproc_per_node="$NUM_GPUS" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- \
  trainer.max_iter="$MAX_STEPS" checkpoint.save_iter="$SAVE_ITER" $MFU_OVERRIDES $EXTRA \
  2>&1 | tee "$RANKLOG"
