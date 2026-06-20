#!/usr/bin/env bash
# Full FD posttrain: Cosmos3-Nano (BASE) -> wam_fold_wm, 8-GPU FSDP single node.
# Tier 1+2+3 data: wam_fold_v1 + robocoin + AgiBot (18,441 unique eps).
# Base DCP must already exist at $BASE_CKPT_DCP (run smoke_validate.sh once to create it).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'

CF=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3
VENV=$CF/.venv
TOML=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/train/recipe_wm_nano.toml

export PYTHONPATH="$CF:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH=/mnt/pfs/p46h4f/cosmos/uvbin:$PATH
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home
export HF_HUB_OFFLINE=1

export BASE_CKPT_DCP=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/checkpoints/Cosmos3-Nano-dcp
export WAN_VAE_PATH=/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
export WAM_WM_LATENT_CACHE=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/latent_cache

export IMAGINAIRE_OUTPUT_ROOT=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/train_out
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT" "$WAM_WM_LATENT_CACHE"

cd "$CF"

NGPU=${NGPU:-8}
MAXITER=${MAXITER:-10000}
SAVEITER=${SAVEITER:-500}
PORT=$(( ( $$ % 20000 ) + 31500 ))

echo "=== WM FULL TRAIN start $(date) | ngpu=$NGPU max_iter=$MAXITER save_iter=$SAVEITER ==="
"$VENV/bin/torchrun" --nproc_per_node=$NGPU --master_port=$PORT \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- \
  trainer.max_iter=$MAXITER \
  checkpoint.save_iter=$SAVEITER
echo "=== WM_FULL_TRAIN_DONE rc=$? $(date) ==="
