#!/bin/bash
# Task_A pure_vis600 cold-start training on gf1.
# Pair experiment with mix_vis600 (gf0) — same hyperparams, same step count, different data.
#
# Dataset: 560 train (289 orig + 271 mir) + 40 val (20 orig + 20 mir, paired by source).
#   "pure" = 309 vis_base ORIGINALS + 291 left-right MIRRORS (hflip videos + state/action swap).
#   ZERO kai0_base/kai0_dagger source — only sim01 visrobot01 cam domain.
# 40k steps, peak_lr=1.5e-5 cosine to 1.5e-6, warmup=1k, ema=0.9999.
# Save every 2k step (20 ckpts × ~12 GB = 240 GB).
# ETA ~21 hr (independent gf1, no I/O competition since visrobot01_only finished).
set -euo pipefail

export PATH=/home/tim/miniconda3/bin:/home/tim/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export KAI0_DATA_ROOT=/vePFS/tim/workspace/deepdive_kai0/kai0
export OPENPI_DATA_HOME=/vePFS/tim/workspace/openpi_cache
export PYTORCH_CKPT_BASE=/vePFS/tim/workspace/openpi_cache/modelscope_cache/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_DATASETS_CACHE=/home/tim/.cache/huggingface/datasets
export WANDB_MODE=offline
export LD_LIBRARY_PATH=/home/tim/miniconda3/lib:/home/tim/.cuda_compat:/usr/local/cuda-12.8/targets/x86_64-linux/lib
for d in /home/tim/.kai0_venv/lib/python3.11/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done

cd /vePFS/tim/workspace/deepdive_kai0/kai0

echo "[train] === START $(date) ==="
# --resume: safe whether ckpts exist or not (first run auto-falls back to weight_loader).
# NEVER --overwrite (it rmtrees the entire exp dir).
.venv/bin/python scripts/train.py pi05_flatten_fold_pure_vis600 \
  --exp_name=pure_vis600_v1 \
  --resume 2>&1
echo "[train] === END $(date) ==="
