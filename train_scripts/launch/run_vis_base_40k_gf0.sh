#!/bin/bash
# Task_A vis_base_40k cold-start training on gf0.
# Pair with mix_vis600 (gf0) + pure_vis600 (gf1) — same hyperparams, vis_base ONLY data.
#
# Dataset: 288 train + 22 val from vis_base 310 ep (no kai0, no hflip mirror).
#   This is the SIMPLEST baseline of the 4-experiment series:
#   - mix_vis600 (vis 310 + kai0 290): MAE@1=0.0146 ✅
#   - pure_vis600 (vis 309 + hflip 291): in progress
#   - vis_base_40k (vis 310 only): THIS RUN
#   - mixed_gf0_173 (vis 173 + base 173 + dagger 173, 13k step): MAE@1=0.0129 ✅
#
# 40k steps, peak_lr=1.5e-5 cosine to 1.5e-6, warmup=1k, ema=0.9999.
# Save every 2k step. Same config as pure_vis600 except dataset.
# ETA ~21 hr (gf0 alone, no I/O competition since pure_vis600 on gf1 separate node).
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
.venv/bin/python scripts/train.py pi05_flatten_fold_vis_base_40k \
  --exp_name=vis_base_40k_v1 \
  --resume 2>&1
echo "[train] === END $(date) ==="
