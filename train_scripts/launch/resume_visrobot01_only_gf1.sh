#!/bin/bash
# Resume Task_A visrobot01-only from step 9000 ckpt with vis_base 288-ep dataset.
# Dataset rebuilt from /home/tim/workspace/deepdive_kai0/kai0/data/Task_A/vis_base
# (288 train / 22 val, source 2026-04-{23,24,25}). norm_stats kept = old (model-consistent).
# Continues 12k cosine schedule -> 3000 more steps. LR ~3.66e-6 -> 1.5e-6.
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

echo "[train] === RESUME START $(date) ==="
.venv/bin/python scripts/train.py pi05_flatten_fold_visrobot01_only \
  --exp_name=visrobot01_only_v1 \
  --resume 2>&1
echo "[train] === RESUME END $(date) ==="
