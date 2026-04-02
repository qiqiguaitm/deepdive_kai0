#!/bin/bash
# 启动 Policy Server，启用 JAX 编译缓存
unset XLA_FLAGS
export JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache
export CUDA_VISIBLE_DEVICES=0
cd /data1/tim/workspace/deepdive_kai0/kai0
exec .venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config=pi05_flatten_fold_normal \
  --policy.dir=checkpoints/Task_A/mixed_1
