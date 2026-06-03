#!/bin/bash
# 百度 AIHC 多节点 VAE latent 预抽:每 pod 起 8 个分片进程(每 GPU 一个),全集群 NNODES*8 路并行。
# 每进程跑 compute_latents.py 的一个 episode 分片(取模),DataLoader 内再并行解码。
# 写 {emb}/vae_latent/episode_*.pt 到共享 PFS;各分片写不同 episode,无冲突;已存在则跳过(可断点续抽)。
# 提交:aihc job create -f scripts/aihc/aijob_precompute_4n8g.json(command 指向本脚本)。
set -e
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy
source "$REPO/env.sh"; cd "$REPO"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-${WORLD_SIZE:-4}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}
NUM_SHARDS=$((NNODES * NUM_GPUS))
STRIDE=${STRIDE:-4}
WORKERS=${WORKERS:-10}
echo "[precompute] node $NODE_RANK/$NNODES  gpus=$NUM_GPUS  num_shards=$NUM_SHARDS  stride=$STRIDE"

for EMB in visrobot01_train kairobot01; do
  echo "[precompute] === $EMB ==="
  pids=()
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD=$((NODE_RANK * NUM_GPUS + g))
    CUDA_VISIBLE_DEVICES=$g python -m scripts.wam_pipeline.compute_latents \
      --emb "$EMB" --stride "$STRIDE" --shard "$SHARD" --num-shards "$NUM_SHARDS" \
      --workers "$WORKERS" --batch 8 > "$REPO/.wam_run/precompute_${EMB}_shard${SHARD}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "[precompute] $EMB done on node $NODE_RANK"
done
echo "PRECOMPUTE_NODE_DONE $NODE_RANK"
