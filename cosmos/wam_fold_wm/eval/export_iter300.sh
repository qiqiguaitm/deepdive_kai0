#!/usr/bin/env bash
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
CF=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3
VENV=$CF/.venv; PY=$VENV/bin/python
RUNS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs
CK=$RUNS/smoke_out/cosmos3/action/wam_fold_wm_nano/checkpoints/iter_000000300
CFG=$RUNS/smoke_out/cosmos3/action/wam_fold_wm_nano/config.yaml
EXP=$RUNS/exported/wam_fold_wm_iter300
export PYTHONPATH="$CF:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib"
export PATH=/mnt/pfs/p46h4f/cosmos/uvbin:$PATH
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home HF_HUB_OFFLINE=1
export WAN_VAE_PATH=/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$EXP"
cd "$CF"
echo "=== export iter300 $(date +%T) ==="
"$PY" -m cosmos_framework.scripts.export_model \
  --checkpoint-path "$CK" --config-file "$CFG" --no-vit -o "$EXP"
echo "=== export rc=$? $(date +%T) ==="
[ -f "$EXP/config.json" ] && echo "EXPORT_OK $EXP" || echo "EXPORT_FAILED"
