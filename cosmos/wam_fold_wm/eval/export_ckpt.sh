#!/usr/bin/env bash
# Export one DCP checkpoint to HF safetensors format.
# Usage:  bash export_ckpt.sh <iter_num>   e.g.  bash export_ckpt.sh 500
# Output: $RUNS/exported/wam_fold_wm_iter<N>/config.json + model.safetensors
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'

ITER=${1:?Usage: bash export_ckpt.sh <iter_num>}
ITER9=$(printf "%09d" "$ITER")   # iter_000000500

CF=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3
VENV=$CF/.venv; PY=$VENV/bin/python
RUNS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs
CKPT_BASE=$RUNS/train_out_5n8g/cosmos3/action/wam_fold_wm_nano
CK=$CKPT_BASE/checkpoints/iter_$ITER9
CFG=$CKPT_BASE/config.yaml
EXP=$RUNS/exported/wam_fold_wm_iter$ITER

export PYTHONPATH="$CF:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib"
export PATH=/mnt/pfs/p46h4f/cosmos/uvbin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WAN_VAE_PATH=/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
export CUDA_VISIBLE_DEVICES=0

if [ ! -d "$CK" ]; then
  echo "[export_ckpt] ERROR: DCP dir not found: $CK" >&2; exit 1
fi
if [ -f "$EXP/config.json" ]; then
  echo "[export_ckpt] Already exported: $EXP (skip)"; exit 0
fi
mkdir -p "$EXP"
echo "[export_ckpt] iter=$ITER  dcp=$CK  out=$EXP  $(date +%T)"
cd "$CF"
"$PY" -m cosmos_framework.scripts.export_model \
  --checkpoint-path "$CK" --config-file "$CFG" --no-vit -o "$EXP"
RC=$?
if [ $RC -eq 0 ] && [ -f "$EXP/config.json" ]; then
  echo "[export_ckpt] EXPORT_OK  iter=$ITER  $(date +%T)"
else
  echo "[export_ckpt] EXPORT_FAILED rc=$RC  iter=$ITER" >&2
  exit 1
fi
