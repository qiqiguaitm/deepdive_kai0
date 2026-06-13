#!/usr/bin/env bash
# ms_download.sh <dataset_id> <dest_subdir> [include_pattern]
# ModelScope dataset downloader with retries (proxy bypassed, per download_cosmos3_models.sh).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy='*' NO_PROXY='*' MODELSCOPE_DOMAIN=www.modelscope.cn
MS=/mnt/pfs/p46h4f/cosmos/.ms-tool/bin/modelscope
ROOT=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/external_cloth
ID="$1"; DEST="$ROOT/$2"; INC="${3:-}"
mkdir -p "$DEST"
for attempt in 1 2 3 4 5 6 7 8; do
  if [ -n "$INC" ]; then
    "$MS" download --dataset "$ID" --include "$INC" --local_dir "$DEST" && { echo "DONE $ID"; exit 0; }
  else
    "$MS" download --dataset "$ID" --local_dir "$DEST" && { echo "DONE $ID"; exit 0; }
  fi
  echo "---- attempt $attempt failed for $ID, retry in 30s ----"; sleep 30
done
echo "FAILED $ID after 8 attempts"; exit 1
