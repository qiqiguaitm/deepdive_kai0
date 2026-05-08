#!/bin/bash
# Pull a packed ckpt tarball from uc01/uc02/uc03 to sim01.
#
# Benchmark notes (2026-05-08, 12 GB tar):
#   - TOS path:      8:00 (uc upload 33 MB/s + sim01 download 97 MB/s, serial)
#   - single scp:    6:13 (33.4 MB/s, single TCP stream)
#   - 8x parallel:   5:51 (35.4 MB/s, only +6% — ISP cap ~280 Mbps)
#
# Conclusion: ISP egress is the bottleneck; parallelism doesn't help.
# Use single scp PULL for simplicity unless transferring multiple files.
#
# Usage (run on sim01):
#   ./transfer_ckpt_parallel.sh <uc_host> <uc_tar_path> <sim01_dst_dir>
# Example:
#   ./transfer_ckpt_parallel.sh tim@106.75.68.254 \
#       /tmp/ckpt_pack/task_p_v2_aligned_step19999.tar /tmp/ckpt_pack/

set -euo pipefail

UC_HOST="${1:?Usage: $0 <uc_user@host> <uc_tar_path> <sim01_dst_dir>}"
UC_TAR="${2:?need uc tar path}"
DST_DIR="${3:?need sim01 dst dir}"

mkdir -p "$DST_DIR"

echo "[transfer] sim01 PULL: $UC_HOST:$UC_TAR → $DST_DIR/"
time scp "$UC_HOST:$UC_TAR" "$DST_DIR/"
echo ""
echo "[transfer] DONE: $DST_DIR/$(basename "$UC_TAR")"
ls -lh "$DST_DIR/$(basename "$UC_TAR")"
