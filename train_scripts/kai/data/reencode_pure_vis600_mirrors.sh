#!/bin/bash
# Re-encode pure_vis600 mirror mp4 files for fast random-seek decode.
# Source: 873 real (non-symlink) h264 files in pure_vis600_flat_backup/videos/.
# Strategy: -preset ultrafast -bf 0 -x264opts keyint=15:min-keyint=15:scenecut=0
#   - 0 B-frames (no inter-frame dependency forward/backward)
#   - keyframe every 15 frames (max seek-back distance bounded)
# Result: 3.16 ms/seek -> 0.93 ms/seek (3.4x speedup, matches orig vis_base files)
# Cost: file size ~2.5x larger but only +1.7 GB total

set -euo pipefail

BACKUP=/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/pure_vis600_flat_backup
LOG=/tmp/reencode_mirrors.log
WORKERS=16
FFMPEG=/home/tim/miniconda3/bin/ffmpeg

reencode_one() {
    local f="$1"
    local tmp="${f}.tmp.mp4"
    if "$FFMPEG" -nostdin -y -i "$f" \
        -c:v libx264 -preset ultrafast -bf 0 \
        -x264opts keyint=15:min-keyint=15:scenecut=0 \
        -pix_fmt yuv420p -an \
        "$tmp" >/dev/null 2>&1; then
        mv -f "$tmp" "$f"
        echo "OK $f"
    else
        rm -f "$tmp"
        echo "FAIL $f"
    fi
}
export -f reencode_one
export FFMPEG

echo "[$(date)] start re-encode of mirrors in $BACKUP/videos" | tee -a $LOG
START=$(date +%s)

find "$BACKUP/videos" -name '*.mp4' -type f -print0 \
    | xargs -0 -n 1 -P $WORKERS -I {} bash -c 'reencode_one "$@"' _ {} \
    >> $LOG 2>&1

END=$(date +%s)
DUR=$((END - START))
TOTAL=$(grep -c '^OK ' $LOG || true)
FAIL=$(grep -c '^FAIL ' $LOG || true)
echo "[$(date)] done. ${TOTAL} OK, ${FAIL} FAIL, total ${DUR}s" | tee -a $LOG
