#!/usr/bin/env bash
# 把 DATA_ROOT 下所有 AV1 mp4 就地重编码为 H.264 (yuv420p, +faststart)，
# 同步把各 subset 的 meta/info.json 中 video.codec 从 "av1" 改为 "h264"。
# 原始 AV1 备份到 <mp4>.av1.bak（如不需要，运行后手动清理）。
#
# 用法:  bash backend/tools/transcode_av1_to_h264.sh [DATA_ROOT]
#
set -euo pipefail

DATA_ROOT="${1:-${KAI0_DATA_ROOT:-/data1/DATA_IMP/KAI0}}"
echo "[transcode] DATA_ROOT = $DATA_ROOT"
command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe not found"; exit 1; }

converted=0; skipped=0
while IFS= read -r -d '' f; do
    codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name \
        -of default=nk=1:nw=1 "$f" 2>/dev/null || echo "")
    if [[ "$codec" != "av1" ]]; then
        skipped=$((skipped+1)); continue
    fi
    echo "  -> $f"
    tmp="${f%.mp4}.h264.tmp.mp4"
    if ffmpeg -v error -y -i "$f" \
        -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
        -movflags +faststart "$tmp"; then
        mv "$f" "$f.av1.bak"
        mv "$tmp" "$f"
        converted=$((converted+1))
    else
        rm -f "$tmp"
        echo "     [FAIL] keep original"
    fi
done < <(find "$DATA_ROOT" -type f -name "*.mp4" -print0)

# 更新 info.json
while IFS= read -r -d '' info; do
    python3 - "$info" <<'PY'
import json, sys
p = sys.argv[1]
with open(p) as f: d = json.load(f)
changed = False
for k, v in (d.get("features") or {}).items():
    if isinstance(v, dict) and v.get("dtype") == "video":
        for box in ("info", "video_info"):
            if box in v and v[box].get("video.codec") == "av1":
                v[box]["video.codec"] = "h264"; changed = True
if changed:
    with open(p, "w") as f: json.dump(d, f, indent=2, ensure_ascii=False)
    print(f"  updated {p}")
PY
done < <(find "$DATA_ROOT" -type f -path "*/meta/info.json" -print0)

echo "[transcode] converted=$converted skipped=$skipped"
echo "如果一切正常，清理备份: find '$DATA_ROOT' -name '*.mp4.av1.bak' -delete"
