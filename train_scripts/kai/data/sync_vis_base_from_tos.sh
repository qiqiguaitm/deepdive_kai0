#!/usr/bin/env bash
# Hourly FULL incremental sync: TOS Task_A/base → local vis_base, via tosutil cp -r -u (逐版本逐日期).
#
# 策略:
#   - 完整: 每次遍历每个同步版本下所有 <date> (不只新日期) → 接住"旧日期后续追加 episode"。
#   - 增量: tosutil `cp -r -u` 按 size/crc 跳过未变文件, 只拉新/变更, 从不删除本地。
#   - 排除 depth zarr (top_head_depth): 训练只用 RGB 三路, depth 不被下游消费且对象数巨大。
#   - 默认不删除: cp 永不删本地多余文件 → 保护指向 vis_base 的软链 (如有)。
#
# 版本布局演进:
#   - 2026-06-19 TOS 重整: 日期目录从 base/<date> 移入版本层 base/{v2,v3}/<date> (+ analysis/, kai0_official_base/)。
#   - 2026-06-23 TOS 框架变更: 新增 base/v4/<date>-v4 = **新标准** (前裁+尾裁 + 夹爪取主臂 action≠state)。
#   - 本脚本默认同步 **v2(老原始) + v4(新标准)**; v3 是本地 front-trim+tail-cap 加工产物 / 已被 v4 取代
#     → **不从 TOS 拉**(拉了会覆盖本地裁剪成果)。VIS_BASE_SYNC_VERS 可覆盖版本列表。
#   - 路径映射: cp -r .../base/<ver>/<date>/ → vis_base/<ver>/ = vis_base/<ver>/<date>/.
#   - README.md + analysis/ 仍在 base/ 根层 → 同步到 vis_base/ 根。
#
# 运行环境 (host-aware, gf0 / gf3 / uc01-03): 按文件系统探测 KAI0 工作根 + tosutil 路径。
#   各机都需本机 ~/.tosutilconfig 凭据 (cn-shanghai AK/SK) + 可写 tosutil 二进制。
# 安装 cron (每机一次, 每小时):
#   crontab -l 2>/dev/null | grep -q sync_vis_base || \
#     (crontab -l 2>/dev/null; echo "17 * * * * bash <repo>/train_scripts/kai/data/sync_vis_base_from_tos.sh") | crontab -
#
# --mirror 模式 (手动跑, 不进 cron): cp -u 之后再删本地存在但 TOS 已无的日期目录 / orphan episode (逐版本)。
#   安全护栏: TOS 日期清单为空时跳过该版本, 绝不批量删。
#     bash sync_vis_base_from_tos.sh --mirror
set -uo pipefail
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do unset "$v"; done

MIRROR=0
[ "${1:-}" = "--mirror" ] && MIRROR=1

SRC=tos://transfer-shanghai/KAI0/Task_A/base       # README/analysis 仍在 base/ 层
SYNC_VERS="${VIS_BASE_SYNC_VERS:-v4}"            # 默认只同步 v4 新标准 (2026-06-23 TOS v2 已清空弃用; v3 本地加工); env 可覆盖
LOCK=/tmp/vis_base_sync.lock
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ---- host-aware: KAI0_ROOT + TOSUTIL 自动探测 ----
#   探测顺序: North-E / data-shared 优先于 /vePFS/tim (gf3 上 /vePFS/tim 是空壳树)。
_VB=kai0/data/Task_A/vis_base/v2
if   [ -d /vePFS-North-E/vis_robot/workspace/deepdive_kai0/$_VB ];  then
  KAI0_ROOT=/vePFS-North-E/vis_robot/workspace/deepdive_kai0
elif [ -d /data/shared/ubuntu/workspace/deepdive_kai0/$_VB ];      then
  KAI0_ROOT=/data/shared/ubuntu/workspace/deepdive_kai0
elif [ -d /vePFS/tim/workspace/deepdive_kai0/$_VB ];               then
  KAI0_ROOT=/vePFS/tim/workspace/deepdive_kai0
else
  echo "[$(ts)] ERROR: cannot locate deepdive_kai0/$_VB on this host" >&2; exit 1
fi
if   [ -x "$HOME/tosutil" ];        then TOSUTIL="$HOME/tosutil"
elif command -v tosutil >/dev/null; then TOSUTIL="$(command -v tosutil)"
else echo "[$(ts)] ERROR: tosutil not found (~/tosutil or PATH)" >&2; exit 1
fi
DST_ROOT="$KAI0_ROOT/kai0/data/Task_A/vis_base"
LOG="$KAI0_ROOT/logs/vis_base_sync.log"
mkdir -p "$KAI0_ROOT/logs" 2>/dev/null || true

# 防重叠
exec 9>"$LOCK"
flock -n 9 || { echo "[$(ts)] previous run still active, skip" >>"$LOG"; exit 0; }
# 日志轮转 (>5MB)
[ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 5242880 ] && mv "$LOG" "$LOG.old"
[ -x "$TOSUTIL" ] || { echo "[$(ts)] ERROR: tosutil missing at $TOSUTIL" >>"$LOG"; exit 1; }
[ -d "$DST_ROOT" ] || { echo "[$(ts)] ERROR: local DST_ROOT missing $DST_ROOT" >>"$LOG"; exit 1; }

# 护栏: 自愈 — 清理 vis_base 根下不该存在的扁平 <date> 残留 (date 数据应全在 vN/ 子目录, 见 2026-06-03 迁移事故)。
for fd in "$DST_ROOT"/2026-*-v*; do
  [ -d "$fd" ] || continue
  echo "[$(ts)] WARN 清理 vis_base 根扁平残留: $(basename "$fd") (应在 vN/ 下)" >>"$LOG"
  rm -rf "$fd"
done

EXCLUDE='*top_head_depth*'
echo "[$(ts)] sync start (versions: $SYNC_VERS; tosutil cp -r -u, full incremental)" >>"$LOG"

for ver in $SYNC_VERS; do
  SRC_VER="$SRC/$ver"
  DST="$DST_ROOT/$ver"
  mkdir -p "$DST"
  dates=$("$TOSUTIL" ls -d "$SRC_VER/" 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}-v[0-9]+/?$' | sed 's#/$##' | sort -u)
  if [ -z "$dates" ]; then echo "[$(ts)] WARN $ver: TOS 无日期 (该版本不存在/网络故障), 跳过" >>"$LOG"; continue; fi
  total=$(echo "$dates" | wc -w); ok=0; pulled=0
  for d in $dates; do
    out=$("$TOSUTIL" cp -r -u "$SRC_VER/$d/" "$DST/" -exclude="$EXCLUDE" -j 3 -p 8 2>&1); rc=$?
    succ=$(echo "$out" | grep -oE 'Succeed count is:[ ]*[0-9]+' | grep -oE '[0-9]+$' | tail -1)
    skip=$(echo "$out" | grep -oE 'Skip count is:[ ]*[0-9]+'    | grep -oE '[0-9]+$' | tail -1)
    fail=$(echo "$out" | grep -oE 'Failed count is:[ ]*[0-9]+'  | grep -oE '[0-9]+$' | tail -1)
    if [ "$rc" -eq 0 ]; then
      ok=$((ok+1))
      [ "${succ:-0}" -gt "${skip:-0}" ] 2>/dev/null && { pulled=$((pulled+1)); echo "[$(ts)] PULL $ver/$d succ=$succ skip=$skip" >>"$LOG"; }
    else
      echo "[$(ts)] FAIL $ver/$d rc=$rc succ=$succ skip=$skip fail=$fail" >>"$LOG"
      echo "$out" | tail -3 >>"$LOG"
    fi
  done
  echo "[$(ts)] $ver sync end: $ok/$total dates ok, $pulled with new/changed objects" >>"$LOG"

  # ---- --mirror: 传播 TOS 端删除, 仅本版本 ----
  if [ "$MIRROR" -eq 1 ]; then
    echo "[$(ts)] MIRROR $ver start (propagate TOS deletions)" >>"$LOG"
    tos_set=" $dates "
    for ld in "$DST"/*-v*/; do
      [ -d "$ld" ] || continue
      ldd=$(basename "$ld")
      case "$tos_set" in
        *" $ldd "*) : ;;
        *) echo "[$(ts)] MIRROR rm $ver date-dir not on TOS: $ldd ($(du -sh "$ld" 2>/dev/null|cut -f1))" >>"$LOG"; rm -rf "$ld" ;;
      esac
    done
    for d in $dates; do
      [ -d "$DST/$d/data/chunk-000" ] || continue
      tos_eps=$("$TOSUTIL" ls "$SRC_VER/$d/data/chunk-000/" 2>/dev/null | grep -oE 'episode_[0-9]+\.parquet' | sort -u)
      [ -n "$tos_eps" ] || { echo "[$(ts)] MIRROR skip $ver/$d (TOS list empty, guard)" >>"$LOG"; continue; }
      for lp in "$DST/$d/data/chunk-000/"episode_*.parquet; do
        [ -e "$lp" ] || continue
        ep=$(basename "$lp")
        if ! grep -qxF "$ep" <<<"$tos_eps"; then
          epn="${ep%.parquet}"
          rm -f "$lp" "$DST/$d/videos/chunk-000/"*/"$epn.mp4"
          echo "[$(ts)] MIRROR rm orphan $ver/$d/$ep (+videos)" >>"$LOG"
        fi
      done
    done
    echo "[$(ts)] MIRROR $ver end" >>"$LOG"
  fi
done

# ---- README + analysis (数据描述参考; 在 base/ 根层 → 同步到 vis_base/ 根) ----
"$TOSUTIL" cp "$SRC/README.md" "$DST_ROOT/README.md" -f >/dev/null 2>&1 \
  && echo "[$(ts)] README.md synced" >>"$LOG" \
  || echo "[$(ts)] WARN README.md sync failed (可能 TOS 无此文件)" >>"$LOG"
"$TOSUTIL" cp -r -u "$SRC/analysis/" "$DST_ROOT/" >/dev/null 2>&1 \
  && echo "[$(ts)] analysis/ synced" >>"$LOG" \
  || echo "[$(ts)] WARN analysis/ sync failed" >>"$LOG"
