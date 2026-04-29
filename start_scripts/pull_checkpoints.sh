#!/usr/bin/env bash
###############################################################################
# pull_checkpoints.sh — gf → sim01 单向 checkpoint 同步
#
# 把 gf:/vePFS/visrobot01/checkpoints/ rsync 到 sim01:/data1/DATA_IMP/checkpoints/
# (即 kai0/checkpoints/, 因为后者是 → /data1/DATA_IMP/checkpoints 的 symlink).
#
# 设计:
#   - 单向 pull (gf 是 ckpt 训练侧, sim01 拉来做推理).
#   - rsync -a --partial --update: 不删 sim01 端独有文件; sim01 端 newer 不被覆盖.
#   - flock 防并发, 适合 cron 周期跑.
#   - 默认 bwlimit 5 MB/s (gf WAN 链路上限 ~13 MB/s, 留余量给录制 sync).
#   - 默认排除 train_state/ (巨大但纯训练用; 可 KAI0_PULL_KEEP_TRAIN_STATE=1 留下).
#
# 用法:
#   ./pull_checkpoints.sh                    # 真 pull (跟 cron 用)
#   ./pull_checkpoints.sh --dry-run          # 看会传什么, 不动磁盘
#   ./pull_checkpoints.sh --bwlimit 10000    # 限速 10 MB/s (KB)
#   ./pull_checkpoints.sh --no-bwlimit       # 不限速
#   KAI0_PULL_KEEP_TRAIN_STATE=1 ./pull_checkpoints.sh   # 也拉 train_state/ (续训用)
#
# Cron 示例 (每小时 :15 跑一次, 日志 append):
#   15 * * * * /home/tim/workspace/deepdive_kai0/start_scripts/pull_checkpoints.sh \
#       >> /home/tim/workspace/deepdive_kai0/logs/pull_checkpoints.log 2>&1
#
# 排错:
#   - 远端 rsync 不在 PATH → ssh tim@<gf>; sudo apt install rsync
#   - "another pull is running" → 查 /tmp/kai0_pull_checkpoints.lock 持有者; 停旧的再跑
#   - rc=23 (partial transfer) → 通常 1-2 个文件权限/transient I/O 错, 下次 cron 自动续
###############################################################################

set -eo pipefail

SRC_HOST="14.103.44.161"
SRC_PORT="11111"
SRC_USER="tim"
SRC_PATH="/vePFS/visrobot01/checkpoints/"
DST_PATH="/data1/DATA_IMP/checkpoints/"
LOCK_FILE="/tmp/kai0_pull_checkpoints.lock"

# parse args
DRY_RUN=""
BWLIMIT="${KAI0_PULL_BWLIMIT:-5000}"   # default 5 MB/s
KEEP_TRAIN_STATE="${KAI0_PULL_KEEP_TRAIN_STATE:-0}"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)       DRY_RUN="--dry-run"; shift ;;
        --bwlimit)       BWLIMIT="$2"; shift 2 ;;
        --no-bwlimit)    BWLIMIT="0"; shift ;;
        --keep-train-state) KEEP_TRAIN_STATE="1"; shift ;;
        -h|--help)       sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# ── concurrent-run guard ──────────────────────────────────────────────────
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    log "another pull is running (lock=$LOCK_FILE); skip this tick"
    exit 0
fi

# ── build rsync args ──────────────────────────────────────────────────────
log "=== pull start (bwlimit=${BWLIMIT}KB/s, dry_run=${DRY_RUN:-no}, keep_train_state=$KEEP_TRAIN_STATE) ==="

RSYNC_ARGS=(-a --partial --update --info=stats2,name1)

# Limit bandwidth so we don't starve the recording sync (which also goes WAN-bound)
if [ "$BWLIMIT" -gt 0 ]; then
    RSYNC_ARGS+=("--bwlimit=$BWLIMIT")
fi

# Exclude train_state/ unless asked to keep — train_state is HUGE (gradient + optimizer),
# inference doesn't need it. Save ~30-40% bandwidth per ckpt.
if [ "$KEEP_TRAIN_STATE" != "1" ]; then
    RSYNC_ARGS+=("--exclude=train_state/")
fi

# Skip in-progress / atomic-write artifacts
RSYNC_ARGS+=("--exclude=*.tmp" "--exclude=*.lock" "--exclude=.commit_success.json")

# dry-run last so it still gates anything else
[ -n "$DRY_RUN" ] && RSYNC_ARGS+=("$DRY_RUN")

# ssh transport: BatchMode (key-only, no password prompt) + connect timeout
RSYNC_ARGS+=("-e" "ssh -p $SRC_PORT -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30")

# src + dst (trailing slash on src copies contents, not the dir itself)
RSYNC_ARGS+=("${SRC_USER}@${SRC_HOST}:${SRC_PATH}" "$DST_PATH")

# ── run ──────────────────────────────────────────────────────────────────
START=$(date +%s)
if rsync "${RSYNC_ARGS[@]}"; then
    DT=$(( $(date +%s) - START ))
    log "=== pull ok in ${DT}s ==="
    exit 0
else
    rc=$?
    DT=$(( $(date +%s) - START ))
    log "=== pull FAILED rc=$rc after ${DT}s ==="
    # rc=23 (partial transfer) and rc=24 (vanished files) are mostly OK with --partial,
    # treat them as soft failures so cron can retry without paging.
    if [ "$rc" = "23" ] || [ "$rc" = "24" ]; then
        log "    (rc=$rc is partial / vanished — soft fail, will retry next tick)"
        exit 0
    fi
    exit "$rc"
fi
