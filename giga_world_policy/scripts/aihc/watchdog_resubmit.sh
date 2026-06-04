#!/usr/bin/env bash
# 应用层"容错看护":轮询 latent 训练 job,检测到 Failed/Stopped 就自动 resubmit(resume=True 从最新
# ckpt 续),直到训练日志到达 MAXSTEP。用于绕过 aihc 原生 auto-retry 不生效的问题(spec 的
# unconditionalFaultToleranceLimit 被忽略)。配 checkpoint_interval=1000 → 每次崩溃损失 <1k step。
# 密码从环境变量 AIHC_IMG_PASSWORD 取(绝不入库)。
# 用法:
#   AIHC_IMG_PASSWORD='****' JOB=job-xxx WD=runs/..._5x \
#     CONFIG_NAME=world_action_model.configs.visrobot01_fold_aihc_latent_5x.config \
#     nohup bash scripts/aihc/watchdog_resubmit.sh > .wam_run/watchdog_5x.log 2>&1 &
set -uo pipefail
REPO=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy; cd "$REPO"; source env.sh >/dev/null 2>&1 || true
POOL=aihc-serverless; Q=aihcq-z4v1apdppzwy
JOB=${JOB:?set JOB}; WD=${WD:?set WD}; CONFIG_NAME=${CONFIG_NAME:?set CONFIG_NAME}
MAXSTEP=${MAXSTEP:-50000}; POLL=${POLL:-300}; MAX_RESUB=${MAX_RESUB:-10}
: "${AIHC_IMG_PASSWORD:?set AIHC_IMG_PASSWORD}"
resub=0
echo "[watchdog] start JOB=$JOB WD=$WD MAXSTEP=$MAXSTEP POLL=${POLL}s"
while true; do
  sleep "$POLL"
  LOG=$(ls -t "$WD"/logs/train_*.log 2>/dev/null | head -1)
  step=$(grep -aoE 'Step\[[0-9]+/' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
  st=$(echo q | aihc job get "$JOB" -p "$POOL" -q "$Q" 2>&1 | grep -oE '^    status: [A-Za-z]+' | head -1 | awk '{print $2}')
  echo "[$(date '+%m-%d %H:%M')] job=$JOB status=${st:-?} step=${step:-?} resub=$resub"
  if [ -n "${step:-}" ] && [ "$step" -ge "$MAXSTEP" ]; then echo "[watchdog] DONE: reached step $step"; break; fi
  case "${st:-}" in
    Failed|Stopped|ManualTermination)
      if [ "$resub" -ge "$MAX_RESUB" ]; then echo "[watchdog] hit MAX_RESUB=$MAX_RESUB, stop"; break; fi
      echo "[watchdog] $JOB is $st -> auto-resubmit (resume from latest ckpt)"
      NEW=$(CONFIG_NAME="$CONFIG_NAME" bash scripts/aihc/resubmit_latent.sh 2>&1 | grep -oE 'job-[a-z0-9]+' | tail -1)
      if [ -n "$NEW" ]; then JOB="$NEW"; resub=$((resub+1)); echo "[watchdog] resubmitted -> $JOB (#$resub)"; sleep 120
      else echo "[watchdog] resubmit FAILED, retry next poll"; fi
      ;;
  esac
done
