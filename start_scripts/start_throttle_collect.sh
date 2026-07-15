#!/bin/bash
###############################################################################
# kai0 油门加减速数据采集 (throttle-only autonomous collection) — thin wrapper
#
# 场景: 一只主臂返厂维修 → 无法做需要双主臂的 dagger 人工接管。此模式让策略
# 【自主】跑从臂 (两只从臂在线即可), 操作员用 USB 脚踏板控速:
#   - 踩住脚踏板  → 机械臂提速 (throttle_factor, 默认 1.5x), 数据落 inference_fast/
#   - 松开脚踏板  → 回默认速度 (1.0x), 数据落 inference/
# recorder 全程停在 POLICY_RUN (无 master 开关 → 无接管), 只录 inference/ +
# inference_fast/ 两个 subset。复用 dagger 的整套基础设施 + web/dagger_manager 前端。
#
# 与 start_dagger_collect.sh 的唯一区别: 设 KAI0_ENABLE_MASTER=0 (关掉 2× master_servo,
# 也避开缺失的 can_right_mas 让 servo 启动即崩), 其余参数全部透传。
#
# 用法:
#   ./start_scripts/start_throttle_collect.sh                 # 无 ckpt: web UI 里选并启动
#   ./start_scripts/start_throttle_collect.sh --task Task_A
#   ./start_scripts/start_throttle_collect.sh --ckpt <ckpt_dir> --task Task_A
#
# 采集流程:
#   1. 本脚本起基础设施 (CAN/相机/从臂/recorder/pedal) + web (端口 8788/5174)
#   2. 浏览器打开 web → 选 ckpt → Start session (策略自主跑)
#   3. POLICY_RUN 下踩/松脚踏板控速; Rollout「下一个」按钮标记一次任务完成+复位场景
#   4. 数据: <task>/inference/<vN>/  (常速) 与  <task>/inference_fast/<vN>/  (油门)
###############################################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 关掉 master_servo (无人接管 / 主臂返厂)。export 使 web 后端若二次 fork
# start_dagger_collect.sh (一键 system/start) 也继承此设置。
export KAI0_ENABLE_MASTER=0

# 默认 Task_A: 未显式传 --task/--task-name 时注入 Task_A (recorder 据此选数据集目录)。
ARGS=("$@")
_has_task=0
for a in "$@"; do
    [[ "$a" == "--task" || "$a" == "--task-name" ]] && _has_task=1 && break
done
if [[ "$_has_task" == "0" ]]; then
    ARGS+=("--task" "Task_A")
    _task_note="Task_A (默认)"
else
    _task_note="<CLI 指定>"
fi

echo "============================================================"
echo " kai0 油门加减速采集 (throttle-only, master_servo OFF)"
echo " 策略自主跑从臂 + 脚踏板控速; 只录 inference/ + inference_fast/"
echo " task : $_task_note"
echo " 委托 → start_dagger_collect.sh (KAI0_ENABLE_MASTER=0)"
echo "============================================================"

exec "$SCRIPT_DIR/start_dagger_collect.sh" "${ARGS[@]}"
