#!/bin/bash
# CAN 接口诊断脚本 — 排查无数据的 master 臂
#
# 用法: bash can_tools/diagnose_can.sh

set -eo pipefail

echo "============================================"
echo "  CAN 诊断"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

# ── 1. USB 设备枚举 ──────────────────────────────────────────────────────────
echo "--- 1. USB 设备枚举 ---"
lsusb | grep -i -E "can|canable|gs_usb|1d50|0483" || echo "  未找到已知 CAN 适配器 VID"
echo ""
echo "所有 USB 设备:"
lsusb
echo ""

# ── 2. 内核驱动 ──────────────────────────────────────────────────────────────
echo "--- 2. CAN 内核驱动 ---"
lsmod | grep -E "gs_usb|can" || echo "  无 CAN 模块"
echo ""
dmesg | grep -i -E "can|gs_usb" | tail -20
echo ""

# ── 3. CAN 接口详情 ─────────────────────────────────────────────────────────
echo "--- 3. CAN 接口详情 ---"
for iface in $(ip -br link show type can | awk '{print $1}'); do
    echo ""
    echo "=== $iface ==="

    # 状态
    ip -br link show "$iface"

    # bus-info
    bus=$(sudo ethtool -i "$iface" 2>/dev/null | grep "bus-info" | sed 's/.*bus-info: *//')
    echo "  bus-info: $bus"

    # 详细 CAN 状态
    ip -d link show "$iface" | grep -E "can state|bitrate|restart"

    # 统计
    ip -s link show "$iface" | grep -A1 "RX:\|TX:"

    # 错误计数
    ip -d -s link show "$iface" | grep -E "re-started|bus-errors|arbit-lost|error-warn|error-pass|bus-off"

    # RX 测试
    result=$(timeout 1 candump "$iface" -n 1 2>&1)
    if [ -n "$result" ]; then
        echo "  RX: 有数据 ✓"
        echo "  示例帧: $result"
    else
        echo "  RX: 无数据 ✗"
    fi

    # TX 测试
    cansend "$iface" 000#0000000000000000 2>&1 && echo "  TX: OK ✓" || echo "  TX: FAIL ✗"
done
echo ""

# ── 4. 对比分析 ──────────────────────────────────────────────────────────────
echo "--- 4. 对比分析 ---"
echo ""
printf "  %-20s %-15s %-8s %-8s %-10s\n" "接口" "Bus-Info" "RX" "TX" "状态"
printf "  %-20s %-15s %-8s %-8s %-10s\n" "----" "--------" "--" "--" "----"

for iface in $(ip -br link show type can | awk '{print $1}'); do
    bus=$(sudo ethtool -i "$iface" 2>/dev/null | grep "bus-info" | sed 's/.*bus-info: *//')

    rx_result=$(timeout 1 candump "$iface" -n 1 2>&1)
    rx_ok="无数据 ✗"
    [ -n "$rx_result" ] && rx_ok="有数据 ✓"

    tx_ok="FAIL ✗"
    cansend "$iface" 000#0000000000000000 2>/dev/null && tx_ok="OK ✓"

    state=$(ip -br link show "$iface" | awk '{print $2}')

    printf "  %-20s %-15s %-8s %-8s %-10s\n" "$iface" "$bus" "$rx_ok" "$tx_ok" "$state"
done
echo ""

# ── 5. dmesg CAN 错误 ───────────────────────────────────────────────────────
echo "--- 5. 最近 CAN 相关 dmesg ---"
dmesg | grep -i -E "can|gs_usb|error|fault" | tail -30
echo ""

# ── 6. 建议 ──────────────────────────────────────────────────────────────────
echo "--- 6. 排查建议 ---"
echo ""

has_no_rx=false
for iface in $(ip -br link show type can | awk '{print $1}'); do
    rx_result=$(timeout 1 candump "$iface" -n 1 2>&1)
    if [ -z "$rx_result" ]; then
        has_no_rx=true
        bus=$(sudo ethtool -i "$iface" 2>/dev/null | grep "bus-info" | sed 's/.*bus-info: *//')
        echo "  ✗ $iface ($bus) 无 RX 数据"
        echo "    排查步骤:"
        echo "    1. 交叉测试: 把有数据的臂的 CAN 线插到此适配器, 确认适配器本身正常"
        echo "    2. 反向交叉: 把此适配器的 CAN 线插到有数据的臂, 确认臂是否有 CAN 输出"
        echo "    3. 检查臂的 CAN 终端电阻跳线 (120Ω)"
        echo "    4. 用万用表测量 CAN_H/CAN_L 之间电压 (正常应有 2-3V 差分信号)"
        echo "    5. 检查臂是否有独立 CAN 使能开关/拨码"
        echo ""
    fi
done

if ! $has_no_rx; then
    echo "  所有接口 RX 正常 ✓"
fi
