#!/bin/bash
# 扫描所有 CAN 接口, 以随机顺序重命名为 can0, can1, ... 并激活。
#
# 用途: 快速激活所有 CAN 接口, 不关心 bus-info 到符号名的映射。
#       适合调试/测试场景, 只需确保接口 UP 即可。
#
# 用法:
#   bash can_tools/find_and_random_activate.sh
#   bash can_tools/find_and_random_activate.sh --bitrate 500000   # 自定义波特率 (默认 1000000)

set -eo pipefail

BITRATE=1000000

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bitrate) BITRATE="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ $(id -u) -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
    sudo -v || { echo "[FAIL] 无法获取 sudo 权限"; exit 1; }
fi

# ── 加载 gs_usb 驱动 ───────────────────────────────────────────────────────
$SUDO modprobe gs_usb 2>/dev/null || echo "[WARN] 无法加载 gs_usb 模块 (可能已加载或不适用)"

# ── Step 1: 扫描 ────────────────────────────────────────────────────────────
echo "=== Step 1: 扫描 CAN 接口 ==="

mapfile -t IFACES < <(ip -br link show type can 2>/dev/null | awk '{print $1}')

if [[ ${#IFACES[@]} -eq 0 ]]; then
    echo "[FAIL] 未检测到任何 CAN 接口"
    exit 1
fi

echo "检测到 ${#IFACES[@]} 个 CAN 接口:"
for iface in "${IFACES[@]}"; do
    bus=$($SUDO ethtool -i "$iface" 2>/dev/null | grep "bus-info" | sed 's/.*bus-info: *//' || echo "N/A")
    state=$(ip -br link show "$iface" | awk '{print $2}')
    echo "  $iface  bus=$bus  state=$state"
done
echo ""

# ── Step 2: 全部 down ───────────────────────────────────────────────────────
echo "=== Step 2: 关闭所有接口 ==="
for iface in "${IFACES[@]}"; do
    $SUDO ip link set "$iface" down 2>/dev/null || true
done
echo "  已全部 down"
echo ""

# ── Step 3: 打乱顺序, 重命名为 can0, can1, ... ─────────────────────────────
echo "=== Step 3: 随机分配 canX 名称 ==="

# 收集当前接口名 (down 之后重新读取, 名称可能已变)
mapfile -t IFACES < <(ip -br link show type can 2>/dev/null | awk '{print $1}')

# 打乱顺序
mapfile -t SHUFFLED < <(printf '%s\n' "${IFACES[@]}" | shuf)

# 先全部改为临时名, 避免 canX 之间互相冲突
TMP_IDX=0
declare -a TMP_NAMES=()
for iface in "${SHUFFLED[@]}"; do
    tmp="can_rnd_${TMP_IDX}"
    if [[ "$iface" != "$tmp" ]]; then
        $SUDO ip link set "$iface" name "$tmp"
    fi
    TMP_NAMES+=("$tmp")
    TMP_IDX=$((TMP_IDX + 1))
done

# 临时名 → canX 并激活
IDX=0
for tmp in "${TMP_NAMES[@]}"; do
    target="can${IDX}"
    if [[ "$tmp" != "$target" ]]; then
        $SUDO ip link set "$tmp" name "$target"
    fi
    $SUDO ip link set "$target" type can bitrate "$BITRATE"
    $SUDO ip link set "$target" up

    bus=$($SUDO ethtool -i "$target" 2>/dev/null | grep "bus-info" | sed 's/.*bus-info: *//' || echo "N/A")
    echo "  $target  (bus=$bus)  [UP, bitrate=$BITRATE]"
    IDX=$((IDX + 1))
done
echo ""

# ── Step 4: 验证 ────────────────────────────────────────────────────────────
echo "=== Step 4: 接口状态校验 ==="

for iface in $(ip -br link show type can | sort | awk '{print $1}'); do
    # 检查接口是否 UP 且 bitrate 正确 (比 candump 更可靠, 不依赖臂是否在动)
    link_state=$(ip -br link show "$iface" | awk '{print $2}')
    actual_bitrate=$(ip -details link show "$iface" | grep -oP 'bitrate \K\d+' || echo "0")
    bus=$($SUDO ethtool -i "$iface" 2>/dev/null | grep "bus-info" | sed 's/.*bus-info: *//' || echo "N/A")

    if [[ "$link_state" == "UP" ]] && [[ "$actual_bitrate" -eq "$BITRATE" ]]; then
        echo "  $iface: OK  (bus=$bus, bitrate=$actual_bitrate, state=UP)"
    else
        echo "  $iface: FAIL (bus=$bus, bitrate=$actual_bitrate, state=$link_state)"
    fi
done

echo ""
echo "=== 完成: ${#SHUFFLED[@]} 个 CAN 接口已激活 ==="
echo ""
echo "提示: 如需验证数据流, 可手动运行: candump canX"
