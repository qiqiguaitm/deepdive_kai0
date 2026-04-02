#!/usr/bin/env python3
"""
CAN-臂映射校验脚本

用法: python3 can_tools/verify_can_mapping.py

原理: 持续监控所有活跃 CAN 接口的关节数据变化量。
      手动晃动某个臂的关节, 脚本会实时显示哪个 CAN 接口检测到运动。

操作步骤:
  1. 先运行 activate_can.sh 激活并重命名 CAN 接口
  2. 运行本脚本
  3. 依次晃动每个臂, 观察输出中哪个接口标记了 "<<< MOVING"
  4. 核对是否与预期映射一致
  5. Ctrl+C 退出
"""

import re
import subprocess
import sys
import time

from piper_sdk import C_PiperInterface

MOVE_THRESHOLD = 50  # 原始值变化超过此阈值视为运动


def get_can_interfaces():
    """自动检测所有 UP 状态的 CAN 接口"""
    out = subprocess.run(
        ["ip", "-br", "link", "show", "type", "can"],
        capture_output=True, text=True, timeout=5,
    )
    interfaces = []
    for line in out.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        name = parts[0]
        state = parts[1] if len(parts) > 1 else ""
        # UP 或 LOWER_UP 均视为可用
        if "UP" in state or "UP" in line:
            interfaces.append(name)
    return sorted(interfaces)


def read_joints(piper):
    raw = str(piper.GetArmJointMsgs())
    return [int(v) for _, v in re.findall(r"Joint\s+(\d+):(-?\d+)", raw)]


def main():
    interfaces = get_can_interfaces()
    if not interfaces:
        print("未检测到活跃的 CAN 接口, 请先运行: bash can_tools/activate_can.sh")
        sys.exit(1)

    print(f"检测到 {len(interfaces)} 个 CAN 接口: {', '.join(interfaces)}")
    print("正在连接...")

    pipers = {}
    for iface in interfaces:
        try:
            p = C_PiperInterface(can_name=iface)
            p.ConnectPort()
            pipers[iface] = p
        except Exception as e:
            print(f"  {iface}: 连接失败 — {e}")

    if not pipers:
        print("所有接口连接失败")
        sys.exit(1)

    time.sleep(2)

    # 初始读数
    prev = {}
    for iface in pipers:
        prev[iface] = read_joints(pipers[iface])

    print()
    print("开始监控, 请依次晃动每个臂的关节 (Ctrl+C 退出)")
    print("=" * 60)

    # 先打印占位行
    for iface in pipers:
        print(f"  {iface:20s}: max_delta=     0")

    try:
        while True:
            time.sleep(0.3)
            lines = []
            for iface in pipers:
                cur = read_joints(pipers[iface])
                if prev[iface] and cur:
                    diffs = [abs(a - b) for a, b in zip(cur, prev[iface])]
                    max_diff = max(diffs)
                    marker = " <<< MOVING" if max_diff > MOVE_THRESHOLD else ""
                    lines.append(f"  {iface:20s}: max_delta={max_diff:6d}{marker}")
                else:
                    lines.append(f"  {iface:20s}: 读取失败")
                prev[iface] = cur

            # 覆盖上一轮输出
            sys.stdout.write(f"\033[{len(lines)}A")
            for line in lines:
                sys.stdout.write(f"\033[2K{line}\n")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
