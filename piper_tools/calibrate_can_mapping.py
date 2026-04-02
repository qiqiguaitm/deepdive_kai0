#!/usr/bin/env python3
"""
交互式 CAN-臂映射校准脚本

用法: python3 can_tools/calibrate_can_mapping.py

前置条件: 所有 CAN 接口已激活为 canX (由 setup_can.sh Step 2 完成)

流程:
  1. 枚举所有已激活的 CAN 接口及其 bus-info
  2. 依次提示用户晃动指定臂 (左master, 左slave, 右master, 右slave)
  3. 自动检测哪个接口在动, 记录映射
  4. 输出校准结果, 保存到 config/pipers.yml 和 activate_can.sh
"""

import os
import re
import select
import subprocess
import sys
import time

# ── 配置 ─────────────────────────────────────────────────────────────────────

ROLES = [
    ("left_master", "左 master (示教左臂)"),
    ("left_slave", "左 slave (执行左臂)"),
    ("right_master", "右 master (示教右臂)"),
    ("right_slave", "右 slave (执行右臂)"),
]

SYMBOLIC_NAMES = {
    "left_master": "can_left_mas",
    "left_slave": "can_left_slave",
    "right_master": "can_right_mas",
    "right_slave": "can_right_slave",
}

MOVE_THRESHOLD = 100
DETECT_SECONDS = 8
BITRATE = 1000000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # can_tools/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def get_can_interfaces():
    """返回 {接口名: bus_info} 字典, 只返回 UP 状态的接口"""
    out = subprocess.run(
        ["ip", "-br", "link", "show", "type", "can"],
        capture_output=True, text=True, timeout=5,
    )
    result = {}
    for line in out.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        name = parts[0]
        # 只选 UP 的
        if "UP" not in line:
            continue
        try:
            out2 = subprocess.run(
                ["sudo", "ethtool", "-i", name],
                capture_output=True, text=True, timeout=5,
            )
            bus = ""
            for l in out2.stdout.split("\n"):
                if "bus-info" in l:
                    idx = l.index("bus-info:")
                    bus = l[idx + len("bus-info:"):].strip()
            result[name] = bus
        except Exception:
            result[name] = ""
    return result


def check_can_ready(iface):
    """检查接口是否 UP 且 bitrate 已设置 (不依赖 candump, Piper 静止时不主动发帧)"""
    try:
        out = subprocess.run(
            ["ip", "-details", "link", "show", iface],
            capture_output=True, text=True, timeout=5,
        )
        text = out.stdout
        is_up = "state UP" in text
        has_bitrate = "bitrate" in text
        return is_up and has_bitrate
    except Exception:
        return False


def classify_slave_master(interfaces):
    """自动分类 slave/master: slave 有伺服环, 静止时关节值非零; master 无伺服, 静止时全零。
    返回 (slave_list, master_list)"""
    from piper_sdk import C_PiperInterface_V2

    slaves = []
    masters = []

    for iface in interfaces:
        try:
            p = C_PiperInterface_V2(can_name=iface)
            p.ConnectPort()
            time.sleep(1.5)  # 等 SDK 收到反馈帧
            msgs = p.GetArmJointMsgs()
            js = msgs.joint_state
            vals = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
            hz = msgs.Hz
            # slave: 伺服环持续通信, Hz > 0 且关节值不全为零
            if hz > 0 and any(v != 0 for v in vals):
                slaves.append(iface)
            else:
                masters.append(iface)
        except Exception:
            masters.append(iface)  # 连不上的归为 master (保守处理)

    return slaves, masters


def detect_moving_sdk(interfaces, seconds):
    """用 SDK 读关节角度变化量, 检测哪个 slave 在动。返回 (best_iface, deltas_dict)"""
    from piper_sdk import C_PiperInterface_V2

    pipers = {}
    for iface in interfaces:
        try:
            p = C_PiperInterface_V2(can_name=iface)
            p.ConnectPort()
            pipers[iface] = p
        except Exception:
            pass

    if not pipers:
        return None, {}

    time.sleep(1)

    def read(iface):
        msgs = pipers[iface].GetArmJointMsgs()
        js = msgs.joint_state
        return [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]

    baselines = {iface: read(iface) for iface in pipers}
    max_deltas = {iface: 0 for iface in pipers}
    t0 = time.time()
    last_print = 0

    while time.time() - t0 < seconds:
        time.sleep(0.2)
        elapsed = time.time() - t0

        for iface in pipers:
            cur = read(iface)
            base = baselines[iface]
            if cur and base:
                diffs = [abs(a - b) for a, b in zip(cur, base)]
                md = max(diffs)
                if md > max_deltas[iface]:
                    max_deltas[iface] = md

        if int(elapsed) > last_print:
            last_print = int(elapsed)
            remaining = seconds - int(elapsed)
            parts = []
            for iface in pipers:
                marker = " <<<" if max_deltas[iface] > MOVE_THRESHOLD else ""
                parts.append(f"{iface}={max_deltas[iface]:5d}{marker}")
            sys.stdout.write(f"\r  [{remaining:2d}s] {' | '.join(parts)}   ")
            sys.stdout.flush()

    sys.stdout.write("\n")

    best = max(max_deltas, key=max_deltas.get)
    return best if max_deltas[best] > MOVE_THRESHOLD else None, max_deltas


def detect_moving_candump(interfaces, seconds):
    """用 candump 计帧数, 检测哪个 master 在动。master 静止时无帧, 晃动时有帧。
    返回 (best_iface, frame_counts_dict)"""
    import signal

    procs = {}
    for iface in interfaces:
        # candump 持续输出, 每行一帧
        p = subprocess.Popen(
            ["candump", iface],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        procs[iface] = p

    t0 = time.time()
    last_print = 0
    counts = {iface: 0 for iface in interfaces}

    try:
        while time.time() - t0 < seconds:
            time.sleep(0.3)
            elapsed = time.time() - t0

            # 非阻塞读取已有输出, 统计行数
            for iface, p in procs.items():
                while select.select([p.stdout], [], [], 0)[0]:
                    line = p.stdout.readline()
                    if line:
                        counts[iface] += 1
                    else:
                        break

            if int(elapsed) > last_print:
                last_print = int(elapsed)
                remaining = seconds - int(elapsed)
                parts = []
                for iface in interfaces:
                    marker = " <<<" if counts[iface] > 10 else ""
                    parts.append(f"{iface}={counts[iface]:5d}{marker}")
                sys.stdout.write(f"\r  [{remaining:2d}s] {' | '.join(parts)}   ")
                sys.stdout.flush()
    finally:
        for p in procs.values():
            p.terminate()
            p.wait()

    sys.stdout.write("\n")

    if not counts:
        return None, counts
    best = max(counts, key=counts.get)
    return best if counts[best] > 10 else None, counts


# ── 配置写入 ─────────────────────────────────────────────────────────────────

def write_pipers_yml(mapping, bus_infos):
    """更新 config/pipers.yml"""
    path = os.path.join(PROJECT_ROOT, "config", "pipers.yml")

    lines = [
        "# Piper 双臂配置 — sim01 部署",
        f"# 通过 calibrate_can_mapping.py 校准, {time.strftime('%Y-%m-%d %H:%M')}",
        "#",
        "# 启动前运行: bash can_tools/setup_can.sh --quick",
        "",
        "arms:",
    ]

    for role, desc in ROLES:
        iface = mapping[role]
        bus = bus_infos.get(iface, "")
        symbolic = SYMBOLIC_NAMES[role]
        mode = 0 if "master" in role else 1
        side = "左" if "left" in role else "右"
        kind = "master" if "master" in role else "slave"
        topic_prefix = "/master" if "master" in role else "/puppet"
        topic_side = "left" if "left" in role else "right"

        lines += [
            f"  {role}:",
            f"    can_physical: {iface}",
            f"    can_symbolic: {symbolic}",
            f'    usb_bus_info: "{bus}"',
            f"    role: {side}臂 ({kind})",
            f"    mode: {mode}",
            f"    dof: 6",
            f"    feedback_hz: 200",
            f"    can_bitrate: {BITRATE}",
            f"    ros2_topic_joint: {topic_prefix}/joint_{topic_side}",
            "",
        ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  已写入 {path}")


def write_activate_script(mapping, bus_infos):
    """重写 can_tools/activate_can.sh"""
    path = os.path.join(SCRIPT_DIR, "activate_can.sh")
    if not os.path.exists(path):
        print(f"  {path} 不存在, 跳过")
        return

    with open(path) as f:
        content = f.read()

    # 构建新的映射条目
    slave_entries = []
    master_entries = []
    for role, _ in ROLES:
        bus = bus_infos.get(mapping[role], "")
        symbolic = SYMBOLIC_NAMES[role]
        entry = f'    "{bus}:{symbolic}"'
        if "slave" in role:
            slave_entries.append(entry)
        else:
            master_entries.append(entry)

    # 替换 SLAVE_MAPPINGS 块
    content = re.sub(
        r'SLAVE_MAPPINGS=\([^)]*\)',
        'SLAVE_MAPPINGS=(\n' + '\n'.join(slave_entries) + '\n)',
        content,
    )
    # 替换 MASTER_MAPPINGS 块
    content = re.sub(
        r'MASTER_MAPPINGS=\([^)]*\)',
        'MASTER_MAPPINGS=(\n' + '\n'.join(master_entries) + '\n)',
        content,
    )

    # 更新头部注释
    comment_lines = []
    for role, desc in ROLES:
        bus = bus_infos.get(mapping[role], "?")
        symbolic = SYMBOLIC_NAMES[role]
        comment_lines.append(f"#   {bus} → {symbolic}  ({desc})")
    comment_block = "\n".join(comment_lines)
    content = re.sub(
        r'# sim01 bus-info \(.*?\n(?:#   .*\n)+',
        f"# sim01 bus-info ({time.strftime('%Y-%m-%d')} calibrate_can_mapping.py 校准):\n{comment_block}\n",
        content,
    )

    with open(path, "w") as f:
        f.write(content)
    print(f"  已更新 {path}")


def write_activate_can_arms(mapping, bus_infos):
    """更新 kai0 下的 activate_can_arms.sh"""
    path = os.path.join(
        PROJECT_ROOT, "kai0", "train_deploy_alignment", "dagger", "agilex",
        "activate_can_arms.sh",
    )
    if not os.path.exists(path):
        return

    lines = [
        "#!/bin/bash",
        "# Activate all four USB-CAN interfaces for dual master + dual slave.",
        f"# 由 calibrate_can_mapping.py 自动生成, {time.strftime('%Y-%m-%d %H:%M')}",
        "#",
        "# sim01 bus-info 映射:",
    ]
    for role, desc in ROLES:
        bus = bus_infos.get(mapping[role], "?")
        lines.append(f"#   {bus} → {SYMBOLIC_NAMES[role]} ({desc})")
    lines.append("")

    for role, _ in ROLES:
        bus = bus_infos.get(mapping[role], "")
        symbolic = SYMBOLIC_NAMES[role]
        lines.append(f'bash ./can_activate.sh {symbolic:20s} {BITRATE} "{bus}"')

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  已更新 {path}")


def _manual_pick(candidates, desc):
    """手动选择接口"""
    print(f"  可选接口: {', '.join(candidates)}")
    manual = input(f"  请手动输入 【{desc}】 对应的接口名 (或 Enter 跳过): ").strip()
    if manual in candidates:
        print(f"  已绑定: {manual}")
        return manual
    if manual:
        print(f"  无效输入 '{manual}'")
    return None


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CAN-臂映射交互式校准")
    print("=" * 60)
    print()

    # 1. 枚举接口
    all_interfaces = get_can_interfaces()
    if not all_interfaces:
        print("未检测到已激活的 CAN 接口。")
        print("请先运行: bash can_tools/setup_can.sh")
        sys.exit(1)

    print(f"检测到 {len(all_interfaces)} 个活跃 CAN 接口:")
    not_ready = []
    for iface, bus in all_interfaces.items():
        ready = check_can_ready(iface)
        status = "UP ✓" if ready else "异常 ✗"
        if not ready:
            not_ready.append(iface)
        print(f"  {iface:10s}  bus={bus:15s}  {status}")
    print()

    if not_ready:
        print(f"警告: {', '.join(not_ready)} 接口状态异常 (未 UP 或 bitrate 未设置)。")
        cont = input("是否继续校准? [Y/n] ").strip().lower()
        if cont not in ("", "y", "yes"):
            sys.exit(0)
        print()

    # ══ 阶段 1: 自动分类 slave / master ══════════════════════════════════════
    print("=" * 60)
    print("  阶段 1: 自动识别 slave / master")
    print("=" * 60)
    print()
    print("  slave 臂有伺服环, 静止时持续通信; master 臂无伺服, 静止时无数据。")
    print("  正在连接所有接口并检测...")
    print()

    iface_list = list(all_interfaces.keys())
    slaves, masters = classify_slave_master(iface_list)

    print(f"  Slave  接口 ({len(slaves)}): {', '.join(slaves) if slaves else '无'}")
    print(f"  Master 接口 ({len(masters)}): {', '.join(masters) if masters else '无'}")
    print()

    if len(slaves) != 2 or len(masters) != 2:
        print(f"  预期 2 slave + 2 master, 实际 {len(slaves)} slave + {len(masters)} master")
        print("  自动分类结果异常, 将回退到手动模式。")
        # 回退: 全部手动
        slaves = []
        masters = iface_list[:]
    else:
        confirm = input("  分类正确? [Y/n] ").strip().lower()
        if confirm not in ("", "y", "yes"):
            print("  请手动指定。")
            print(f"  所有接口: {', '.join(iface_list)}")
            s = input("  输入两个 slave 接口名 (空格分隔): ").strip().split()
            if len(s) == 2 and all(x in iface_list for x in s):
                slaves = s
                masters = [x for x in iface_list if x not in slaves]
            else:
                print("  输入无效, 退出。")
                sys.exit(1)
    print()

    mapping = {}
    # slave 和 master 各自需要区分左右
    SLAVE_ROLES = [r for r in ROLES if "slave" in r[0]]   # left_slave, right_slave
    MASTER_ROLES = [r for r in ROLES if "master" in r[0]]  # left_master, right_master

    # ══ 阶段 2: 区分 slave 左右 (SDK 关节值检测) ═════════════════════════════
    if len(slaves) >= 2:
        print("=" * 60)
        print("  阶段 2: 区分 slave 左右 (晃动 slave 臂)")
        print("=" * 60)
        print()

        remaining_slaves = slaves[:]
        for role, desc in SLAVE_ROLES:
            if len(remaining_slaves) == 1:
                last = remaining_slaves[0]
                print(f"  仅剩一个 slave, 自动绑定: 【{desc}】 = {last}")
                mapping[role] = last
                remaining_slaves.remove(last)
                continue

            print("-" * 60)
            print(f"  请晃动 【{desc}】")
            print(f"  候选: {', '.join(remaining_slaves)}")
            print(f"  检测时间: {DETECT_SECONDS} 秒 (SDK 关节值变化)")
            print()
            input("  准备好后按 Enter 开始检测...")

            detected, deltas = detect_moving_sdk(remaining_slaves, DETECT_SECONDS)

            if detected:
                bus = all_interfaces.get(detected, "?")
                print(f"  ✓ 检测到 【{desc}】 → {detected} (bus: {bus}, delta: {deltas[detected]})")
                confirm = input("  确认? [Y/n] ").strip().lower()
                if confirm in ("", "y", "yes"):
                    mapping[role] = detected
                    remaining_slaves.remove(detected)
                else:
                    manual = _manual_pick(remaining_slaves, desc)
                    if manual:
                        mapping[role] = manual
                        remaining_slaves.remove(manual)
            else:
                print("  ✗ 未检测到明显运动")
                manual = _manual_pick(remaining_slaves, desc)
                if manual:
                    mapping[role] = manual
                    remaining_slaves.remove(manual)
            print()
    elif len(slaves) == 1:
        # 只有 1 个 slave, 需要手动指定是左还是右
        print(f"  只有 1 个 slave: {slaves[0]}")
        side = input("  它是 左slave 还是 右slave? [l/r] ").strip().lower()
        if side in ("l", "left"):
            mapping["left_slave"] = slaves[0]
        else:
            mapping["right_slave"] = slaves[0]
        print()

    # ══ 阶段 3: 区分 master 左右 (candump 计帧检测) ══════════════════════════
    if len(masters) >= 2:
        print("=" * 60)
        print("  阶段 3: 区分 master 左右 (晃动 master 臂)")
        print("  master 无伺服, 使用 candump 计帧检测")
        print("=" * 60)
        print()

        remaining_masters = masters[:]
        for role, desc in MASTER_ROLES:
            if len(remaining_masters) == 1:
                last = remaining_masters[0]
                print(f"  仅剩一个 master, 自动绑定: 【{desc}】 = {last}")
                mapping[role] = last
                remaining_masters.remove(last)
                continue

            print("-" * 60)
            print(f"  请晃动 【{desc}】 (持续晃动, 不要停)")
            print(f"  候选: {', '.join(remaining_masters)}")
            print(f"  检测时间: {DETECT_SECONDS} 秒 (candump 帧计数)")
            print()
            input("  准备好后按 Enter 开始检测...")

            detected, counts = detect_moving_candump(remaining_masters, DETECT_SECONDS)

            if detected:
                bus = all_interfaces.get(detected, "?")
                print(f"  ✓ 检测到 【{desc}】 → {detected} (bus: {bus}, frames: {counts[detected]})")
                confirm = input("  确认? [Y/n] ").strip().lower()
                if confirm in ("", "y", "yes"):
                    mapping[role] = detected
                    remaining_masters.remove(detected)
                else:
                    manual = _manual_pick(remaining_masters, desc)
                    if manual:
                        mapping[role] = manual
                        remaining_masters.remove(manual)
            else:
                print("  ✗ 未检测到 CAN 帧 (确保在晃动臂)")
                manual = _manual_pick(remaining_masters, desc)
                if manual:
                    mapping[role] = manual
                    remaining_masters.remove(manual)
            print()
    elif len(masters) == 1:
        print(f"  只有 1 个 master: {masters[0]}")
        side = input("  它是 左master 还是 右master? [l/r] ").strip().lower()
        if side in ("l", "left"):
            mapping["left_master"] = masters[0]
        else:
            mapping["right_master"] = masters[0]
        print()

    # 3. 显示结果
    print("=" * 60)
    print("  校准结果")
    print("=" * 60)
    print()
    print(f"  {'角色':<20s} {'接口':<10s} {'Bus-Info':<15s} {'符号名':<20s}")
    print(f"  {'─'*18:<20s} {'─'*8:<10s} {'─'*13:<15s} {'─'*18:<20s}")
    for role, desc in ROLES:
        if role in mapping:
            iface = mapping[role]
            bus = all_interfaces.get(iface, "?")
            symbolic = SYMBOLIC_NAMES[role]
            print(f"  {desc:<20s} {iface:<10s} {bus:<15s} {symbolic:<20s}")
        else:
            print(f"  {desc:<20s} {'—':<10s} {'—':<15s} {'未绑定':<20s}")
    print()

    # 4. 保存
    mapped_count = len(mapping)
    if mapped_count == 0:
        print("未完成任何映射, 退出。")
        sys.exit(1)

    if mapped_count < 4:
        print(f"注意: 只完成了 {mapped_count}/4 个映射。")

    save = input("是否保存到配置文件? [Y/n] ").strip().lower()
    if save in ("", "y", "yes"):
        print()
        write_pipers_yml(mapping, all_interfaces)
        write_activate_script(mapping, all_interfaces)
        write_activate_can_arms(mapping, all_interfaces)
        print()
        print("配置已保存。")
    else:
        print("未保存。")
        sys.exit(0)


if __name__ == "__main__":
    main()
