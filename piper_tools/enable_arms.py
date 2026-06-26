#!/usr/bin/env python3
"""安全使能/失能指定机械臂 — 只在【当前位姿原地上力】, 不发任何关节目标、不运动。

⚠️ 主臂使能后会变僵硬、拖不动(无拖动示教)。正常遥操请用 start_data_collect.sh 让
   teleop 节点按正确模式使能(主臂带拖动示教)。本脚本仅用于手动给臂上力/卸力的场景。

用法:
  python3 enable_arms.py                         # 默认使能两个从臂(原地上力)
  python3 enable_arms.py --cans can_right_mas     # 指定某条臂
  python3 enable_arms.py --cans can_left_slave,can_right_slave
  python3 enable_arms.py --disable --cans ...      # 失能(卸力)
只调 ConnectPort + EnableArm/DisableArm + 读状态; 不发 JointCtrl/MotionCtrl → 不运动。
"""
import argparse
import time

from piper_sdk import C_PiperInterface_V2

SLAVES = ["can_left_slave", "can_right_slave"]


def status(p):
    try:
        return p.GetArmEnableStatus()
    except Exception:
        return ["?"] * 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cans", default=",".join(SLAVES), help="逗号分隔 CAN 名(默认两个从臂)")
    ap.add_argument("--disable", action="store_true", help="改为失能(卸力)")
    args = ap.parse_args()
    cans = [c.strip() for c in args.cans.split(",") if c.strip()]
    act = "失能" if args.disable else "使能"

    masters = [c for c in cans if "mas" in c]
    if masters and not args.disable:
        print(f"⚠️  你要使能主臂: {masters} —— 使能后会僵硬拖不动(无拖动示教)。")
        if input("确认继续? [yes/no]: ") != "yes":
            print("已取消。"); return

    for c in cans:
        try:
            p = C_PiperInterface_V2(c); p.ConnectPort(); time.sleep(0.4)
            before = status(p)
            if args.disable:
                p.DisableArm(7)
            else:
                # EnableArm(7): 7=全部电机, 在当前位姿上力, 不发目标
                p.EnableArm(7)
            time.sleep(0.6)
            after = status(p)
            print(f"{c:18} {act}: {before} -> {after}  {'✓全' + act if (all(after) ^ args.disable) else ''}")
        except Exception as e:
            print(f"{c:18} 失败: {e}")
    print(f"\n完成({act}, 原地未运动)。验证: python3 piper_tools/diag_left_arm_align.py 或重读 GetArmEnableStatus")


if __name__ == "__main__":
    main()
