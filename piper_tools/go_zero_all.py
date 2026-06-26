#!/usr/bin/env python3
"""把指定机械臂(默认 4 条)控制到零位 J=[0,0,0,0,0,0] + 夹爪 0。

⚠️ 这是【运动指令】: 会使能电机并移动到零位。运行前清空臂周围、手放急停旁。
   默认 dry-run 只打印计划; 必须加 --execute 且交互确认 yes 才真正运动。
   使不能的臂(无电/急停/不通信)会自动跳过, 不会动。

用法:
  python3 go_zero_all.py                       # dry-run, 只打印
  python3 go_zero_all.py --execute             # 真正回零(4 条, 速度 20%)
  python3 go_zero_all.py --execute --speed 15 --cans can_left_slave,can_right_slave
"""
import argparse
import time

from piper_sdk import C_PiperInterface_V2

ALL = ["can_left_mas", "can_left_slave", "can_right_mas", "can_right_slave"]


def jdeg(p):
    js = p.GetArmJointMsgs().joint_state
    return [round(getattr(js, f"joint_{i}") / 1000.0, 1) for i in range(1, 7)]


def enable(p, timeout=5.0):
    t0 = 0.0
    while not p.EnablePiper():
        time.sleep(0.05); t0 += 0.05
        if t0 >= timeout:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cans", default=",".join(ALL))
    ap.add_argument("--speed", type=int, default=20, help="速度百分比 (默认 20, 安全慢速)")
    ap.add_argument("--execute", action="store_true", help="真正运动(否则只 dry-run)")
    ap.add_argument("--hold", type=float, default=3.0, help="持续下发回零命令的秒数")
    args = ap.parse_args()
    cans = [c.strip() for c in args.cans.split(",") if c.strip()]
    sp = max(1, min(100, args.speed))

    # 先连上读当前位姿
    arms = {}
    for c in cans:
        try:
            p = C_PiperInterface_V2(c); p.ConnectPort(); time.sleep(0.4)
            arms[c] = p
            print(f"  {c:18} 当前 J={jdeg(p)}")
        except Exception as e:
            print(f"  {c:18} 连接失败: {e}")

    if not args.execute:
        print(f"\n[dry-run] 将把以上 {len(arms)} 条臂回零 (速度 {sp}%)。确认无误后加 --execute 重跑。")
        return

    print(f"\n⚠️ 即将让 {len(arms)} 条臂以 {sp}% 速度移动到零位。请确认臂周围无障碍、手放急停旁。")
    if input("确认回零? [yes/no]: ") != "yes":
        print("已取消, 未运动。"); return

    # 逐条使能 + 下发回零; 持续下发 hold 秒确保到位
    ready = []
    for c, p in arms.items():
        if enable(p, timeout=5.0):
            p.MotionCtrl_2(0x01, 0x01, sp, 0x00)  # CAN控制 + 关节模式 + 速度
            ready.append((c, p))
            print(f"  {c:18} 已使能 → 下发回零")
        else:
            print(f"  {c:18} ⚠️ 使能超时, 跳过(不会动)")

    t = 0.0
    while t < args.hold:
        for c, p in ready:
            p.MotionCtrl_2(0x01, 0x01, sp, 0x00)
            p.JointCtrl(0, 0, 0, 0, 0, 0)
            p.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(0.1); t += 0.1

    print("\n回零命令已下发。最终位姿:")
    for c, p in ready:
        time.sleep(0.1)
        print(f"  {c:18} J={jdeg(p)}")
    print("(到位后如需卸力: python3 piper_tools/enable_arms.py --disable --cans <...>)")


if __name__ == "__main__":
    main()
