#!/usr/bin/env python3
"""把指定机械臂设为【从臂/普通可控模式】(MasterSlaveConfig 0xFC)。

用途: 标定后臂被卡在 MASTER 模式(0xFA)→ 没力/不报反馈/使不能。设回 0xFC 复位为
普通可控从臂。只改联动角色, 不动关节零点 → 不影响部署坐标系; teleop 启动会重设角色。

⚠️ 改完角色【必须把臂断电重启】才生效 (SDK 规定)。
用法:
  python3 set_arm_slave_mode.py                       # 默认 4 条全设为从臂(复位)
  python3 set_arm_slave_mode.py --cans can_right_mas,can_left_slave,can_right_slave
  python3 set_arm_slave_mode.py --master --cans can_left_mas   # (反向) 设为主臂 0xFA
"""
import argparse
import time

from piper_sdk import C_PiperInterface_V2

ALL = ["can_left_mas", "can_left_slave", "can_right_mas", "can_right_slave"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cans", default=",".join(ALL))
    ap.add_argument("--master", action="store_true", help="改设为主臂(0xFA)而非从臂(0xFC)")
    args = ap.parse_args()
    cans = [c.strip() for c in args.cans.split(",") if c.strip()]
    role_byte = 0xFA if args.master else 0xFC
    role_name = "主臂(0xFA)" if args.master else "从臂(0xFC)"

    print(f"将以下臂设为 {role_name}: {cans}")
    print("注意: 改完必须断电重启臂才生效; 只改联动角色, 不动关节零点。")
    if input("确认? [yes/no]: ") != "yes":
        print("已取消, 未写入。"); return

    for c in cans:
        try:
            p = C_PiperInterface_V2(c); p.ConnectPort(); time.sleep(0.4)
            p.MasterSlaveConfig(role_byte, 0, 0, 0)
            time.sleep(0.3)
            print(f"  {c:18} 已下发 MasterSlaveConfig(0x{role_byte:02X})  ✓")
        except Exception as e:
            print(f"  {c:18} 失败: {e}")
    print(f"\n完成。现在请【断电重启全部臂】(先 slave 后 master),再重启采集:")
    print("  ./web/data_manager/run.sh stop ; ./start_scripts/start_data_collect.sh")


if __name__ == "__main__":
    main()
