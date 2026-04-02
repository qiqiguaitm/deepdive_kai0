#!/usr/bin/env python3
"""
对比 websocket 模式 vs ros2 模式的推理结果

用法: 先分别采集两种模式的数据，保存为 npz，然后对比。

步骤:
  1. 启动 websocket 模式，运行本脚本 --collect ws
  2. 停掉，启动 ros2 模式，运行本脚本 --collect ros2
  3. 运行本脚本 --compare 对比
"""
import argparse
import sys
import os
import numpy as np

def collect(mode_name):
    import rclpy, time
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("mode_collector")
    actions = []
    states = []

    def action_cb(msg):
        actions.append(list(msg.position))

    def state_cb(msg):
        states.append(list(msg.position))

    node.create_subscription(JointState, "/policy/actions", action_cb, 10)
    node.create_subscription(JointState, "/puppet/joint_left", state_cb, 10)

    print(f"Collecting {mode_name} mode data for 10 seconds...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    a = np.array(actions)
    s = np.array(states) if states else np.array([])
    outfile = f"/tmp/mode_{mode_name}.npz"
    np.savez(outfile, actions=a, states=s)
    print(f"Saved {len(a)} actions, {len(s)} states to {outfile}")
    print(f"  Actions range: [{a.min():.4f}, {a.max():.4f}]")
    print(f"  Actions std:   {a.std():.4f}")
    print(f"  Actions mean per joint:")
    names = [f"L_j{i}" for i in range(7)] + [f"R_j{i}" for i in range(7)]
    for i in range(min(14, a.shape[1])):
        print(f"    {names[i]}: {a[:,i].mean():+.4f}")

def compare():
    ws_file = "/tmp/mode_ws.npz"
    ros2_file = "/tmp/mode_ros2.npz"

    if not os.path.exists(ws_file) or not os.path.exists(ros2_file):
        print(f"Missing data files. Run --collect ws and --collect ros2 first.")
        sys.exit(1)

    ws = np.load(ws_file)
    r2 = np.load(ros2_file)

    a_ws = ws['actions']
    a_r2 = r2['actions']

    print("=" * 60)
    print("WebSocket vs ROS2 Mode 推理结果对比")
    print("=" * 60)
    print(f"\n  WebSocket: {a_ws.shape[0]} frames, ROS2: {a_r2.shape[0]} frames")

    # 统计对比
    print(f"\n{'Metric':<25} {'WebSocket':>12} {'ROS2':>12} {'Diff':>12}")
    print("-" * 60)

    names = [f"L_j{i}" for i in range(7)] + [f"R_j{i}" for i in range(7)]
    for i in range(14):
        m_ws = a_ws[:, i].mean()
        m_r2 = a_r2[:, i].mean()
        diff = abs(m_ws - m_r2)
        print(f"  {names[i]} mean       {m_ws:+12.4f} {m_r2:+12.4f} {diff:12.4f}")

    print("-" * 60)
    print(f"  {'Overall mean':<23} {a_ws.mean():+12.4f} {a_r2.mean():+12.4f} {abs(a_ws.mean()-a_r2.mean()):12.4f}")
    print(f"  {'Overall std':<23} {a_ws.std():12.4f} {a_r2.std():12.4f} {abs(a_ws.std()-a_r2.std()):12.4f}")
    print(f"  {'Overall range':<23} [{a_ws.min():.3f},{a_ws.max():.3f}] [{a_r2.min():.3f},{a_r2.max():.3f}]")

    # Step jitter 对比
    j_ws = np.abs(np.diff(a_ws, axis=0)).mean()
    j_r2 = np.abs(np.diff(a_r2, axis=0)).mean()
    print(f"  {'Step jitter (avg)':<23} {j_ws:12.4f} {j_r2:12.4f} {abs(j_ws-j_r2):12.4f}")

    # 判定
    print(f"\n判定:")
    max_mean_diff = max(abs(a_ws[:, i].mean() - a_r2[:, i].mean()) for i in range(14))
    std_diff = abs(a_ws.std() - a_r2.std())
    print(f"  最大关节均值差异: {max_mean_diff:.4f} rad ({max_mean_diff*180/3.14:.1f} deg)")
    print(f"  标准差差异: {std_diff:.4f}")

    # 注意: 两种模式输入不完全相同(时间不同,观测不同), 所以不期望完全一致
    # 但统计分布应该相似
    if max_mean_diff < 0.3 and std_diff < 0.1:
        print(f"  结论: PASS — 两种模式统计分布一致")
    else:
        print(f"  结论: REVIEW — 均值差异 {max_mean_diff:.3f} rad, 可能是不同观测导致")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", choices=["ws", "ros2"], help="Collect data for mode")
    parser.add_argument("--compare", action="store_true", help="Compare collected data")
    args = parser.parse_args()

    if args.collect:
        collect(args.collect)
    elif args.compare:
        compare()
    else:
        parser.print_help()
