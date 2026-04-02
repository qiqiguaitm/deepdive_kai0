#!/usr/bin/env python3
"""从 /policy/actions topic 采样并验证动作质量"""
import rclpy, numpy as np, time
from rclpy.node import Node
from sensor_msgs.msg import JointState

rclpy.init()
node = Node("action_checker")
actions = []
def cb(msg): actions.append(list(msg.position))
node.create_subscription(JointState, "/policy/actions", cb, 10)

t0 = time.monotonic()
while len(actions) < 90 and time.monotonic() - t0 < 5:
    rclpy.spin_once(node, timeout_sec=0.1)
node.destroy_node()
rclpy.shutdown()

if not actions:
    print("NO DATA"); exit(1)

a = np.array(actions)
print(f"Collected {len(a)} frames in {time.monotonic()-t0:.1f}s ({len(a)/(time.monotonic()-t0):.1f} Hz)")
print(f"Shape: {a.shape}")
print(f"Range: [{a.min():.4f}, {a.max():.4f}]")
print(f"Std:   {a.std():.4f}")
print()
names = [f"L_j{i}" for i in range(7)] + [f"R_j{i}" for i in range(7)]
for i in range(14):
    print(f"  {names[i]}: [{a[:,i].min():+.3f}, {a[:,i].max():+.3f}] mean={a[:,i].mean():+.3f}")

diffs = np.abs(np.diff(a, axis=0))
print(f"\nStep jitter: avg={diffs.mean():.4f} max={diffs.max():.4f} rad")

in_limits = a.min() > -3.0 and a.max() < 3.0
smooth = diffs.max() < 0.5
non_zero = a.std() > 0.001

print()
print(f"Limits OK:   {'PASS' if in_limits else 'FAIL'}")
print(f"Smooth:      {'PASS' if smooth else 'FAIL'}")
print(f"Non-zero:    {'PASS' if non_zero else 'FAIL'}")
print(f"Overall:     {'PASS' if in_limits and smooth and non_zero else 'FAIL'}")
