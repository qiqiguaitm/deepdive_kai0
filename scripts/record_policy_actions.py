#!/usr/bin/env python3
"""录制 /policy/actions 到 npy 文件."""
import sys, time, argparse
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class Recorder(Node):
    def __init__(self, duration):
        super().__init__('action_recorder')
        self.duration = duration
        self.actions = []
        self.create_subscription(JointState, '/policy/actions', self._cb, 10)
        self._t0 = None

    def _cb(self, msg):
        if self._t0 is None:
            self._t0 = time.monotonic()
            self.get_logger().info(f'Recording {self.duration}s...')
        if time.monotonic() - self._t0 > self.duration:
            return
        self.actions.append(np.array(msg.position))
        if len(self.actions) % 100 == 0:
            self.get_logger().info(f'  {len(self.actions)} steps')

    @property
    def done(self):
        return self._t0 is not None and (time.monotonic() - self._t0) > self.duration

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=10)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    rclpy.init()
    rec = Recorder(args.duration)
    while not rec.done and rclpy.ok():
        rclpy.spin_once(rec, timeout_sec=0.1)
    arr = np.array(rec.actions)
    np.save(args.output, arr)
    print(f'Saved {len(arr)} steps to {args.output}')
    rec.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
