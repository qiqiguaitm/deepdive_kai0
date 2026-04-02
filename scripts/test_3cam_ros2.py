#!/usr/bin/env python3
"""
ROS2 三相机 RGB+Depth 30fps 验证脚本

前置条件:
  ros2 launch scripts/launch_3cam.py   (另一个终端)

本脚本订阅 6 个 topic (3 RGB + 3 Depth)，统计 10 秒内的:
  - 实际接收 FPS
  - 帧间延迟 jitter
  - 图像尺寸和内容有效性 (非全黑)
  - 端到端延迟 (ROS header stamp → 收到时刻)

用法:
  source /opt/ros/jazzy/setup.bash
  python3 scripts/test_3cam_ros2.py [--duration 10]
"""
import argparse
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


TOPICS = [
    # (topic, label, expected_type)
    ('/camera_f/camera_f/color/image_raw',       'D435  RGB',   'rgb'),
    ('/camera_f/camera_f/depth/image_rect_raw',  'D435  Depth', 'depth'),
    ('/camera_l/camera_l/color/image_rect_raw',  'D405-L RGB',  'rgb'),
    ('/camera_l/camera_l/depth/image_rect_raw',  'D405-L Depth','depth'),
    ('/camera_r/camera_r/color/image_rect_raw',  'D405-R RGB',  'rgb'),
    ('/camera_r/camera_r/depth/image_rect_raw',  'D405-R Depth','depth'),
]


class TopicStats:
    def __init__(self, label, img_type):
        self.label = label
        self.img_type = img_type
        self.count = 0
        self.first_ts = None
        self.last_ts = None
        self.intervals = []
        self.e2e_latencies = []   # header stamp → receive time (ms)
        self.shape = None
        self.dtype = None
        self.mean_val = None
        self.nonzero_ratio = None


class CameraTestNode(Node):
    def __init__(self, duration):
        super().__init__('camera_test_node')
        self.bridge = CvBridge()
        self.duration = duration
        self.stats = {}
        self.start_time = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        for topic, label, img_type in TOPICS:
            self.stats[topic] = TopicStats(label, img_type)
            self.create_subscription(
                Image, topic,
                lambda msg, t=topic: self._callback(msg, t),
                qos,
            )

        self.get_logger().info(f'Subscribing to {len(TOPICS)} topics, waiting {duration}s...')
        self.start_time = time.monotonic()

    def _callback(self, msg, topic):
        now = time.monotonic()
        st = self.stats[topic]
        st.count += 1

        if st.first_ts is None:
            st.first_ts = now
        if st.last_ts is not None:
            st.intervals.append(now - st.last_ts)
        st.last_ts = now

        # End-to-end latency
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        clock_sec = self.get_clock().now().nanoseconds * 1e-9
        e2e = (clock_sec - stamp_sec) * 1000  # ms
        if 0 < e2e < 5000:  # filter unreasonable values
            st.e2e_latencies.append(e2e)

        # Image info (sample first frame only)
        if st.shape is None:
            try:
                if st.img_type == 'rgb':
                    img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                else:
                    img = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                st.shape = img.shape
                st.dtype = str(img.dtype)
                st.mean_val = float(img.mean())
                st.nonzero_ratio = float(np.count_nonzero(img) / img.size)
            except Exception:
                pass

    def is_done(self):
        return time.monotonic() - self.start_time > self.duration


def main():
    parser = argparse.ArgumentParser(description='ROS2 三相机验证')
    parser.add_argument('--duration', type=int, default=10, help='采集秒数')
    args = parser.parse_args()

    rclpy.init()
    node = CameraTestNode(args.duration)

    try:
        while rclpy.ok() and not node.is_done():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    # ── 报告 ──────────────────────────────────────────────────────────────
    print('\n' + '=' * 75)
    print('ROS2 三相机 RGB+Depth 验证报告')
    print('=' * 75)

    all_ok = True
    for topic, label, img_type in TOPICS:
        st = node.stats[topic]
        print(f'\n  [{label}] {topic}')

        if st.count == 0:
            print(f'    NO DATA RECEIVED')
            all_ok = False
            continue

        elapsed = st.last_ts - st.first_ts if st.last_ts and st.first_ts else 1
        fps = (st.count - 1) / elapsed if elapsed > 0 and st.count > 1 else 0

        intervals = np.array(st.intervals) * 1000 if st.intervals else np.array([0])
        e2e = np.array(st.e2e_latencies) if st.e2e_latencies else np.array([0])

        print(f'    帧数:      {st.count}')
        print(f'    实际FPS:   {fps:.1f}')
        print(f'    图像:      {st.shape}  dtype={st.dtype}')
        print(f'    均值:      {st.mean_val:.1f}  非零率: {st.nonzero_ratio:.2%}')
        print(f'    帧间隔:    avg={intervals.mean():.1f}ms  std={intervals.std():.1f}ms  max={intervals.max():.1f}ms')
        print(f'    端到端延迟: avg={e2e.mean():.1f}ms  p99={np.percentile(e2e,99):.1f}ms')

        # 判定
        if fps < 25:
            print(f'    [WARN] FPS < 25')
            all_ok = False
        if st.nonzero_ratio is not None and st.nonzero_ratio < 0.01:
            print(f'    [WARN] 图像几乎全黑')

    print('\n' + '-' * 75)
    if all_ok:
        print('结论: PASS')
    else:
        print('结论: WARN - 部分 topic 未达标，见上方详情')
    print('-' * 75)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
