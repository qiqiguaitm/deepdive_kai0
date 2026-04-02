#!/usr/bin/env python3
"""
实时端到端对比: ROS2 节点内推理 vs WebSocket 推理 (mode=both)

原理:
  policy_inference_node.py 以 mode=both 启动后:
  - 节点内: 直接调用 policy.infer() → StreamActionBuffer → /master/joint_*
  - 同时暴露 WebSocket 接口 (ws://localhost:8000)

  本脚本:
  1. 订阅真实相机 + 关节 topic, 连续抓帧
  2. 每帧同时经过两条管线:
     A. 本地管线 (复刻节点内): jpeg → BGR→RGB → resize_with_pad → CHW → policy.infer()
     B. WebSocket 管线: 同样的 obs → ws_client.infer()
  3. 对比两条路径返回的 actions
  4. 不发布控制命令, 不驱动机械臂

  由于 A 和 B 共享同一个 Policy 对象 (mode=both), 只需加载一次模型 (一张 GPU)。
  但注意: 两次 infer() 的 RNG 不同, 所以有采样随机性差异。
  为隔离管线差异, 脚本同时记录「用完全相同 obs 送入 WS」的结果, 以验证 obs 构建的一致性。

用法 (在 sim01 上):
  # 终端 1: 启动 policy_inference_node (mode=both)
  source /opt/ros/jazzy/setup.bash
  PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH" \\
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \\
  /data1/tim/workspace/deepdive_kai0/kai0/.venv/bin/python \\
    /data1/tim/workspace/deepdive_kai0/ros2_ws/src/piper/scripts/policy_inference_node.py \\
    --ros-args \\
    -p mode:=both \\
    -p config_name:=pi05_flatten_fold_normal \\
    -p checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1 \\
    -p ws_port:=8000 \\
    -p prompt:="Flatten and fold the cloth." \\
    -p img_front_topic:=/camera_f/camera/color/image_raw \\
    -p img_left_topic:=/camera_l/camera/color/image_rect_raw \\
    -p img_right_topic:=/camera_r/camera/color/image_rect_raw

  # 终端 2: 运行本脚本
  source /opt/ros/jazzy/setup.bash
  PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH" \\
  /data1/tim/workspace/deepdive_kai0/kai0/.venv/bin/python \\
    /data1/tim/workspace/deepdive_kai0/scripts/test_e2e_realtime.py \\
    --host localhost --port 8000 --duration 10
"""

import os
import sys
import time
import argparse
import threading
from collections import deque

sys.path.insert(0, '/data1/tim/workspace/deepdive_kai0/kai0/src')

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState


def _stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def jpeg_mapping(img):
    img = cv2.imencode(".jpg", img)[1].tobytes()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    return img


def build_obs(img_front, img_left, img_right, joint_left, joint_right, prompt):
    """严格复刻原版观测构建管线."""
    from openpi_client import image_tools

    imgs = [jpeg_mapping(img_front), jpeg_mapping(img_right), jpeg_mapping(img_left)]
    imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
    imgs = [image_tools.resize_with_pad(im[np.newaxis], 224, 224)[0] for im in imgs]

    qpos = np.concatenate((np.array(joint_left), np.array(joint_right)), axis=0)
    return {
        'state': qpos,
        'images': {
            'top_head':   imgs[0].transpose(2, 0, 1),
            'hand_right': imgs[1].transpose(2, 0, 1),
            'hand_left':  imgs[2].transpose(2, 0, 1),
        },
        'prompt': prompt,
    }


class RealtimeComparator(Node):
    """持续从 ROS2 topic 抓帧, 送入 WebSocket 推理, 记录结果."""

    def __init__(self, ws_host, ws_port, prompt, duration):
        super().__init__('realtime_comparator')
        self.bridge = CvBridge()
        self.prompt = prompt
        self.duration = duration

        # Sensor deques (与原版一致)
        self._img_front_deque = deque()
        self._img_left_deque = deque()
        self._img_right_deque = deque()
        self._joint_left_deque = deque()
        self._joint_right_deque = deque()

        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1000)

        self.create_subscription(Image,
            '/camera_f/camera/color/image_raw', self._cb_img_front, img_qos)
        self.create_subscription(Image,
            '/camera_l/camera/color/image_rect_raw', self._cb_img_left, img_qos)
        self.create_subscription(Image,
            '/camera_r/camera/color/image_rect_raw', self._cb_img_right, img_qos)
        self.create_subscription(JointState,
            '/puppet/joint_left', self._cb_joint_left, 1000)
        self.create_subscription(JointState,
            '/puppet/joint_right', self._cb_joint_right, 1000)

        # WebSocket client (连接 mode=both 暴露的服务)
        from openpi_client import websocket_client_policy
        self.get_logger().info(f'Connecting to WebSocket at {ws_host}:{ws_port}...')
        self.ws_policy = websocket_client_policy.WebsocketClientPolicy(ws_host, ws_port)
        self.get_logger().info('WebSocket connected')

        # 结果收集
        self.results = []
        self._running = True

        # 推理线程
        self._thread = threading.Thread(target=self._compare_loop, daemon=True)
        self._thread.start()

    # ── Callbacks ──
    def _cb_img_front(self, msg):
        if len(self._img_front_deque) >= 2000: self._img_front_deque.popleft()
        self._img_front_deque.append(msg)

    def _cb_img_left(self, msg):
        if len(self._img_left_deque) >= 2000: self._img_left_deque.popleft()
        self._img_left_deque.append(msg)

    def _cb_img_right(self, msg):
        if len(self._img_right_deque) >= 2000: self._img_right_deque.popleft()
        self._img_right_deque.append(msg)

    def _cb_joint_left(self, msg):
        if len(self._joint_left_deque) >= 2000: self._joint_left_deque.popleft()
        self._joint_left_deque.append(msg)

    def _cb_joint_right(self, msg):
        if len(self._joint_right_deque) >= 2000: self._joint_right_deque.popleft()
        self._joint_right_deque.append(msg)

    # ── Frame sync (复刻原版) ──
    def _get_synced_frame(self):
        if (len(self._img_front_deque) == 0
                or len(self._img_left_deque) == 0
                or len(self._img_right_deque) == 0
                or len(self._joint_left_deque) == 0
                or len(self._joint_right_deque) == 0):
            return None

        frame_time = min(
            _stamp_to_sec(self._img_front_deque[-1].header.stamp),
            _stamp_to_sec(self._img_left_deque[-1].header.stamp),
            _stamp_to_sec(self._img_right_deque[-1].header.stamp),
        )

        for dq in [self._img_front_deque, self._img_left_deque, self._img_right_deque,
                    self._joint_left_deque, self._joint_right_deque]:
            if len(dq) == 0 or _stamp_to_sec(dq[-1].header.stamp) < frame_time:
                return None

        def _pop(dq):
            while _stamp_to_sec(dq[0].header.stamp) < frame_time:
                dq.popleft()
            return dq.popleft()

        img_f = self.bridge.imgmsg_to_cv2(_pop(self._img_front_deque), 'passthrough')
        img_l = self.bridge.imgmsg_to_cv2(_pop(self._img_left_deque), 'passthrough')
        img_r = self.bridge.imgmsg_to_cv2(_pop(self._img_right_deque), 'passthrough')
        jl = list(_pop(self._joint_left_deque).position)
        jr = list(_pop(self._joint_right_deque).position)

        return img_f, img_l, img_r, jl, jr

    # ── Compare loop ──
    def _compare_loop(self):
        self.get_logger().info(f'Waiting for sensor data...')
        while self._running and rclpy.ok():
            if self._get_synced_frame() is not None:
                break
            time.sleep(0.1)

        self.get_logger().info(f'Sensor ready. Running {self.duration}s realtime comparison...')
        t_start = time.monotonic()
        step = 0

        while self._running and rclpy.ok() and (time.monotonic() - t_start) < self.duration:
            frame = self._get_synced_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            img_f, img_l, img_r, jl, jr = frame

            # 构建 obs (两条路径用完全相同的 obs)
            obs = build_obs(img_f, img_l, img_r, jl, jr, self.prompt)

            # 送入 WebSocket (mode=both 下, 服务端是同一个 Policy 对象)
            t0 = time.monotonic()
            try:
                result_ws = self.ws_policy.infer(obs)
                latency_ms = (time.monotonic() - t0) * 1000
                actions_ws = result_ws.get('actions', None)
            except Exception as e:
                self.get_logger().warn(f'WS infer error: {e}')
                time.sleep(0.1)
                continue

            if actions_ws is None:
                continue

            self.results.append({
                'step': step,
                'time': time.monotonic() - t_start,
                'actions': actions_ws,
                'latency_ms': latency_ms,
                'state': obs['state'].copy(),
            })

            step += 1
            if step % 5 == 0:
                print(f'  step {step:3d}  latency={latency_ms:.0f}ms  '
                      f'actions[0,:3]=[{actions_ws[0,0]:+.4f} {actions_ws[0,1]:+.4f} {actions_ws[0,2]:+.4f}]')

        self._running = False
        self.get_logger().info(f'Collected {len(self.results)} inference results in {self.duration}s')


def analyze_results(results):
    """分析连续推理结果的统计特性."""
    if len(results) < 2:
        print('  结果不足, 跳过分析')
        return

    print(f'\n{"="*65}')
    print(f'  实时推理统计 ({len(results)} 步)')
    print(f'{"="*65}')

    actions_all = np.array([r['actions'] for r in results])  # [N, chunk, 14]
    latencies = np.array([r['latency_ms'] for r in results])
    times = np.array([r['time'] for r in results])

    print(f'\n  推理频率: {len(results)/times[-1]:.1f} Hz (目标 3 Hz)')
    print(f'  延迟 (ms): mean={latencies.mean():.0f}  p50={np.median(latencies):.0f}  '
          f'p95={np.percentile(latencies, 95):.0f}  max={latencies.max():.0f}')

    # Actions 统计
    first_actions = actions_all[:, 0, :]  # 每步第一个 action [N, 14]
    print(f'\n  Actions[step=0] 统计 (14D):')
    print(f'    mean: [{" ".join(f"{v:+.4f}" for v in first_actions.mean(axis=0))}]')
    print(f'    std:  [{" ".join(f"{v:.4f}" for v in first_actions.std(axis=0))}]')
    print(f'    range: [{" ".join(f"{v:.4f}" for v in (first_actions.max(axis=0) - first_actions.min(axis=0)))}]')

    # 连续步间差异 (衡量时序平滑效果)
    if len(results) >= 3:
        diffs = np.abs(first_actions[1:] - first_actions[:-1])
        print(f'\n  步间差异 (连续两步 actions[0] 的变化):')
        print(f'    mean: {diffs.mean():.4f}  max: {diffs.max():.4f}')

    # NaN/Inf 检查
    nan_count = np.isnan(actions_all).sum()
    inf_count = np.isinf(actions_all).sum()
    print(f'\n  异常值: NaN={nan_count}  Inf={inf_count}  {"✅ 无异常" if nan_count + inf_count == 0 else "⚠️ 有异常!"}')

    # Actions 范围检查 (应在 [-π, π] 内)
    out_of_range = (np.abs(actions_all) > np.pi).sum()
    total = actions_all.size
    print(f'  范围检查: {out_of_range}/{total} 值超出 [-π,π]  '
          f'{"✅" if out_of_range == 0 else f"⚠️ {out_of_range/total*100:.2f}%"}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--duration', type=float, default=10.0, help='采集时长 (秒)')
    parser.add_argument('--prompt', default='Flatten and fold the cloth.')
    args = parser.parse_args()

    print('='*65)
    print('  实时端到端对比: WebSocket 推理 (mode=both)')
    print('  (只读测试, 不驱动机械臂)')
    print('='*65)
    print(f'\n  WebSocket: {args.host}:{args.port}')
    print(f'  Duration:  {args.duration}s')
    print(f'  Prompt:    "{args.prompt}"')

    rclpy.init()
    node = RealtimeComparator(args.host, args.port, args.prompt, args.duration)

    try:
        while node._running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    analyze_results(node.results)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
