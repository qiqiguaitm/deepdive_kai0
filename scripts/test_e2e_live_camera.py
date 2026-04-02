#!/usr/bin/env python3
"""
端到端真实相机验证: ROS2 节点内推理 vs WebSocket 推理管线

从真实 ROS2 相机 topic 抓取一帧, 分别经过两条管线处理后送入同一个模型推理,
对比 actions 是否完全一致。不发布控制命令, 不会驱动机械臂。

用法 (在 sim01 上, 相机和 piper 节点已启动):
  cd /data1/tim/workspace/deepdive_kai0/kai0
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run python ../scripts/test_e2e_live_camera.py

需要在线的 ROS2 topic:
  /camera_f/camera/color/image_raw      (头顶)
  /camera_l/camera/color/image_rect_raw (左腕)
  /camera_r/camera/color/image_rect_raw (右腕)
  /puppet/joint_left                    (左臂关节)
  /puppet/joint_right                   (右臂关节)
"""

import os
import sys
import time
import threading

sys.path.insert(0, '/data1/tim/workspace/deepdive_kai0/kai0/src')

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState


# ── 图像管线: 严格复刻原版 ─────────────────────────────────────────

def jpeg_mapping(img):
    """原版 JPEG encode/decode."""
    img = cv2.imencode(".jpg", img)[1].tobytes()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    return img


def build_obs_ws_pipeline(img_front, img_left, img_right, joint_left, joint_right, prompt):
    """原版 WebSocket 管线 (agilex_inference_openpi_temporal_smoothing_ros2.py)."""
    from openpi_client import image_tools

    imgs = [jpeg_mapping(img_front), jpeg_mapping(img_right), jpeg_mapping(img_left)]
    imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
    # 逐个 resize (相机分辨率可能不同: D435 640x480, D405 848x480)
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


def build_obs_node_pipeline(img_front, img_left, img_right, joint_left, joint_right, prompt):
    """修复后的 policy_inference_node.py 管线."""
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


def build_obs_old_broken(img_front, img_left, img_right, joint_left, joint_right, prompt):
    """修复前的旧管线 (cv2.resize, 无 JPEG/BGR→RGB)."""
    imgs = [img_front, img_right, img_left]
    imgs = [cv2.resize(im, (224, 224)) for im in imgs]
    state = np.array(list(joint_left) + list(joint_right), dtype=np.float32)
    return {
        'state': state,
        'images': {
            'top_head':   imgs[0].transpose(2, 0, 1),
            'hand_right': imgs[1].transpose(2, 0, 1),
            'hand_left':  imgs[2].transpose(2, 0, 1),
        },
        'prompt': prompt,
    }


# ── ROS2 帧抓取 ───────────────────────────────────────────────────

class FrameGrabber(Node):
    """一次性抓取一帧所有传感器数据."""

    def __init__(self):
        super().__init__('e2e_test_frame_grabber')
        self.bridge = CvBridge()
        self.img_front = None
        self.img_left = None
        self.img_right = None
        self.joint_left = None
        self.joint_right = None

        # 用 BEST_EFFORT 兼容 RealSense 默认 QoS
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(Image,
            '/camera_f/camera/color/image_raw',
            self._cb_front, img_qos)
        self.create_subscription(Image,
            '/camera_l/camera/color/image_rect_raw',
            self._cb_left, img_qos)
        self.create_subscription(Image,
            '/camera_r/camera/color/image_rect_raw',
            self._cb_right, img_qos)
        self.create_subscription(JointState,
            '/puppet/joint_left', self._cb_jl, 10)
        self.create_subscription(JointState,
            '/puppet/joint_right', self._cb_jr, 10)

    def _cb_front(self, msg):
        if self.img_front is None:
            self.img_front = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            self.get_logger().info(f'  ✓ img_front: {self.img_front.shape} dtype={self.img_front.dtype}')

    def _cb_left(self, msg):
        if self.img_left is None:
            self.img_left = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            self.get_logger().info(f'  ✓ img_left:  {self.img_left.shape} dtype={self.img_left.dtype}')

    def _cb_right(self, msg):
        if self.img_right is None:
            self.img_right = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            self.get_logger().info(f'  ✓ img_right: {self.img_right.shape} dtype={self.img_right.dtype}')

    def _cb_jl(self, msg):
        if self.joint_left is None:
            self.joint_left = list(msg.position)
            self.get_logger().info(f'  ✓ joint_left:  {self.joint_left}')

    def _cb_jr(self, msg):
        if self.joint_right is None:
            self.joint_right = list(msg.position)
            self.get_logger().info(f'  ✓ joint_right: {self.joint_right}')

    def all_received(self):
        return all(x is not None for x in [
            self.img_front, self.img_left, self.img_right,
            self.joint_left, self.joint_right])


def grab_one_frame(timeout_sec=10.0):
    """从 ROS2 topic 抓取一帧传感器数据."""
    rclpy.init()
    node = FrameGrabber()
    print('\n  等待传感器数据...')
    t0 = time.monotonic()
    try:
        while not node.all_received():
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - t0 > timeout_sec:
                missing = []
                if node.img_front is None: missing.append('img_front')
                if node.img_left is None: missing.append('img_left')
                if node.img_right is None: missing.append('img_right')
                if node.joint_left is None: missing.append('joint_left')
                if node.joint_right is None: missing.append('joint_right')
                raise TimeoutError(f'超时 {timeout_sec}s, 缺失: {missing}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print(f'  全部传感器数据已接收 ({time.monotonic()-t0:.1f}s)')
    return (node.img_front, node.img_left, node.img_right,
            node.joint_left, node.joint_right)


# ── 主测试 ─────────────────────────────────────────────────────────

def main():
    print('='*65)
    print('  端到端真实相机验证: ROS2 节点内 vs WebSocket 推理')
    print('  (只读测试, 不驱动机械臂)')
    print('='*65)

    # Step 1: 抓取真实帧
    print('\n[Step 1] 从 ROS2 抓取真实相机帧')
    img_front, img_left, img_right, joint_left, joint_right = grab_one_frame()
    prompt = 'Flatten and fold the cloth.'

    print(f'\n  帧信息:')
    print(f'    img_front: {img_front.shape} dtype={img_front.dtype}')
    print(f'    img_left:  {img_left.shape}  dtype={img_left.dtype}')
    print(f'    img_right: {img_right.shape} dtype={img_right.dtype}')
    print(f'    joints:    left={[f"{v:.3f}" for v in joint_left]}')
    print(f'               right={[f"{v:.3f}" for v in joint_right]}')

    # Step 2: 图像管线对比
    print('\n[Step 2] 图像管线对比 (真实相机数据)')
    obs_ws   = build_obs_ws_pipeline(img_front, img_left, img_right, joint_left, joint_right, prompt)
    obs_node = build_obs_node_pipeline(img_front, img_left, img_right, joint_left, joint_right, prompt)
    obs_old  = build_obs_old_broken(img_front, img_left, img_right, joint_left, joint_right, prompt)

    img_match = all(
        np.array_equal(obs_ws['images'][c], obs_node['images'][c])
        for c in ['top_head', 'hand_right', 'hand_left']
    )
    state_match = np.array_equal(obs_ws['state'], obs_node['state'])
    print(f'  修复后 vs WS原版: images={"一致 ✅" if img_match else "不一致 ❌"}  state={"一致 ✅" if state_match else "不一致 ❌"}')

    # 展示修复前差异
    for cam in ['top_head', 'hand_right', 'hand_left']:
        a, b = obs_ws['images'][cam], obs_old['images'][cam]
        diff_pct = np.count_nonzero(a != b) / a.size * 100
        print(f'  修复前 vs WS原版 [{cam:12s}]: {diff_pct:.1f}% 像素不同')

    # Step 3: 加载模型
    print('\n[Step 3] 加载模型')
    checkpoint_dir = '/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1'

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    import jax

    config_name = 'pi05_flatten_fold_normal'
    print(f'  config={config_name}, ckpt={checkpoint_dir}')

    t0 = time.monotonic()
    train_config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)
    print(f'  模型加载完成: {time.monotonic()-t0:.1f}s')

    # Warmup JIT
    print('  JIT warmup...')
    policy.infer(obs_ws)
    print('  warmup done')

    # Step 4: 推理对比 (固定 RNG)
    print('\n[Step 4] 推理对比 (同一 RNG 种子, 真实相机数据)')
    saved_rng = policy._rng

    policy._rng = saved_rng
    t0 = time.monotonic()
    result_ws = policy.infer(obs_ws)
    t_ws = (time.monotonic() - t0) * 1000

    policy._rng = saved_rng
    t0 = time.monotonic()
    result_node = policy.infer(obs_node)
    t_node = (time.monotonic() - t0) * 1000

    actions_ws = result_ws['actions']
    actions_node = result_node['actions']

    exact = np.array_equal(actions_ws, actions_node)
    print(f'  actions shape: {actions_ws.shape}')
    print(f'  latency: WS管线={t_ws:.0f}ms  Node管线={t_node:.0f}ms')

    if exact:
        print(f'  结果: EXACT MATCH ✅')
    else:
        diff = np.abs(actions_ws.astype(float) - actions_node.astype(float))
        close = np.allclose(actions_ws, actions_node, atol=1e-6)
        print(f'  结果: exact={exact}  allclose(1e-6)={close}  max_diff={diff.max():.2e}')

    # 修复前管线对比
    policy._rng = saved_rng
    result_old = policy.infer(obs_old)
    actions_old = result_old['actions']
    diff_old = np.abs(actions_ws.astype(float) - actions_old.astype(float))
    print(f'\n  修复前管线 vs WS管线: max_diff={diff_old.max():.4f}  mean_diff={diff_old.mean():.4f}')

    # Step 5: 展示 actions 样本
    print(f'\n[Step 5] Actions 样本 (前 3 步)')
    print(f'  {"step":>4s}  {"WS管线 (left 7D)":>45s}  {"Node管线 (left 7D)":>45s}')
    for i in range(min(3, len(actions_ws))):
        ws_str = ' '.join(f'{v:+.4f}' for v in actions_ws[i, :7])
        nd_str = ' '.join(f'{v:+.4f}' for v in actions_node[i, :7])
        print(f'  {i:4d}  {ws_str}  {nd_str}')

    # 总结
    print('\n' + '='*65)
    print('  最终结果')
    print('='*65)
    print(f'  真实相机图像管线一致性: {"PASS ✅" if (img_match and state_match) else "FAIL ❌"}')
    print(f'  真实相机推理结果一致性: {"PASS ✅" if exact else "FAIL ❌"}')
    print(f'  修复前管线偏差量级:     max={diff_old.max():.4f} (确认修复必要性)')
    print()


if __name__ == '__main__':
    main()
