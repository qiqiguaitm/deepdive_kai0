#!/usr/bin/env python3
"""
双路径实时对比: 节点内推理 vs WebSocket 推理 (EXACT MATCH 验证)

前提: policy_inference_node.py 以 mode=both 运行中。

流程:
  1. 订阅 /policy/actions — 节点内推理的输出 (path A: 节点内部)
  2. 订阅相机+关节, 构建 obs, 送入 WS — 获取 actions (path B: WebSocket)
  3. 场景静止时, 两条路径的输入几乎相同, 对比输出 actions 的一致性

注意: 两条路径的 RNG 不同 (模型采样随机性), 因此不会 EXACT MATCH。
但场景静止时, 输入相同 → 输出分布应一致。用统计指标量化。
"""
import os, sys, time, threading
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

def build_obs(img_f, img_l, img_r, jl, jr, prompt):
    from openpi_client import image_tools
    imgs = [jpeg_mapping(img_f), jpeg_mapping(img_r), jpeg_mapping(img_l)]
    imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
    imgs = [image_tools.resize_with_pad(im[np.newaxis], 224, 224)[0] for im in imgs]
    qpos = np.concatenate((np.array(jl), np.array(jr)), axis=0)
    return {
        'state': qpos,
        'images': {
            'top_head':   imgs[0].transpose(2, 0, 1),
            'hand_right': imgs[1].transpose(2, 0, 1),
            'hand_left':  imgs[2].transpose(2, 0, 1),
        },
        'prompt': prompt,
    }


class DualPathComparator(Node):
    def __init__(self, ws_host, ws_port, duration):
        super().__init__('dual_path_comparator')
        self.bridge = CvBridge()
        self.duration = duration
        self.prompt = 'Flatten and fold the cloth.'

        # ── Path A: 订阅节点内推理输出 ──
        self.node_actions = []
        self.create_subscription(JointState, '/policy/actions',
            self._cb_node_action, 10)

        # ── Path B: 抓帧 → WS 推理 ──
        self._img_f_deque = deque()
        self._img_l_deque = deque()
        self._img_r_deque = deque()
        self._jl_deque = deque()
        self._jr_deque = deque()

        img_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1000)
        self.create_subscription(Image, '/camera_f/camera/color/image_raw',
            self._cb_img_f, img_qos)
        self.create_subscription(Image, '/camera_l/camera/color/image_rect_raw',
            self._cb_img_l, img_qos)
        self.create_subscription(Image, '/camera_r/camera/color/image_rect_raw',
            self._cb_img_r, img_qos)
        self.create_subscription(JointState, '/puppet/joint_left', self._cb_jl, 1000)
        self.create_subscription(JointState, '/puppet/joint_right', self._cb_jr, 1000)

        from openpi_client import websocket_client_policy
        self.get_logger().info(f'Connecting WS {ws_host}:{ws_port}...')
        self.ws = websocket_client_policy.WebsocketClientPolicy(ws_host, ws_port)
        self.get_logger().info('WS connected')

        self.ws_actions = []
        self._running = True
        self._thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._thread.start()

    # ── Callbacks ──
    def _cb_node_action(self, msg):
        self.node_actions.append({
            't': _stamp_to_sec(msg.header.stamp),
            'pos': np.array(msg.position),
        })

    def _cb_img_f(self, msg):
        if len(self._img_f_deque) >= 100: self._img_f_deque.popleft()
        self._img_f_deque.append(msg)
    def _cb_img_l(self, msg):
        if len(self._img_l_deque) >= 100: self._img_l_deque.popleft()
        self._img_l_deque.append(msg)
    def _cb_img_r(self, msg):
        if len(self._img_r_deque) >= 100: self._img_r_deque.popleft()
        self._img_r_deque.append(msg)
    def _cb_jl(self, msg):
        if len(self._jl_deque) >= 100: self._jl_deque.popleft()
        self._jl_deque.append(msg)
    def _cb_jr(self, msg):
        if len(self._jr_deque) >= 100: self._jr_deque.popleft()
        self._jr_deque.append(msg)

    def _grab_latest(self):
        if any(len(d) == 0 for d in [self._img_f_deque, self._img_l_deque,
                self._img_r_deque, self._jl_deque, self._jr_deque]):
            return None
        img_f = self.bridge.imgmsg_to_cv2(self._img_f_deque[-1], 'passthrough')
        img_l = self.bridge.imgmsg_to_cv2(self._img_l_deque[-1], 'passthrough')
        img_r = self.bridge.imgmsg_to_cv2(self._img_r_deque[-1], 'passthrough')
        jl = list(self._jl_deque[-1].position)
        jr = list(self._jr_deque[-1].position)
        return img_f, img_l, img_r, jl, jr

    def _ws_loop(self):
        self.get_logger().info('Waiting for sensors...')
        while self._running and rclpy.ok():
            if self._grab_latest() is not None:
                break
            time.sleep(0.1)

        self.get_logger().info(f'Running {self.duration}s dual-path comparison...')
        t0 = time.monotonic()
        step = 0
        while self._running and rclpy.ok() and (time.monotonic() - t0) < self.duration:
            frame = self._grab_latest()
            if frame is None:
                time.sleep(0.01)
                continue
            obs = build_obs(*frame, self.prompt)
            try:
                result = self.ws.infer(obs)
                actions = result['actions']  # [chunk_size, 14]
                self.ws_actions.append({
                    't': time.monotonic() - t0,
                    'actions': actions,
                    'first': actions[0].copy(),
                })
                step += 1
                if step % 5 == 0:
                    print(f'  WS step {step}  actions[0,:3]='
                          f'[{actions[0,0]:+.4f} {actions[0,1]:+.4f} {actions[0,2]:+.4f}]')
            except Exception as e:
                self.get_logger().warn(f'WS error: {e}')
                time.sleep(0.1)
            time.sleep(0.05)  # ~20 Hz max to avoid starving node's own inference

        self._running = False


def analyze(node_actions, ws_actions):
    print(f'\n{"="*65}')
    print(f'  双路径对比结果')
    print(f'{"="*65}')
    print(f'  Path A (节点内推理 /policy/actions): {len(node_actions)} 步')
    print(f'  Path B (WebSocket 推理):             {len(ws_actions)} 步')

    if len(node_actions) < 3 or len(ws_actions) < 3:
        print('  数据不足, 跳过分析')
        return

    # 节点内 actions 是 14D position (每步一个 action)
    node_arr = np.array([a['pos'] for a in node_actions])  # [N, 14]
    # WS actions 是 [chunk, 14], 取 first action
    ws_arr = np.array([a['first'] for a in ws_actions])    # [M, 14]

    print(f'\n  --- 统计对比 (场景静止, 两条路径输入几乎相同) ---')
    print(f'  {"维度":>6s}  {"Node mean":>10s}  {"WS mean":>10s}  {"diff":>10s}  {"Node std":>10s}  {"WS std":>10s}')
    diffs = []
    for d in range(min(14, node_arr.shape[1], ws_arr.shape[1])):
        nm = node_arr[:, d].mean()
        wm = ws_arr[:, d].mean()
        ns = node_arr[:, d].std()
        ws = ws_arr[:, d].std()
        diff = abs(nm - wm)
        diffs.append(diff)
        label = f'{"L" if d < 7 else "R"}j{d % 7}'
        print(f'  {label:>6s}  {nm:+10.5f}  {wm:+10.5f}  {diff:10.5f}  {ns:10.5f}  {ws:10.5f}')

    max_diff = max(diffs)
    mean_diff = np.mean(diffs)
    print(f'\n  均值差异: max={max_diff:.5f} rad ({np.degrees(max_diff):.2f}°)  mean={mean_diff:.5f} rad')

    # 判定
    # 场景静止 + 相同模型 + 相同管线: 差异应来自 RNG 采样随机性
    # 典型 RNG 差异 ~0.01-0.05 rad, 管线 bug 差异 ~0.1-0.5 rad
    threshold = 0.05  # 0.05 rad ≈ 3°
    passed = max_diff < threshold
    print(f'  阈值: {threshold} rad ({np.degrees(threshold):.1f}°)')
    print(f'  结论: {"PASS ✅ 两条路径输出一致 (差异仅为 RNG 采样噪声)" if passed else "FAIL ❌ 差异超出 RNG 噪声范围"}')

    # 额外: 展示前 5 步 side-by-side
    n_show = min(5, len(node_actions), len(ws_actions))
    print(f'\n  --- 前 {n_show} 步对比 (左臂 7D) ---')
    print(f'  {"step":>4s}  {"Path A (节点内)":>50s}  {"Path B (WebSocket)":>50s}')
    for i in range(n_show):
        na = node_arr[i, :7]
        wa = ws_arr[i, :7]
        na_s = ' '.join(f'{v:+.4f}' for v in na)
        wa_s = ' '.join(f'{v:+.4f}' for v in wa)
        print(f'  {i:4d}  {na_s}  {wa_s}')

    return passed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--duration', type=float, default=10.0)
    args = parser.parse_args()

    print('='*65)
    print('  双路径实时对比: 节点内推理 vs WebSocket 推理')
    print('  (场景静止, 只读, 不驱动机械臂)')
    print('='*65)

    rclpy.init()
    node = DualPathComparator(args.host, args.port, args.duration)
    try:
        while node._running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    result = analyze(node.node_actions, node.ws_actions)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
