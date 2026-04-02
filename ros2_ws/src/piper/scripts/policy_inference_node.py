#!/usr/bin/env python3
"""
ROS2 Policy Inference Node — 将 pi0.5 推理直接集成为 ROS2 节点

三种推理模式 (通过 --mode 参数选择):
  1. "ros2"      — 纯 ROS2 模式: JAX 推理在本节点内完成, 无 WebSocket
  2. "websocket" — 原版模式: 通过 WebSocket 连接外部 serve_policy.py
  3. "both"      — 同时启动: 本节点加载模型, 同时兼容 WebSocket 客户端

用法:
  # 模式 1: 纯 ROS2 (推荐, 最低延迟)
  ros2 run piper policy_inference_node.py --ros-args \
    -p mode:=ros2 \
    -p config_name:=pi05_flatten_fold_normal \
    -p checkpoint_dir:=gs://openpi-assets/checkpoints/pi05_base/params

  # 模式 2: WebSocket 客户端 (兼容旧的 serve_policy.py)
  ros2 run piper policy_inference_node.py --ros-args \
    -p mode:=websocket -p host:=localhost -p port:=8000

  # 模式 3: 两者兼有
  ros2 run piper policy_inference_node.py --ros-args \
    -p mode:=both \
    -p config_name:=pi05_flatten_fold_normal \
    -p checkpoint_dir:=gs://openpi-assets/checkpoints/pi05_base/params \
    -p ws_port:=8000

订阅:
  /camera_f/color/image_raw     (sensor_msgs/Image)     — 头顶相机
  /camera_l/color/image_raw     (sensor_msgs/Image)     — 左腕相机
  /camera_r/color/image_raw     (sensor_msgs/Image)     — 右腕相机
  /puppet/joint_left            (sensor_msgs/JointState) — 左臂关节状态
  /puppet/joint_right           (sensor_msgs/JointState) — 右臂关节状态

发布:
  /policy/actions               (sensor_msgs/JointState) — 推理输出动作 (14 维)
  /master/joint_left            (sensor_msgs/JointState) — 左臂控制命令
  /master/joint_right           (sensor_msgs/JointState) — 右臂控制命令
"""

import os
import sys

# ── 自动 re-exec: 确保在 kai0 venv 中运行 ────────────────────────
# ros2 run 通过 shebang (#!/usr/bin/env python3) 启动, 可能命中 conda 的
# python3.13 或系统 python3.12, 而本节点依赖 venv 中的 JAX/numpy/cv2 等包.
# 检测当前是否在 kai0 venv 中, 如果不是则 re-exec.
# KAI0_ROOT 查找顺序: 环境变量 > 相对路径推导 (source 和 install 两种布局)
_KAI0_ROOT = os.environ.get('KAI0_ROOT', '')
if not _KAI0_ROOT or not os.path.isdir(_KAI0_ROOT):
    # 从 __file__ 位置推导: source 布局 (ros2_ws/src/piper/scripts/ → ../../.. → kai0)
    for levels in [
        ('..', '..', '..', '..', 'kai0'),          # source: ros2_ws/src/piper/scripts/
        ('..', '..', '..', '..', '..', 'kai0'),     # install: ros2_ws/install/piper/lib/piper/
    ]:
        candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), *levels))
        if os.path.isdir(os.path.join(candidate, 'src', 'openpi')):
            _KAI0_ROOT = candidate
            break
    if not _KAI0_ROOT:
        # 最终回退: 硬编码常用路径
        for fallback in ['/data1/tim/workspace/deepdive_kai0/kai0',
                         os.path.expanduser('~/workspace/deepdive_kai0/kai0')]:
            if os.path.isdir(os.path.join(fallback, 'src', 'openpi')):
                _KAI0_ROOT = fallback
                break
_VENV_PYTHON = os.path.join(_KAI0_ROOT, '.venv', 'bin', 'python')
_VENV_PREFIX = os.path.join(_KAI0_ROOT, '.venv')

if (os.path.isfile(_VENV_PYTHON)
        and os.path.abspath(sys.prefix) != os.path.abspath(_VENV_PREFIX)):
    # 当前不在 kai0 venv 中 (可能是 conda python3.13 或裸系统 python3.12)
    # 清理 PATH 中的 conda 路径, 防止 conda 的 libpython/importlib 污染 re-exec 后的进程
    _clean_path = ':'.join(p for p in os.environ.get('PATH', '').split(':')
                           if 'conda' not in p.lower())
    os.environ['PATH'] = _clean_path
    # 确保 LD_LIBRARY_PATH 也不含 conda
    _clean_ld = ':'.join(p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':')
                         if 'conda' not in p.lower())
    os.environ['LD_LIBRARY_PATH'] = _clean_ld
    # 用 venv python 重新启动自己, 保留所有命令行参数
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import time
import threading
from collections import deque

# 确保 openpi src 可被 import
_KAI0_SRC = os.path.join(_KAI0_ROOT, 'src')
if os.path.isdir(_KAI0_SRC) and _KAI0_SRC not in sys.path:
    sys.path.insert(0, _KAI0_SRC)

# 确保 CUDA 库路径在 JAX import 前设好
import glob as _glob
_venv_nvidia = os.path.join(_KAI0_ROOT, '.venv', 'lib', 'python3.12', 'site-packages', 'nvidia')
_nvidia_libs = ':'.join(sorted(_glob.glob(os.path.join(_venv_nvidia, '*', 'lib'))))
if _nvidia_libs:
    os.environ['LD_LIBRARY_PATH'] = _nvidia_libs + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    import ctypes
    try:
        for lib_dir in _nvidia_libs.split(':'):
            for so in sorted(_glob.glob(os.path.join(lib_dir, '*.so*'))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    except Exception:
        pass

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header


def _stamp_to_sec(stamp):
    """Convert a ROS2 stamp (sec + nanosec) to float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


# ────────────────────────────────────────────────────────────────────────────
# StreamActionBuffer — 从原版 agilex_inference_openpi_temporal_smoothing_ros2.py
#                      逐行复制, 不做任何修改
# ────────────────────────────────────────────────────────────────────────────
class StreamActionBuffer:
    """
    Maintains a queue of action chunks; each chunk is a deque([action0, action1, ...]).
    - New inferred chunks are appended from the right;
    - For each published step, popleft() the leftmost action from each chunk;
    - Empty chunks are dropped.
    """
    def __init__(self, max_chunks=10, decay_alpha=0.25, state_dim=14, smooth_method="temporal"):
        self.chunks = deque()                 # Kept for backward compatibility
        self.max_chunks = max_chunks
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)  # Smoothing strength (exponential weight)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        self.cur_chunk = deque()              # Current sequence to publish (after smoothing)
        self.k = 0                            # Published step count (for latency trimming)
        self.last_action = None               # Last successfully popped action

    def push_chunk(self, actions_chunk: np.ndarray):
        """Legacy interface (no longer used)."""
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            dq = deque([a.copy() for a in actions_chunk], maxlen=None)
            self.chunks.append(dq)
            while len(self.chunks) > self.max_chunks:
                self.chunks.popleft()

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8):
        """
        Integrate a new inference chunk:
        1) Trim the front of the new chunk by current k and max_k (latency compensation).
        2) If there is an existing chunk (cur_chunk), apply temporal smoothing on the overlap:
           - Overlap: first element 100% old / 0% new, last element 0% old / 100% new.
           - Extra tail from the new chunk is appended.
        3) Reset k=0 as the new current execution sequence.
        """
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                # Entire chunk trimmed; skip this update
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]
            # Build old sequence: if empty but last_action exists, extend with last_action to min_m steps;
            # if non-empty and len < m, pad tail to min_m; if both empty, take new sequence as-is
            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return
            new_list = list(new_chunk)

            # Overlap length = min of remaining old length and new length
            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                # No overlap; use new sequence as-is
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return

            # If old sequence is longer than new, trim old tail
            if len(old_list) > len(new_list):
                old_list = old_list[:len(new_list)]
                overlap_len = len(new_list)

            # Linear weights: first element 100% old, last element 0% old
            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old

            # Smooth the overlap region
            smoothed = [
                (w_old[i] * np.asarray(old_list[i], dtype=float) +
                 w_new[i] * np.asarray(new_list[i], dtype=float))
                for i in range(overlap_len)
            ]
            # Append the extra tail from the new sequence
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
            self.k = 0

    def has_any(self):
        with self.lock:
            return len(self.cur_chunk) > 0

    def pop_next_action(self) -> np.ndarray | None:
        """Pop and return the next action to publish; k += 1."""
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            # If about to pop the last element, save it as last_action
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act


# ────────────────────────────────────────────────────────────────────────────
# PolicyInferenceNode
# ────────────────────────────────────────────────────────────────────────────
class PolicyInferenceNode(Node):
    """ROS2 node that integrates policy inference directly."""

    def __init__(self):
        super().__init__('policy_inference_node')

        # ── Parameters ──
        self.declare_parameter('mode', 'ros2')  # ros2 | websocket | both
        self.declare_parameter('config_name', 'pi05_flatten_fold_normal')
        self.declare_parameter('checkpoint_dir', '')
        self.declare_parameter('host', 'localhost')
        self.declare_parameter('port', 8000)
        self.declare_parameter('ws_port', 8000)
        self.declare_parameter('prompt', 'Flatten and fold the cloth.')
        self.declare_parameter('publish_rate', 30)
        self.declare_parameter('inference_rate', 3.0)
        self.declare_parameter('chunk_size', 50)
        self.declare_parameter('latency_k', 8)
        self.declare_parameter('min_smooth_steps', 8)
        self.declare_parameter('decay_alpha', 0.25)
        self.declare_parameter('gripper_offset', 0.003)
        self.declare_parameter('img_front_topic', '/camera_f/color/image_raw')
        self.declare_parameter('img_left_topic', '/camera_l/color/image_raw')
        self.declare_parameter('img_right_topic', '/camera_r/color/image_raw')
        self.declare_parameter('puppet_left_topic', '/puppet/joint_left')
        self.declare_parameter('puppet_right_topic', '/puppet/joint_right')
        self.declare_parameter('gpu_id', 0)

        self.mode = self.get_parameter('mode').value
        self.prompt = self.get_parameter('prompt').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.inference_rate = self.get_parameter('inference_rate').value
        self.chunk_size = self.get_parameter('chunk_size').value
        self.latency_k = self.get_parameter('latency_k').value
        self.min_smooth_steps = self.get_parameter('min_smooth_steps').value
        self.decay_alpha = self.get_parameter('decay_alpha').value
        self.gripper_offset = self.get_parameter('gripper_offset').value

        self.get_logger().info(f'Mode: {self.mode}')

        # ── State ──
        self.bridge = CvBridge()
        self.policy = None
        self.stream_buffer = StreamActionBuffer(
            decay_alpha=self.decay_alpha, state_dim=14)

        # ── Sensor deques (原版帧同步模式: 回调 append, get_synced_frame 消费) ──
        self._img_front_deque = deque()
        self._img_left_deque = deque()
        self._img_right_deque = deque()
        self._joint_left_deque = deque()
        self._joint_right_deque = deque()

        # ── Subscribers ──
        # 图像: BEST_EFFORT 匹配 RealSense 默认 QoS, depth=1000 用于帧同步 deque
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1000)
        self.create_subscription(Image,
            self.get_parameter('img_front_topic').value,
            self._cb_img_front, img_qos)
        self.create_subscription(Image,
            self.get_parameter('img_left_topic').value,
            self._cb_img_left, img_qos)
        self.create_subscription(Image,
            self.get_parameter('img_right_topic').value,
            self._cb_img_right, img_qos)
        self.create_subscription(JointState,
            self.get_parameter('puppet_left_topic').value,
            self._cb_joint_left, 1000)
        self.create_subscription(JointState,
            self.get_parameter('puppet_right_topic').value,
            self._cb_joint_right, 1000)

        # ── Publishers ──
        self.pub_action = self.create_publisher(JointState, '/policy/actions', 10)
        self.pub_left = self.create_publisher(JointState, '/master/joint_left', 10)
        self.pub_right = self.create_publisher(JointState, '/master/joint_right', 10)

        # ── Load policy ──
        self._load_policy()

        # ── Inference thread ──
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_thread.start()

        # ── Publish timer ──
        period = 1.0 / self.publish_rate
        self.create_timer(period, self._publish_action)

        self.get_logger().info('Policy inference node ready')

    # ── Policy loading ──────────────────────────────────────────────

    def _load_policy(self):
        """Load policy based on mode."""
        if self.mode in ('ros2', 'both'):
            self._load_jax_policy()
            if self.mode == 'both':
                self._start_ws_server()
        elif self.mode == 'websocket':
            self._load_ws_policy()

    def _load_jax_policy(self):
        """Load JAX model directly (no WebSocket)."""
        config_name = self.get_parameter('config_name').value
        checkpoint_dir = self.get_parameter('checkpoint_dir').value
        gpu_id = str(self.get_parameter('gpu_id').value)

        if not checkpoint_dir:
            raise ValueError(
                'checkpoint_dir is required for ros2/both mode. '
                'Set via: -p checkpoint_dir:=gs://openpi-assets/checkpoints/pi05_base/params')

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '/tmp/xla_cache')
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['JAX_COMPILATION_CACHE_DIR'] = cache_dir

        # 确保 CUDA 库路径
        venv_nvidia = os.path.join(_KAI0_ROOT, '.venv', 'lib', 'python3.12', 'site-packages', 'nvidia')
        if os.path.exists(venv_nvidia):
            cuda_libs = ':'.join(
                os.path.join(venv_nvidia, d, 'lib')
                for d in os.listdir(venv_nvidia)
                if os.path.isdir(os.path.join(venv_nvidia, d, 'lib'))
            )
            os.environ['LD_LIBRARY_PATH'] = cuda_libs + ':' + os.environ.get('LD_LIBRARY_PATH', '')

        self.get_logger().info(f'Loading JAX policy: config={config_name}, ckpt={checkpoint_dir}, GPU={gpu_id}')
        t0 = time.monotonic()

        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        train_config = _config.get_config(config_name)
        self.policy = _policy_config.create_trained_policy(
            train_config, checkpoint_dir)

        self.get_logger().info(f'JAX policy loaded in {time.monotonic()-t0:.1f}s')

    def _load_ws_policy(self):
        """Connect to external serve_policy.py via WebSocket."""
        host = self.get_parameter('host').value
        port = self.get_parameter('port').value
        self.get_logger().info(f'Connecting to WebSocket policy at {host}:{port}')

        from openpi_client import websocket_client_policy
        self.policy = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.get_logger().info('WebSocket policy connected')

    def _start_ws_server(self):
        """In 'both' mode, also serve the loaded policy via WebSocket."""
        ws_port = self.get_parameter('ws_port').value
        self.get_logger().info(f'Starting WebSocket server on :{ws_port}')

        from openpi.serving import websocket_policy_server
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=self.policy, host='0.0.0.0', port=ws_port,
            metadata=getattr(self.policy, 'metadata', {}))
        ws_thread = threading.Thread(target=server.serve_forever, daemon=True)
        ws_thread.start()
        self.get_logger().info(f'WebSocket server running on :{ws_port}')

    # ── Sensor callbacks (原版 deque 模式, 容量 2000) ──────────────

    def _cb_img_front(self, msg):
        if len(self._img_front_deque) >= 2000:
            self._img_front_deque.popleft()
        self._img_front_deque.append(msg)

    def _cb_img_left(self, msg):
        if len(self._img_left_deque) >= 2000:
            self._img_left_deque.popleft()
        self._img_left_deque.append(msg)

    def _cb_img_right(self, msg):
        if len(self._img_right_deque) >= 2000:
            self._img_right_deque.popleft()
        self._img_right_deque.append(msg)

    def _cb_joint_left(self, msg):
        if len(self._joint_left_deque) >= 2000:
            self._joint_left_deque.popleft()
        self._joint_left_deque.append(msg)

    def _cb_joint_right(self, msg):
        if len(self._joint_right_deque) >= 2000:
            self._joint_right_deque.popleft()
        self._joint_right_deque.append(msg)

    # ── Frame sync (复刻原版 get_frame, 基于 min(timestamp) 对齐) ──

    def _get_synced_frame(self):
        """Return timestamp-aligned (img_front, img_left, img_right, joint_left, joint_right)
        or None if any sensor data is missing/stale."""
        if (len(self._img_front_deque) == 0
                or len(self._img_left_deque) == 0
                or len(self._img_right_deque) == 0
                or len(self._joint_left_deque) == 0
                or len(self._joint_right_deque) == 0):
            return None

        # Sync time = min of latest timestamps across 3 cameras
        frame_time = min(
            _stamp_to_sec(self._img_front_deque[-1].header.stamp),
            _stamp_to_sec(self._img_left_deque[-1].header.stamp),
            _stamp_to_sec(self._img_right_deque[-1].header.stamp),
        )

        # Check all sensors have data at or after frame_time
        for dq, name in [
            (self._img_front_deque, 'img_front'),
            (self._img_left_deque, 'img_left'),
            (self._img_right_deque, 'img_right'),
            (self._joint_left_deque, 'joint_left'),
            (self._joint_right_deque, 'joint_right'),
        ]:
            if len(dq) == 0 or _stamp_to_sec(dq[-1].header.stamp) < frame_time:
                return None

        # Pop frames up to frame_time (discard stale data)
        def _pop_synced(dq):
            while _stamp_to_sec(dq[0].header.stamp) < frame_time:
                dq.popleft()
            return dq.popleft()

        img_front_msg = _pop_synced(self._img_front_deque)
        img_left_msg = _pop_synced(self._img_left_deque)
        img_right_msg = _pop_synced(self._img_right_deque)
        joint_left_msg = _pop_synced(self._joint_left_deque)
        joint_right_msg = _pop_synced(self._joint_right_deque)

        # 强制输出 BGR: RealSense ROS2 默认 rgb8, cv_bridge 会自动转换
        # 后续管线 jpeg_mapping + COLOR_BGR2RGB 假设输入为 BGR
        img_front = self.bridge.imgmsg_to_cv2(img_front_msg, 'bgr8')
        img_left = self.bridge.imgmsg_to_cv2(img_left_msg, 'bgr8')
        img_right = self.bridge.imgmsg_to_cv2(img_right_msg, 'bgr8')

        return img_front, img_left, img_right, joint_left_msg, joint_right_msg

    # ── Image preprocessing (严格复刻原版管线) ──────────────────────

    @staticmethod
    def _jpeg_mapping(img):
        """JPEG encode/decode 对齐训练数据的 MP4 视频压缩 artifacts.

        原版链路: passthrough → imencode (按 BGR 编码) → imdecode (固定输出 BGR)
        无论输入是 rgb8 还是 bgr8, 输出均为 BGR (OpenCV 约定).
        """
        img = cv2.imencode(".jpg", img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    def _get_observation(self):
        """Pack current sensor data into policy input.

        严格复刻原版 update_observation_window + inference_fn 的图像管线:
        1. get_synced_frame (timestamp 对齐)
        2. JPEG encode/decode (训练对齐)
        3. BGR → RGB
        4. resize_with_pad(224, 224) — 保持宽高比, 零填充
        5. HWC → CHW
        """
        frame = self._get_synced_frame()
        if frame is None:
            return None

        img_front, img_left, img_right, joint_left_msg, joint_right_msg = frame

        from openpi_client import image_tools

        # 原版顺序: front, right, left (camera_names = [front, right, left])
        imgs = [
            self._jpeg_mapping(img_front),
            self._jpeg_mapping(img_right),
            self._jpeg_mapping(img_left),
        ]
        imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
        # 与原版一致: 相同分辨率时走 batch resize, 否则逐张
        if imgs[0].shape == imgs[1].shape == imgs[2].shape:
            imgs = list(image_tools.resize_with_pad(np.array(imgs), 224, 224))
        else:
            imgs = [image_tools.resize_with_pad(im[np.newaxis], 224, 224)[0] for im in imgs]

        qpos = np.concatenate((
            np.array(joint_left_msg.position),
            np.array(joint_right_msg.position),
        ), axis=0)

        return {
            'state': qpos,
            'images': {
                'top_head':   imgs[0].transpose(2, 0, 1),   # CHW
                'hand_right': imgs[1].transpose(2, 0, 1),
                'hand_left':  imgs[2].transpose(2, 0, 1),
            },
            'prompt': self.prompt,
        }

    # ── Inference loop ──────────────────────────────────────────────

    def _inference_loop(self):
        """Background thread: continuously infer and push to buffer."""
        # Wait for policy to be loaded
        while self.policy is None and rclpy.ok():
            time.sleep(0.1)

        self.get_logger().info('Inference loop started')
        rate_hz = self.inference_rate
        period = 1.0 / rate_hz

        # Warmup
        self.get_logger().info('Waiting for sensor data...')
        while rclpy.ok():
            obs = self._get_observation()
            if obs is not None:
                break
            time.sleep(0.1)

        self.get_logger().info('Running warmup inference...')
        t0 = time.monotonic()
        try:
            self.policy.infer(obs)
        except Exception as e:
            self.get_logger().warn(f'Warmup failed: {e}')
        self.get_logger().info(f'Warmup done in {(time.monotonic()-t0)*1000:.0f}ms')

        # Main loop
        while rclpy.ok():
            t_start = time.monotonic()
            try:
                obs = self._get_observation()
                if obs is None:
                    time.sleep(0.01)
                    continue

                result = self.policy.infer(obs)
                actions = result.get('actions', None)

                if actions is not None and len(actions) > 0:
                    self.stream_buffer.integrate_new_chunk(
                        actions,
                        max_k=self.latency_k,
                        min_m=self.min_smooth_steps)

                elapsed = time.monotonic() - t_start
                sleep_time = max(0, period - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                import traceback
                self.get_logger().error(f'Inference error: {e}\n{traceback.format_exc()}')
                time.sleep(1.0)

    # ── Action publishing ───────────────────────────────────────────

    def _publish_action(self):
        """Timer callback: pop smoothed action and publish."""
        act = self.stream_buffer.pop_next_action()
        if act is None:
            return

        left = act[:7].copy()
        right = act[7:14].copy()
        left[6] = max(0.0, left[6] - self.gripper_offset)
        right[6] = max(0.0, right[6] - self.gripper_offset)

        now = self.get_clock().now().to_msg()

        # Publish to /master/joint_left and /master/joint_right
        for pub, values in [(self.pub_left, left), (self.pub_right, right)]:
            msg = JointState()
            msg.header.stamp = now
            msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            msg.position = values.tolist()
            pub.publish(msg)

        # Also publish combined action on /policy/actions
        msg = JointState()
        msg.header.stamp = now
        msg.name = [f'left_j{i}' for i in range(7)] + [f'right_j{i}' for i in range(7)]
        msg.position = act.tolist()
        self.pub_action.publish(msg)


def main():
    rclpy.init()
    node = PolicyInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
