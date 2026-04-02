#!/usr/bin/env python3
"""
分别运行 ros2 模式和 websocket 模式, 录制 /policy/actions, 对比一致性。

流程:
  Phase 1: 启动 policy_inference_node (mode=ros2), 录制 10s, 停止
  Phase 2: 启动 serve_policy.py + policy_inference_node (mode=websocket), 录制 10s, 停止
  Phase 3: 对比两组 actions

用法:
  cd /data1/tim/workspace/deepdive_kai0/kai0
  source /opt/ros/jazzy/setup.bash
  PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH" \
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  /data1/tim/workspace/deepdive_kai0/kai0/.venv/bin/python \
    /data1/tim/workspace/deepdive_kai0/scripts/test_dual_mode_sequential.py
"""
import os, sys, time, signal, subprocess
sys.path.insert(0, '/data1/tim/workspace/deepdive_kai0/kai0/src')

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

VENV_PY = '/data1/tim/workspace/deepdive_kai0/kai0/.venv/bin/python'
NODE_SCRIPT = '/data1/tim/workspace/deepdive_kai0/ros2_ws/src/piper/scripts/policy_inference_node.py'
SERVE_SCRIPT = '/data1/tim/workspace/deepdive_kai0/kai0/scripts/serve_policy.py'
CKPT = '/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1'
CONFIG = 'pi05_flatten_fold_normal'
PROMPT = 'Flatten and fold the cloth.'
RECORD_SECS = 10
WARMUP_SECS = 30  # 模型加载 + JIT warmup


def _stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class ActionRecorder(Node):
    """订阅 /policy/actions, 录制指定时长."""
    def __init__(self, duration):
        super().__init__('action_recorder')
        self.duration = duration
        self.actions = []
        self.create_subscription(JointState, '/policy/actions', self._cb, 10)
        self._t0 = None

    def _cb(self, msg):
        if self._t0 is None:
            self._t0 = time.monotonic()
            self.get_logger().info(f'First action received, recording {self.duration}s...')
        elapsed = time.monotonic() - self._t0
        if elapsed > self.duration:
            return
        self.actions.append(np.array(msg.position))

    @property
    def done(self):
        if self._t0 is None:
            return False
        return (time.monotonic() - self._t0) > self.duration


def record_actions(duration):
    """录制 /policy/actions 指定秒数, 返回 [N, 14] array."""
    rclpy.init()
    recorder = ActionRecorder(duration)
    try:
        while not recorder.done and rclpy.ok():
            rclpy.spin_once(recorder, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.destroy_node()
        rclpy.shutdown()
    return np.array(recorder.actions) if recorder.actions else np.array([])


def make_env():
    """构建子进程环境变量."""
    env = os.environ.copy()
    ros_pkgs = '/opt/ros/jazzy/lib/python3.12/site-packages'
    env['PYTHONPATH'] = ros_pkgs + ':' + env.get('PYTHONPATH', '')
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    env['no_proxy'] = 'localhost,127.0.0.1'
    env['NO_PROXY'] = 'localhost,127.0.0.1'
    return env


def run_phase(mode_label, start_fn, duration):
    """启动推理进程, 等待 warmup, 录制, 停止, 返回 actions."""
    print(f'\n{"="*65}')
    print(f'  Phase: {mode_label}')
    print(f'{"="*65}')

    procs = start_fn()
    print(f'  等待模型加载 + warmup ({WARMUP_SECS}s)...')
    time.sleep(WARMUP_SECS)

    print(f'  录制 /policy/actions {duration}s...')
    actions = record_actions(duration)
    print(f'  录制完成: {len(actions)} 步')

    # 停止子进程
    for p in procs:
        p.send_signal(signal.SIGINT)
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    time.sleep(3)  # 等 GPU 释放

    return actions


def start_ros2_mode():
    """启动 policy_inference_node (mode=ros2)."""
    env = make_env()
    p = subprocess.Popen([
        VENV_PY, NODE_SCRIPT, '--ros-args',
        '-p', 'mode:=ros2',
        '-p', f'config_name:={CONFIG}',
        '-p', f'checkpoint_dir:={CKPT}',
        '-p', f'prompt:={PROMPT}',
        '-p', 'img_front_topic:=/camera_f/camera/color/image_raw',
        '-p', 'img_left_topic:=/camera_l/camera/color/image_rect_raw',
        '-p', 'img_right_topic:=/camera_r/camera/color/image_rect_raw',
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f'  node PID={p.pid} (mode=ros2)')
    return [p]


def start_ws_mode():
    """启动 serve_policy.py + policy_inference_node (mode=websocket)."""
    env = make_env()

    # 先启动 serve_policy.py (占 GPU)
    p_serve = subprocess.Popen([
        VENV_PY, SERVE_SCRIPT,
        'policy:checkpoint',
        f'--policy.config={CONFIG}',
        f'--policy.dir={CKPT}',
        '--port=8000',
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f'  serve_policy PID={p_serve.pid}')

    # 等 serve_policy 加载完成
    time.sleep(WARMUP_SECS)

    # 启动 node (mode=websocket, 不占 GPU)
    env_no_gpu = env.copy()
    env_no_gpu['CUDA_VISIBLE_DEVICES'] = ''  # 不需要 GPU
    p_node = subprocess.Popen([
        VENV_PY, NODE_SCRIPT, '--ros-args',
        '-p', 'mode:=websocket',
        '-p', 'host:=localhost',
        '-p', 'port:=8000',
        '-p', f'prompt:={PROMPT}',
        '-p', 'img_front_topic:=/camera_f/camera/color/image_raw',
        '-p', 'img_left_topic:=/camera_l/camera/color/image_rect_raw',
        '-p', 'img_right_topic:=/camera_r/camera/color/image_rect_raw',
    ], env=env_no_gpu, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f'  node PID={p_node.pid} (mode=websocket)')

    return [p_serve, p_node]


def compare(actions_ros2, actions_ws):
    """对比两组 actions."""
    print(f'\n{"="*65}')
    print(f'  对比结果')
    print(f'{"="*65}')
    print(f'  mode=ros2:      {len(actions_ros2)} 步')
    print(f'  mode=websocket: {len(actions_ws)} 步')

    if len(actions_ros2) < 5 or len(actions_ws) < 5:
        print('  数据不足, 跳过')
        return

    # 统计对比 (场景静止, 两组应分布一致)
    print(f'\n  --- 均值对比 ---')
    print(f'  {"dim":>6s}  {"ros2 mean":>11s}  {"ws mean":>11s}  {"diff":>10s}  {"ros2 std":>10s}  {"ws std":>10s}')
    diffs = []
    for d in range(14):
        rm = actions_ros2[:, d].mean()
        wm = actions_ws[:, d].mean()
        rs = actions_ros2[:, d].std()
        ws = actions_ws[:, d].std()
        diff = abs(rm - wm)
        diffs.append(diff)
        label = f'{"L" if d < 7 else "R"}j{d % 7}'
        print(f'  {label:>6s}  {rm:+11.5f}  {wm:+11.5f}  {diff:10.5f}  {rs:10.5f}  {ws:10.5f}')

    max_diff = max(diffs)
    mean_diff = np.mean(diffs)

    print(f'\n  均值差异: max={max_diff:.5f} rad ({np.degrees(max_diff):.2f}°)  mean={mean_diff:.5f} rad')

    # 分布重叠度 (per-dim, 用 range overlap)
    print(f'\n  --- 范围对比 ---')
    print(f'  {"dim":>6s}  {"ros2 [min, max]":>25s}  {"ws [min, max]":>25s}')
    for d in range(14):
        label = f'{"L" if d < 7 else "R"}j{d % 7}'
        r_min, r_max = actions_ros2[:, d].min(), actions_ros2[:, d].max()
        w_min, w_max = actions_ws[:, d].min(), actions_ws[:, d].max()
        print(f'  {label:>6s}  [{r_min:+.4f}, {r_max:+.4f}]  [{w_min:+.4f}, {w_max:+.4f}]')

    # 判定
    # 场景静止 + 同模型 + 同管线: 差异来自 RNG + 帧时间微差
    # 时序平滑后的 actions 是多个 chunk 融合, RNG 差异被平均化
    threshold = 0.02  # 0.02 rad ≈ 1.1°
    passed = max_diff < threshold
    print(f'\n  阈值: {threshold} rad ({np.degrees(threshold):.1f}°)')
    print(f'  结论: {"PASS ✅ 两种模式输出一致" if passed else "FAIL ❌ (查看上方 diff 列定位差异维度)"}')

    # 前 5 步
    n = min(5, len(actions_ros2), len(actions_ws))
    print(f'\n  --- 前 {n} 步 (左臂 7D) ---')
    print(f'  {"step":>4s}  {"mode=ros2":>50s}  {"mode=websocket":>50s}')
    for i in range(n):
        r_s = ' '.join(f'{v:+.4f}' for v in actions_ros2[i, :7])
        w_s = ' '.join(f'{v:+.4f}' for v in actions_ws[i, :7])
        print(f'  {i:4d}  {r_s}  {w_s}')

    return passed


def main():
    print('='*65)
    print('  双模式顺序对比: mode=ros2 vs mode=websocket')
    print('  (场景静止, 只读, 不驱动机械臂)')
    print('='*65)
    print(f'  config:     {CONFIG}')
    print(f'  checkpoint: {CKPT}')
    print(f'  录制时长:    {RECORD_SECS}s × 2')
    print(f'  warmup:     {WARMUP_SECS}s')

    # Phase 1: mode=ros2
    actions_ros2 = run_phase('mode=ros2 (节点内推理)', start_ros2_mode, RECORD_SECS)

    # Phase 2: mode=websocket
    actions_ws = run_phase('mode=websocket (serve_policy + WS)', start_ws_mode, RECORD_SECS)

    # Phase 3: 对比
    compare(actions_ros2, actions_ws)

    # 保存原始数据
    out_dir = '/data1/tim/workspace/deepdive_kai0/scripts'
    np.save(f'{out_dir}/actions_ros2.npy', actions_ros2)
    np.save(f'{out_dir}/actions_ws.npy', actions_ws)
    print(f'\n  原始数据已保存: actions_ros2.npy, actions_ws.npy')


if __name__ == '__main__':
    main()
