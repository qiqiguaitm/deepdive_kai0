#!/usr/bin/env python3
"""
手眼标定数据采集 — 两阶段: Preview → Replay

硬件直连 (不依赖 ROS2):
  - pyrealsense2 直连 RealSense 相机
  - Piper SDK 直连 CAN 总线读写关节角

用法:
  # 阶段 1: Preview — 手动遥控, 确认姿态
  python3 calib/capture_handeye.py --phase preview --arm left \
    --can can3 --camera-serial 409122273074 --session left_calib

  # 阶段 2: Replay — 自动运动到已确认姿态, 采集数据
  python3 calib/capture_handeye.py --phase replay --arm left --session left_calib

  # D435 头顶标定 — 单帧采集
  python3 calib/capture_handeye.py --phase head --camera-serial 254622070889 --session head_calib
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, os.path.dirname(__file__))
from board_def import detect_charuco, get_board, N_CORNERS, N_MARKERS
from piper_fk import PiperFK

sys.path.insert(0, '/home/tim/workspace/piper_sdk')
from piper_sdk import C_PiperInterface

# ── 硬件配置 (从 config/ 自动读取, CLI 参数可覆盖) ─────────────────────────────
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')

def _load_hardware_config() -> tuple[dict, dict]:
    """从 config/cameras.yml 和 config/pipers.yml 读取硬件配置。"""
    camera_serials = {'left': '', 'right': '', 'head': ''}
    can_ports = {'left': '', 'right': ''}
    try:
        import yaml
        cam_path = os.path.join(CONFIG_DIR, 'cameras.yml')
        if os.path.exists(cam_path):
            with open(cam_path) as f:
                cam_cfg = yaml.safe_load(f)
            cams = cam_cfg.get('cameras', {})
            camera_serials['head'] = cams.get('top_head', {}).get('serial_number', '')
            camera_serials['left'] = cams.get('hand_left', {}).get('serial_number', '')
            camera_serials['right'] = cams.get('hand_right', {}).get('serial_number', '')

        piper_path = os.path.join(CONFIG_DIR, 'pipers.yml')
        if os.path.exists(piper_path):
            with open(piper_path) as f:
                piper_cfg = yaml.safe_load(f)
            arms = piper_cfg.get('arms', {})
            can_ports['left'] = arms.get('left_slave', {}).get('can_symbolic', '') or \
                                arms.get('left_slave', {}).get('can_physical', '')
            can_ports['right'] = arms.get('right_slave', {}).get('can_symbolic', '') or \
                                 arms.get('right_slave', {}).get('can_physical', '')
    except Exception as e:
        print(f"[WARN] Failed to load config: {e}, using hardcoded defaults")
        camera_serials = {'left': '409122273074', 'right': '409122271568', 'head': '254622070889'}
        can_ports = {'left': 'can_left_slave', 'right': 'can_right_slave'}
    return camera_serials, can_ports

CAMERA_SERIALS, CAN_PORTS = _load_hardware_config()
JOINT_FACTOR = 57295.7795   # rad → 0.001°

# ── 可用性检查阈值 ────────────────────────────────────────────────────────────
MIN_CORNERS = 8             # 最少角点数 (24 的 1/3)
MAX_REPROJ_ERR = 1.0        # 最大重投影误差 (px)
MIN_BOARD_RATIO = 0.05      # 最小板面积占比 (手腕 D405, 近距离时仅部分可见)
MIN_BOARD_RATIO_HEAD = 0.02 # 最小板面积占比 (头顶 D435 @80cm, 板面仅占 ~4%)
MAX_BOARD_RATIO = 0.80      # 最大板面积占比
MIN_SHARPNESS = 50.0        # 最小清晰度 (Laplacian variance)
MAX_MOTION_PX = 2.0         # 最大运动模糊 (连续帧角点位移 px)

# ── 采集参数 ──────────────────────────────────────────────────────────────────
REPLAY_SETTLE_THRESHOLD_DEG = 0.5   # 到位判断阈值 (°)
REPLAY_SETTLE_WAIT_S = 0.5          # 到位后额外等待 (s)
REPLAY_SETTLE_TIMEOUT_S = 20.0      # 到位超时 (s)
REPLAY_AVG_FRAMES = 5               # 采集帧数 (取平均)
REPLAY_SPEED_PCT = 30               # 运动速度 (%)
REPLAY_DETECT_RETRIES = 3           # 检测失败重试次数

# 安全中间位姿: 零位 (go_zero), 手臂竖直伸起远离桌面
SAFE_HOME_JOINTS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Piper 关节极限 (rad) — official spec + 5° practical padding
# (datasheet bounds are approximate; real hardware allows slightly beyond)
_PAD = np.radians(5.0)
JOINT_LIMITS_RAD = np.array([
    [-2.688, 2.688],    # J1: ±154°
    [ 0.000, 3.403],    # J2: 0° ~ 195°
    [-3.054, 0.000],    # J3: -175° ~ 0°
    [-1.850, 1.850],    # J4: ±106°
    [-1.309, 1.309],    # J5: ±75°
    [-1.745, 1.745],    # J6: ±100°
])
JOINT_LIMITS_RAD[:, 0] -= _PAD
JOINT_LIMITS_RAD[:, 1] += _PAD


# ══════════════════════════════════════════════════════════════════════════════
# 相机工具
# ══════════════════════════════════════════════════════════════════════════════

class RealsenseCamera:
    """pyrealsense2 直连封装"""

    def __init__(self, serial: str, width=640, height=480, fps=30):
        self.serial = serial
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        # 等自动曝光稳定
        for _ in range(30):
            self.pipeline.wait_for_frames(timeout_ms=3000)
        # 读 color 内参
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1],
        ], dtype=np.float64)
        self.dist_coeffs = np.array(intr.coeffs, dtype=np.float64)
        self.intrinsics_dict = {
            'fx': intr.fx, 'fy': intr.fy,
            'cx': intr.ppx, 'cy': intr.ppy,
            'dist': list(intr.coeffs),
            'width': intr.width, 'height': intr.height,
        }
        # 读 depth 内参 (点云投影用, 与 color 内参不同)
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        dintr = depth_stream.get_intrinsics()
        self.depth_intrinsics_dict = {
            'fx': dintr.fx, 'fy': dintr.fy,
            'cx': dintr.ppx, 'cy': dintr.ppy,
            'dist': list(dintr.coeffs),
            'width': dintr.width, 'height': dintr.height,
        }
        # 读 depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # e.g. 0.001 for mm

    def grab(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (bgr_image, depth_image)"""
        frames = self.pipeline.wait_for_frames(timeout_ms=3000)
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        return color, depth

    def grab_avg(self, n: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """采集 n 帧 RGB 取平均 (消除噪声), depth 取最后一帧"""
        acc = None
        depth = None
        for _ in range(n):
            bgr, depth = self.grab()
            acc = bgr.astype(np.float64) if acc is None else acc + bgr.astype(np.float64)
        return (acc / n).astype(np.uint8), depth

    def stop(self):
        self.pipeline.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Piper 机械臂工具
# ══════════════════════════════════════════════════════════════════════════════

class PiperArm:
    """Piper SDK 直连封装"""

    def __init__(self, can_name: str):
        self.piper = C_PiperInterface(can_name)
        self.piper.ConnectPort()
        self.fk = PiperFK()

    def read_joints_rad(self) -> np.ndarray:
        """读取当前 6 个关节角 (rad)"""
        msg = self.piper.GetArmJointMsgs()
        js = msg.joint_state
        mdeg = np.array([js.joint_1, js.joint_2, js.joint_3,
                         js.joint_4, js.joint_5, js.joint_6], dtype=np.float64)
        return mdeg / JOINT_FACTOR  # 0.001° → rad

    def read_fk(self) -> np.ndarray:
        """读取当前 FK 4×4"""
        return self.fk.fk_homogeneous(self.read_joints_rad())

    def move_to(self, q_rad: np.ndarray | list, speed_pct: int = 30):
        """发送关节角指令 (rad)。EnablePiper 失败时抛出异常。"""
        # Enable
        enabled = False
        for _ in range(100):
            if self.piper.EnablePiper():
                enabled = True
                break
            time.sleep(0.01)
        if not enabled:
            raise RuntimeError("EnablePiper() failed after 100 attempts — check CAN bus and power")
        # Motion mode: CAN control, MOVE J
        self.piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)
        time.sleep(0.05)
        ctrl = [int(q * JOINT_FACTOR) for q in q_rad]
        self.piper.JointCtrl(*ctrl)

    def wait_settled(self, target_rad: np.ndarray, threshold_deg=0.5,
                     timeout_s=5.0, extra_wait_s=0.5,
                     cam: 'RealsenseCamera | None' = None,
                     status_text: str = '',
                     speed_pct: int = None) -> bool:
        """等待机械臂到达目标位置, 可选实时刷新相机画面。

        Features:
            - Re-sends move command every 2s if still far from target
            - Checks both position error AND velocity (stability)
            - Shows live error in camera preview

        Args:
            cam: 如果提供, 等待期间持续显示相机画面
            status_text: 叠加在画面上的状态文字
            speed_pct: 运动速度, 用于重发命令
        """
        t0 = time.monotonic()
        last_resend = t0
        prev_joints = None
        stable_count = 0

        while time.monotonic() - t0 < timeout_s:
            current = self.read_joints_rad()
            err_deg = np.abs(np.degrees(current - np.array(target_rad)))
            max_err = np.max(err_deg)

            # Check velocity (position change between reads)
            velocity_ok = True
            if prev_joints is not None:
                vel_deg = np.abs(np.degrees(current - prev_joints)) / 0.05  # deg/s
                velocity_ok = np.max(vel_deg) < 2.0  # < 2 deg/s = settled
            prev_joints = current.copy()

            if cam is not None:
                bgr, _ = cam.grab()
                _show_replay_status(bgr, status_text, f"err={max_err:.1f}° vel={'ok' if velocity_ok else 'moving'}")

            if np.all(err_deg < threshold_deg) and velocity_ok:
                stable_count += 1
                if stable_count >= 3:  # stable for 3 consecutive reads
                    time.sleep(extra_wait_s)
                    return True
            else:
                stable_count = 0

            # Re-send command every 2s if still far from target
            if speed_pct is not None and time.monotonic() - last_resend > 2.0 and max_err > threshold_deg:
                self.move_to(target_rad, speed_pct=speed_pct)
                last_resend = time.monotonic()

            time.sleep(0.05)
        return False

    def disable(self):
        """禁用电机 (被动模式, 可手动拖动)"""
        self.piper.DisablePiper()


# ══════════════════════════════════════════════════════════════════════════════
# 可用性检查
# ══════════════════════════════════════════════════════════════════════════════

def check_quality(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    prev_corners: np.ndarray | None = None,
    prev_ids: np.ndarray | None = None,
    is_head: bool = False,
) -> dict:
    """检查当前帧的标定可用性。

    Args:
        is_head: True 时使用头顶相机的宽松面积阈值 (D435 @80cm 板面仅占 ~4%)

    Returns:
        dict with keys: corners, ids, rvec, tvec, n_corners, n_markers,
        reproj_err, board_ratio, sharpness, motion_px, checks (dict of bool), usable (bool)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]

    corners, ids, rvec, tvec, reproj_err, n_markers = detect_charuco(
        image, camera_matrix, dist_coeffs
    )

    n_corners = len(corners) if corners is not None else 0

    # 板面积占比
    board_ratio = 0.0
    if corners is not None and len(corners) >= 4:
        pts = corners.reshape(-1, 2)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        board_ratio = (x1 - x0) * (y1 - y0) / (w * h)

    # 图像清晰度
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 运动模糊 (按 ID 匹配角点, 避免顺序不同导致误判)
    motion_px = 0.0
    if prev_corners is not None and prev_ids is not None and corners is not None and ids is not None:
        prev_ids_flat = prev_ids.flatten()
        curr_ids_flat = ids.flatten()
        prev_map = {int(prev_ids_flat[i]): i for i in range(len(prev_ids_flat))}
        curr_map = {int(curr_ids_flat[i]): i for i in range(len(curr_ids_flat))}
        common = set(prev_map.keys()) & set(curr_map.keys())
        if len(common) >= 4:
            prev_pts = np.array([prev_corners[prev_map[cid]].flatten() for cid in common])
            curr_pts = np.array([corners[curr_map[cid]].flatten() for cid in common])
            motion_px = np.sqrt(np.mean((prev_pts - curr_pts) ** 2))

    checks = {
        'corners': n_corners >= MIN_CORNERS,
        'reproj_err': reproj_err is not None and reproj_err < MAX_REPROJ_ERR,
        'board_ratio': (MIN_BOARD_RATIO_HEAD if is_head else MIN_BOARD_RATIO) <= board_ratio <= MAX_BOARD_RATIO,
        'sharpness': sharpness > MIN_SHARPNESS,
        'motion': motion_px < MAX_MOTION_PX,
    }
    usable = all(checks.values())

    return {
        'corners': corners, 'ids': ids, 'rvec': rvec, 'tvec': tvec,
        'n_corners': n_corners, 'n_markers': n_markers,
        'reproj_err': reproj_err, 'board_ratio': board_ratio,
        'sharpness': sharpness, 'motion_px': motion_px,
        'checks': checks, 'usable': usable,
        'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs,
    }


def draw_quality_overlay(image: np.ndarray, quality: dict, pose_idx: int, total: int) -> np.ndarray:
    """在图像上叠加检测结果和可用性面板"""
    vis = image.copy()
    h, w = vis.shape[:2]

    # 绘制 ChArUco 检测
    if quality['corners'] is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, quality['corners'], quality['ids'])
    if quality['rvec'] is not None and quality.get('camera_matrix') is not None:
        cv2.drawFrameAxes(vis, quality['camera_matrix'], quality['dist_coeffs'],
                          quality['rvec'], quality['tvec'], 0.05)

    # 右侧面板
    panel_x = w + 10
    panel_w = 300
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = vis
    canvas[:, w:w+2] = 128  # 分隔线

    y = 30
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 标题
    cv2.putText(canvas, f"Pose {pose_idx+1}/{total}", (panel_x, y), font, 0.7, (255, 255, 255), 2)
    y += 40

    # 检查项
    items = [
        ('corners', f"Corners: {quality['n_corners']}/{N_CORNERS}"),
        (None, f"Markers: {quality['n_markers']}/{N_MARKERS}"),
        ('reproj_err', f"Reproj err: {quality['reproj_err']:.2f}px" if quality['reproj_err'] else "Reproj err: N/A"),
        ('board_ratio', f"Board area: {quality['board_ratio']:.0%}"),
        ('sharpness', f"Sharpness: {quality['sharpness']:.0f}"),
        ('motion', f"Motion: {quality['motion_px']:.1f}px"),
    ]
    for key, text in items:
        if key is None:
            # Info-only line (no pass/fail check)
            cv2.putText(canvas, f"     {text}", (panel_x, y), font, 0.5, (200, 200, 200), 1)
        else:
            ok = quality['checks'].get(key, False)
            color = (0, 255, 0) if ok else (0, 0, 255)
            symbol = "OK" if ok else "NG"
            cv2.putText(canvas, f"  {symbol} {text}", (panel_x, y), font, 0.5, color, 1)
        y += 28

    # 综合判定
    y += 10
    if quality['usable']:
        cv2.putText(canvas, "USABLE", (panel_x, y), font, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(canvas, "NOT USABLE", (panel_x, y), font, 0.8, (0, 0, 255), 2)

    # 操作提示
    y += 50
    cv2.putText(canvas, "[Enter] Confirm", (panel_x, y), font, 0.5, (200, 200, 200), 1)
    y += 25
    cv2.putText(canvas, "[s] Skip  [q] Quit", (panel_x, y), font, 0.5, (200, 200, 200), 1)

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Preview
# ══════════════════════════════════════════════════════════════════════════════

def run_preview(args):
    """Preview 阶段: 实时显示 + 确认姿态"""
    session_dir = os.path.join(os.path.dirname(__file__), 'data', args.session, args.arm)
    os.makedirs(session_dir, exist_ok=True)

    serial = args.camera_serial or CAMERA_SERIALS.get(args.arm, '')
    can_name = args.can or CAN_PORTS.get(args.arm, '')

    print(f"[Preview] arm={args.arm}, camera={serial}, can={can_name}")
    print(f"[Preview] session dir: {session_dir}")

    cam = RealsenseCamera(serial)
    arm = PiperArm(can_name)
    # 被动模式，允许手动拖动
    arm.disable()
    print("[Preview] Arm set to passive mode (drag to position)")

    pose_list = []
    target_count = args.num_poses
    prev_corners = None
    prev_ids = None

    print(f"\n=== Preview: confirm {target_count} poses ===")
    print("Drag arm to desired pose, then press Enter to confirm.\n")

    while len(pose_list) < target_count:
        bgr, _ = cam.grab()
        quality = check_quality(bgr, cam.camera_matrix, cam.dist_coeffs, prev_corners, prev_ids)
        prev_corners = quality['corners']
        prev_ids = quality['ids']

        canvas = draw_quality_overlay(bgr, quality, len(pose_list), target_count)
        cv2.imshow('Preview', canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == 13:  # Enter
            joints = arm.read_joints_rad()
            # 关节极限检查
            violations = (joints < JOINT_LIMITS_RAD[:, 0]) | (joints > JOINT_LIMITS_RAD[:, 1])
            if np.any(violations):
                bad = np.where(violations)[0].tolist()
                print(f"  [REJECT] joints {bad} exceed limits, reposition arm")
                print(f"    q = {np.degrees(joints).round(1)} deg")
                continue
            entry = {
                'label': f'pose_{len(pose_list):02d}',
                'joints': joints.tolist(),
                'quality': {k: bool(v) for k, v in quality['checks'].items()},
                'usable': quality['usable'],
            }
            if not quality['usable']:
                print(f"  [WARN] Pose {len(pose_list)} quality check failed, saving anyway")
            pose_list.append(entry)
            print(f"  Confirmed pose {len(pose_list)}/{target_count}: "
                  f"corners={quality['n_corners']}, err={quality['reproj_err']:.2f}px"
                  if quality['reproj_err'] else f"  Confirmed pose {len(pose_list)}/{target_count}")
        elif key == ord('s'):
            print("  Skipped")
        elif key == ord('q'):
            print("  Quit")
            break

    cv2.destroyAllWindows()
    cam.stop()

    # 保存 (包含硬件上下文, replay 阶段用于校验)
    session_data = {
        'arm': args.arm,
        'camera_serial': serial,
        'can': can_name,
        'poses': pose_list,
    }
    pose_file = os.path.join(session_dir, 'pose_list.json')
    with open(pose_file, 'w') as f:
        json.dump(session_data, f, indent=2)
    print(f"\nSaved {len(pose_list)} poses to {pose_file}")


def _show_replay_status(image: np.ndarray, line1: str, line2: str = ''):
    """在 Replay 窗口显示带状态文字的相机画面"""
    vis = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, line1, (10, 30), font, 0.7, (0, 255, 255), 2)
    if line2:
        cv2.putText(vis, line2, (10, 60), font, 0.6, (200, 200, 200), 1)
    cv2.imshow('Replay', vis)
    cv2.waitKey(1)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Replay
# ══════════════════════════════════════════════════════════════════════════════

def run_replay(args):
    """Replay 阶段: 自动运动 + 稳定后采集"""
    session_dir = os.path.join(os.path.dirname(__file__), 'data', args.session, args.arm)
    pose_file = os.path.join(session_dir, 'pose_list.json')
    if not os.path.exists(pose_file):
        print(f"ERROR: {pose_file} not found. Run --phase preview first.")
        return

    with open(pose_file) as f:
        session_data = json.load(f)

    # 兼容新旧格式: 新格式有 'poses' key, 旧格式直接是 list
    if isinstance(session_data, list):
        pose_list = session_data
    else:
        pose_list = session_data['poses']
        # 使用 preview 阶段保存的硬件配置 (可被命令行覆盖)
        if args.arm and session_data.get('arm') and args.arm != session_data['arm']:
            print(f"  [WARN] --arm={args.arm} but preview used arm={session_data['arm']}")
        if not args.camera_serial and session_data.get('camera_serial'):
            args.camera_serial = session_data['camera_serial']
        if not args.can and session_data.get('can'):
            args.can = session_data['can']

    serial = args.camera_serial or CAMERA_SERIALS.get(args.arm, '')
    can_name = args.can or CAN_PORTS.get(args.arm, '')

    print(f"[Replay] arm={args.arm}, camera={serial}, can={can_name}")
    print(f"[Replay] {len(pose_list)} poses to capture")

    # 关节极限预检查 — 跳过超限姿态, 不中止
    skip_indices = set()
    for i, pose in enumerate(pose_list):
        q = np.array(pose['joints'])
        violations = (q < JOINT_LIMITS_RAD[:, 0]) | (q > JOINT_LIMITS_RAD[:, 1])
        if np.any(violations):
            bad_joints = np.where(violations)[0]
            print(f"  [WARN] {pose['label']}: joints {bad_joints.tolist()} exceed limits, will skip")
            print(f"    q = {np.degrees(q).round(1)} deg")
            skip_indices.add(i)

    # 用户确认
    n_valid = len(pose_list) - len(skip_indices)
    if skip_indices:
        print(f"\n  {len(skip_indices)} pose(s) will be skipped (joint limits).")
    print(f"\n  Will move arm to {n_valid}/{len(pose_list)} poses at {REPLAY_SPEED_PCT}% speed.")
    print(f"  Ensure workspace is clear and e-stop is accessible.")
    resp = input("  Proceed? [y/N] ").strip().lower()
    if resp != 'y':
        print("  Cancelled.")
        return
    print()

    cam = RealsenseCamera(serial)
    arm = PiperArm(can_name)
    fk = PiperFK()

    for i, pose in enumerate(pose_list):
        if i in skip_indices:
            print(f"  [{i+1}/{len(pose_list)}] {pose['label']}: SKIPPED (joint limit)")
            continue
        target = np.array(pose['joints'])
        label = pose['label']
        print(f"  [{i+1}/{len(pose_list)}] {label}: ", end='', flush=True)

        # 安全路径: 先经过零位, 避免手臂扫过桌面/标定板
        print("raise...", end='', flush=True)
        arm.move_to(SAFE_HOME_JOINTS, speed_pct=REPLAY_SPEED_PCT)
        if not arm.wait_settled(SAFE_HOME_JOINTS, threshold_deg=1.0, timeout_s=10.0,
                                extra_wait_s=0.2, cam=cam,
                                status_text=f"[{i+1}/{len(pose_list)}] Raising to zero",
                                speed_pct=REPLAY_SPEED_PCT):
            print(" raise TIMEOUT, skipping")
            continue

        # 从零位移到目标
        print("move...", end='', flush=True)
        arm.move_to(target, speed_pct=REPLAY_SPEED_PCT)
        settled = arm.wait_settled(
            target,
            threshold_deg=REPLAY_SETTLE_THRESHOLD_DEG,
            timeout_s=REPLAY_SETTLE_TIMEOUT_S,
            extra_wait_s=REPLAY_SETTLE_WAIT_S,
            cam=cam,
            status_text=f"[{i+1}/{len(pose_list)}] Moving to {label}",
            speed_pct=REPLAY_SPEED_PCT,
        )
        if not settled:
            # Not at exact target, but check if arm is stable (not moving)
            j1 = arm.read_joints_rad()
            time.sleep(0.2)
            j2 = arm.read_joints_rad()
            vel = np.max(np.abs(np.degrees(j2 - j1))) / 0.2
            if vel < 1.0:  # < 1 deg/s = stable enough
                err_deg = np.max(np.abs(np.degrees(j2 - target)))
                print(f" stable (err={err_deg:.1f}°)", end='')
            else:
                print(f" TIMEOUT (still moving), skipping")
                continue
        else:
            print(" settled", end='')

        # 采集 (带重试)
        bgr, depth, corners, ids, rvec, tvec, reproj_err = None, None, None, None, None, None, None
        for attempt in range(REPLAY_DETECT_RETRIES):
            bgr, depth = cam.grab_avg(REPLAY_AVG_FRAMES)
            corners, ids, rvec, tvec, reproj_err, _ = detect_charuco(
                bgr, cam.camera_matrix, cam.dist_coeffs
            )
            if corners is not None and rvec is not None:
                break
            if attempt < REPLAY_DETECT_RETRIES - 1:
                time.sleep(0.3)

        joints_actual = arm.read_joints_rad()
        T_base_ee = fk.fk_homogeneous(joints_actual)

        if corners is None or rvec is None:
            print(f" -> detection FAILED after {REPLAY_DETECT_RETRIES} attempts, skipping")
            if bgr is not None:
                _show_replay_status(bgr, f"[{i+1}/{len(pose_list)}] DETECTION FAILED", "skipping")
                cv2.waitKey(1000)
            continue

        # 保存
        npz_path = os.path.join(session_dir, f'{label}.npz')
        np.savez(
            npz_path,
            rgb_image=bgr,
            depth_image=depth,
            joint_angles=joints_actual,
            T_base_ee=T_base_ee,
            rvec=rvec,
            tvec=tvec,
            reproj_err=np.array([reproj_err]),
            camera_matrix=cam.camera_matrix,
            dist_coeffs=cam.dist_coeffs,
            intrinsics=json.dumps(cam.intrinsics_dict),
            depth_intrinsics=json.dumps(cam.depth_intrinsics_dict),
            depth_scale=np.array([cam.depth_scale]),
            charuco_corners=corners,
            charuco_ids=ids,
        )
        print(f" -> saved ({len(corners)} corners, err={reproj_err:.2f}px)")

        # 显示采集结果
        vis = bgr.copy()
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
        cv2.drawFrameAxes(vis, cam.camera_matrix, cam.dist_coeffs, rvec, tvec, 0.05)
        _show_replay_status(vis,
                            f"[{i+1}/{len(pose_list)}] {label} CAPTURED",
                            f"{len(corners)} corners, err={reproj_err:.2f}px")
        cv2.waitKey(800)

    # 收回到安全位姿后再释放
    print("  Returning to safe position...", end='', flush=True)
    safe_home = SAFE_HOME_JOINTS
    arm.move_to(safe_home, speed_pct=REPLAY_SPEED_PCT)
    arm.wait_settled(safe_home, threshold_deg=1.0, timeout_s=5.0, extra_wait_s=0.3)
    print(" done")

    cv2.destroyAllWindows()
    cam.stop()
    arm.disable()
    print(f"\nReplay complete. Data saved to {session_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Phase: Head (D435 single-frame)
# ══════════════════════════════════════════════════════════════════════════════

def run_head(args):
    """D435 头顶标定: 单帧采集"""
    session_dir = os.path.join(os.path.dirname(__file__), 'data', args.session)
    os.makedirs(session_dir, exist_ok=True)

    serial = args.camera_serial or CAMERA_SERIALS['head']
    print(f"[Head] camera={serial}")

    cam = RealsenseCamera(serial)
    prev_corners = None
    prev_ids = None

    print("\n=== D435 Head Calibration ===")
    print("Ensure ChArUco board is visible. Press Enter to capture.\n")

    saved = False
    while True:
        bgr, depth = cam.grab()
        quality = check_quality(bgr, cam.camera_matrix, cam.dist_coeffs, prev_corners, prev_ids,
                                is_head=True)
        prev_corners = quality['corners']
        prev_ids = quality['ids']

        canvas = draw_quality_overlay(bgr, quality, 0, 1)
        cv2.imshow('Head Calibration', canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == 13:  # Enter
            if quality['rvec'] is None:
                print("  Detection failed, try again")
                continue
            # 多帧平均降噪，然后对平均图像重新检测 (确保 rvec/tvec 与图像一致)
            bgr_avg, depth = cam.grab_avg(REPLAY_AVG_FRAMES)
            corners, ids, rvec, tvec, reproj_err, _ = detect_charuco(
                bgr_avg, cam.camera_matrix, cam.dist_coeffs
            )
            if rvec is None:
                print("  Detection failed on averaged image, try again")
                continue
            npz_path = os.path.join(session_dir, 'head.npz')
            np.savez(
                npz_path,
                rgb_image=bgr_avg,
                depth_image=depth,
                rvec=rvec,
                tvec=tvec,
                reproj_err=np.array([reproj_err]),
                camera_matrix=cam.camera_matrix,
                dist_coeffs=cam.dist_coeffs,
                intrinsics=json.dumps(cam.intrinsics_dict),
                depth_intrinsics=json.dumps(cam.depth_intrinsics_dict),
                depth_scale=np.array([cam.depth_scale]),
                charuco_corners=corners,
                charuco_ids=ids,
                camera_serial=np.array([serial], dtype='U'),
            )
            print(f"  Saved to {npz_path} ({len(corners)} corners, err={reproj_err:.2f}px)")
            saved = True
            break
        elif key == ord('q'):
            print("  Quit")
            break

    cv2.destroyAllWindows()
    cam.stop()

    # 保存 pose_list.json (与 arm session 格式一致, 供 solve 读取硬件信息)
    if saved:
        session_data = {
            'arm': 'head',
            'camera_serial': serial,
            'can': '',
            'poses': [],
        }
        pose_file = os.path.join(session_dir, 'pose_list.json')
        with open(pose_file, 'w') as f:
            json.dump(session_data, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='手眼标定数据采集')
    parser.add_argument('--phase', required=True, choices=['preview', 'replay', 'head'])
    parser.add_argument('--arm', choices=['left', 'right'], help='机械臂 (preview/replay)')
    parser.add_argument('--can', type=str, help='CAN 接口 (默认按 arm 自动选择)')
    parser.add_argument('--camera-serial', type=str, help='RealSense 序列号 (默认按 arm 自动选择)')
    parser.add_argument('--session', type=str, required=True, help='数据保存目录名')
    parser.add_argument('--num-poses', type=int, default=20, help='目标姿态数 (preview)')
    args = parser.parse_args()

    if args.phase in ('preview', 'replay') and args.arm is None:
        parser.error('--arm is required for preview/replay')

    if args.phase == 'preview':
        run_preview(args)
    elif args.phase == 'replay':
        run_replay(args)
    elif args.phase == 'head':
        run_head(args)


if __name__ == '__main__':
    main()
