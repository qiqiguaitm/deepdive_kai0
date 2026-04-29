"""
端到端推理测试 Launch — Piper (mode=0 只读) + 3 相机

piper 以 mode=0 启动 (arm_reader_node 被动读取模式): 只发布关节状态，不接受
控制命令 → 安全，不会控制臂运动。推理脚本发布的动作命令无接收者 → 不产生任何运动。

Topic 映射:
  /puppet/joint_left   ← piper 左臂关节反馈 (can_left_slave)
  /puppet/joint_right  ← piper 右臂关节反馈 (can_right_slave)
  /camera_f/color/image_raw  ← D435 RGB
  /camera_l/color/image_raw  ← D405-L RGB
  /camera_r/color/image_raw  ← D405-R RGB

Usage:
  ros2 launch scripts/launch_e2e_test.py
"""
import importlib.util
from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def _load_depth_enabled_map() -> dict:
    """Read CAMERA_DEPTH_ENABLED from config/camera_depth_flags.py — same
    macro file the data-collection / autonomy stack uses, so e2e parity
    tests automatically reflect production depth toggles.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / 'config' / 'camera_depth_flags.py'
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                'kai0_camera_depth_flags_e2e', candidate)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return dict(mod.CAMERA_DEPTH_ENABLED)
    return {}


_DEPTH_ENABLED = _load_depth_enabled_map()


def _depth_node_params(serial: str, name: str, enable_depth: bool) -> dict:
    p = {
        'serial_no': serial,
        'camera_name': name,
        'enable_color': True,
        'enable_depth': enable_depth,
        'enable_infra1': False,
        'enable_infra2': False,
        'enable_gyro': False,
        'enable_accel': False,
        'rgb_camera.color_profile': '640x480x30',
        'align_depth.enable': False,
        'initial_reset': False,
    }
    if enable_depth:
        p['depth_module.depth_profile'] = '640x480x30'
    return p


def _depth_remaps(ns: str, enable_depth: bool) -> list:
    color = [
        ('color/image_raw', f'/{ns}/color/image_raw'),
        ('color/image_rect_raw', f'/{ns}/color/image_raw'),
        ('color/camera_info', f'/{ns}/color/camera_info'),
    ]
    if enable_depth:
        color += [
            ('depth/image_rect_raw', f'/{ns}/depth/image_raw'),
            ('depth/camera_info', f'/{ns}/depth/camera_info'),
        ]
    return color


def generate_launch_description():
    # ── Piper 左臂 (can_left_slave, mode=0 只读) ──
    piper_left = Node(
        package='piper',
        executable='arm_reader_node.py',
        name='piper_left',
        output='screen',
        parameters=[{
            'can_port': 'can_left_slave',
            'mode': 0,          # 只读: 发布关节状态，不接受控制
            'auto_enable': False,
        }],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_left'),
            ('/master/joint_states', '/master/joint_left'),
            ('/puppet/arm_status', '/puppet/arm_status_left'),
            ('/puppet/end_pose', '/puppet/end_pose_left'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_left'),
        ],
    )

    # ── Piper 右臂 (can_right_slave, mode=0 只读) ──
    piper_right = Node(
        package='piper',
        executable='arm_reader_node.py',
        name='piper_right',
        output='screen',
        parameters=[{
            'can_port': 'can_right_slave',
            'mode': 0,
            'auto_enable': False,
        }],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_right'),
            ('/master/joint_states', '/master/joint_right'),
            ('/puppet/arm_status', '/puppet/arm_status_right'),
            ('/puppet/end_pose', '/puppet/end_pose_right'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_right'),
        ],
    )

    # 每相机的 enable_depth 由 config/camera_depth_flags.py 宏定义统一控制.
    en_top = _DEPTH_ENABLED.get('top_head', False)
    en_l = _DEPTH_ENABLED.get('hand_left', False)
    en_r = _DEPTH_ENABLED.get('hand_right', False)

    # ── D435 头顶相机 (namespace=camera_f, 但 topic 重映射到 /camera_f/*) ──
    cam_f = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera_f',
        namespace='',
        output='screen',
        parameters=[_depth_node_params('254622070889', 'camera_f', en_top)],
        remappings=_depth_remaps('camera_f', en_top),
    )

    # ── D405-L 左腕相机 ──
    cam_l = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera_l',
        namespace='',
        output='screen',
        parameters=[_depth_node_params('409122273074', 'camera_l', en_l)],
        remappings=_depth_remaps('camera_l', en_l),
    )

    # ── D405-R 右腕相机 ──
    cam_r = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera_r',
        namespace='',
        output='screen',
        parameters=[_depth_node_params('409122271568', 'camera_r', en_r)],
        remappings=_depth_remaps('camera_r', en_r),
    )

    return LaunchDescription([piper_left, piper_right, cam_f, cam_l, cam_r])
