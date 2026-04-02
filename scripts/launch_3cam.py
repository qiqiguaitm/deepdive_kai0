"""
Launch 3 RealSense cameras via ROS2 realsense2_camera nodes.

  D435 (top)    → namespace: camera_f  | RGB 1920x1080 + Depth 1280x720 @ 30fps
  D405-A (left) → namespace: camera_l  | RGB 1280x720  + Depth 1280x720 @ 30fps
  D405-B (right)→ namespace: camera_r  | RGB 1280x720  + Depth 1280x720 @ 30fps

Usage:
  ros2 launch scripts/launch_3cam.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def make_camera_node(name, namespace, serial,
                     rgb_w, rgb_h, depth_w, depth_h, fps=30):
    return Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name=name,
        namespace=namespace,
        output='screen',
        parameters=[{
            'serial_no': serial,
            'camera_name': name,
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': False,
            'enable_accel': False,
            'rgb_camera.color_profile': f'{rgb_w}x{rgb_h}x{fps}',
            'depth_module.depth_profile': f'{depth_w}x{depth_h}x{fps}',
            'align_depth.enable': False,
        }],
    )


def generate_launch_description():
    cam_f = make_camera_node(
        name='camera_f', namespace='camera_f',
        serial='254622070889',
        rgb_w=1920, rgb_h=1080, depth_w=1280, depth_h=720,
    )
    cam_l = make_camera_node(
        name='camera_l', namespace='camera_l',
        serial='409122273074',
        rgb_w=1280, rgb_h=720, depth_w=1280, depth_h=720,
    )
    cam_r = make_camera_node(
        name='camera_r', namespace='camera_r',
        serial='409122271568',
        rgb_w=1280, rgb_h=720, depth_w=1280, depth_h=720,
    )
    return LaunchDescription([cam_f, cam_l, cam_r])
