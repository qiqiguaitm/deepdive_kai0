"""
Launch 3 RealSense cameras via ROS2 realsense2_camera nodes.

  D435 (top)    → namespace: camera_f  | RGB 640x480   + Depth 640x480  @ 30fps
  D405-A (left) → namespace: camera_l  | RGB 640x480   + Depth 640x480  @ 30fps
  D405-B (right)→ namespace: camera_r  | RGB 640x480   + Depth 640x480  @ 30fps

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
        name='camera_f', namespace='',
        serial='254622070889',
        rgb_w=640, rgb_h=480, depth_w=640, depth_h=480,
    )
    cam_l = make_camera_node(
        name='camera_l', namespace='',
        serial='409122273074',
        rgb_w=640, rgb_h=480, depth_w=640, depth_h=480,
    )
    cam_r = make_camera_node(
        name='camera_r', namespace='',
        serial='409122271568',
        rgb_w=640, rgb_h=480, depth_w=640, depth_h=480,
    )
    return LaunchDescription([cam_f, cam_l, cam_r])
