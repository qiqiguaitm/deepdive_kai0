"""
一键启动全套推理栈 (ROS2 native 模式)

包含: 2x piper (mode=0 只读) + 3x 相机 + policy_inference_node

Usage:
  # 纯 ROS2 模式 (推荐, 最低延迟)
  ros2 launch piper inference_full_launch.py mode:=ros2

  # WebSocket 模式 (兼容旧 serve_policy.py, 需先启动 serve_policy)
  ros2 launch piper inference_full_launch.py mode:=websocket

  # 两者兼有模式
  ros2 launch piper inference_full_launch.py mode:=both
"""
import os
import glob
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# 自动构建 CUDA LD_LIBRARY_PATH
_VENV = '/data1/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.12/site-packages'
_NVIDIA_LIBS = ':'.join(
    sorted(glob.glob(os.path.join(_VENV, 'nvidia', '*', 'lib')))
)
_PYTHONPATH = _VENV + ':' + os.path.join(
    '/data1/tim/workspace/deepdive_kai0/kai0', 'src'
)


def generate_launch_description():
    # ── Arguments ──
    mode_arg = DeclareLaunchArgument('mode', default_value='ros2',
        description='ros2 | websocket | both')
    gpu_arg = DeclareLaunchArgument('gpu_id', default_value='0')
    # ── Policy 配置 (参照 serve_policy.py 的 Checkpoint 模式) ──
    # config_name: 决定 transform 链 (图像预处理、归一化、action 后处理)
    # checkpoint_dir: 决定模型权重来源 (GCS 路径会自动下载到 $OPENPI_DATA_HOME)
    #
    # 常见组合:
    #   kai0 最佳模型: config=pi05_flatten_fold_normal  ckpt=.../checkpoints/Task_A/mixed_1
    #   自训练模型:    config=pi05_flatten_fold_normal  ckpt=.../checkpoints/<config>/<exp>/<step>
    config_arg = DeclareLaunchArgument('config_name',
        default_value='pi05_flatten_fold_normal',
        description='Training config name (determines transform pipeline)')
    ckpt_arg = DeclareLaunchArgument('checkpoint_dir',
        default_value='/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1',
        description='Trained model checkpoint path (kai0 best model or your own trained checkpoint)')
    host_arg = DeclareLaunchArgument('host', default_value='localhost',
        description='WebSocket server host (only for mode=websocket)')
    port_arg = DeclareLaunchArgument('port', default_value='8000',
        description='WebSocket server port')
    prompt_arg = DeclareLaunchArgument('prompt',
        default_value='Flatten and fold the cloth.',
        description='Language prompt (must match training config default_prompt)')

    # ── Piper 左臂 (can1, mode=0 只读) ──
    piper_left = Node(
        package='piper', executable='piper_start_ms_node.py',
        name='piper_left', output='screen',
        parameters=[{'can_port': 'can1', 'mode': 0, 'auto_enable': False}],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_left'),
            ('/master/joint_states', '/master/joint_left'),
            ('/puppet/arm_status', '/puppet/arm_status_left'),
            ('/puppet/end_pose', '/puppet/end_pose_left'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_left'),
        ],
    )

    # ── Piper 右臂 (can2, mode=0 只读) ──
    piper_right = Node(
        package='piper', executable='piper_start_ms_node.py',
        name='piper_right', output='screen',
        parameters=[{'can_port': 'can2', 'mode': 0, 'auto_enable': False}],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_right'),
            ('/master/joint_states', '/master/joint_right'),
            ('/puppet/arm_status', '/puppet/arm_status_right'),
            ('/puppet/end_pose', '/puppet/end_pose_right'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_right'),
        ],
    )

    # ── D435 头顶相机 ──
    # RealSense topic 结构: /<namespace>/<camera_name>/color/image_raw
    # namespace='camera_f' → /camera_f/camera/color/image_raw
    cam_f = Node(
        package='realsense2_camera', executable='realsense2_camera_node',
        name='camera', namespace='camera_f', output='screen',
        parameters=[{
            'serial_no': '254622070889',
            'camera_name': 'camera',
            'enable_color': True,
            'enable_depth': False,  # kai0 推理不用 depth
            'enable_infra1': False, 'enable_infra2': False,
            'enable_gyro': False, 'enable_accel': False,
            'rgb_camera.color_profile': '640x480x30',
        }],
    )

    # ── D405-L 左腕相机 ──
    cam_l = Node(
        package='realsense2_camera', executable='realsense2_camera_node',
        name='camera_l', namespace='', output='screen',
        parameters=[{
            'serial_no': '409122273074',
            'camera_name': 'camera_l',
            'enable_color': True,
            'enable_depth': False,
            'enable_infra1': False, 'enable_infra2': False,
            'enable_gyro': False, 'enable_accel': False,
            'rgb_camera.color_profile': '640x480x30',
        }],
    )

    # ── D405-R 右腕相机 ──
    cam_r = Node(
        package='realsense2_camera', executable='realsense2_camera_node',
        name='camera_r', namespace='', output='screen',
        parameters=[{
            'serial_no': '409122271568',
            'camera_name': 'camera_r',
            'enable_color': True,
            'enable_depth': False,
            'enable_infra1': False, 'enable_infra2': False,
            'enable_gyro': False, 'enable_accel': False,
            'rgb_camera.color_profile': '640x480x30',
        }],
    )

    # ── Policy Inference Node ──
    policy_node = Node(
        package='piper', executable='policy_inference_node.py',
        name='policy_inference', output='screen',
        parameters=[{
            'mode': LaunchConfiguration('mode'),
            'config_name': LaunchConfiguration('config_name'),
            'checkpoint_dir': LaunchConfiguration('checkpoint_dir'),
            'host': LaunchConfiguration('host'),
            'port': LaunchConfiguration('port'),
            'prompt': LaunchConfiguration('prompt'),
            'gpu_id': LaunchConfiguration('gpu_id'),
            'img_front_topic': '/camera_f/camera/color/image_raw',
            'img_left_topic': '/camera_l/camera/color/image_raw',
            'img_right_topic': '/camera_r/camera/color/image_raw',
            'puppet_left_topic': '/puppet/joint_left',
            'puppet_right_topic': '/puppet/joint_right',
        }],
    )

    # 环境变量 (CUDA 库 + Python 路径, 追加到现有值)
    existing_ld = os.environ.get('LD_LIBRARY_PATH', '')
    existing_py = os.environ.get('PYTHONPATH', '')
    set_ld = SetEnvironmentVariable('LD_LIBRARY_PATH',
        _NVIDIA_LIBS + ':' + existing_ld if existing_ld else _NVIDIA_LIBS)
    set_py = SetEnvironmentVariable('PYTHONPATH',
        _PYTHONPATH + ':' + existing_py if existing_py else _PYTHONPATH)
    set_cache = SetEnvironmentVariable('JAX_COMPILATION_CACHE_DIR', '/tmp/xla_cache')

    # 相机顺序启动，每个间隔 3 秒，避免 USB 带宽/电源冲击
    cam_f_delayed = TimerAction(period=0.0, actions=[cam_f])
    cam_l_delayed = TimerAction(period=3.0, actions=[cam_l])
    cam_r_delayed = TimerAction(period=6.0, actions=[cam_r])
    # policy node 等相机全部启动后再启动
    policy_delayed = TimerAction(period=12.0, actions=[policy_node])

    return LaunchDescription([
        set_ld, set_py, set_cache,
        mode_arg, gpu_arg, config_arg, ckpt_arg, host_arg, port_arg, prompt_arg,
        piper_left, piper_right,
        cam_f_delayed, cam_l_delayed, cam_r_delayed,
        policy_delayed,
    ])
