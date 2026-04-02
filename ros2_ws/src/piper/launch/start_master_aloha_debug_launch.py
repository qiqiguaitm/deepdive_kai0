"""ROS2 launch file for start_master_aloha_debug.

Converted from ROS1 start_master_aloha_debug.launch.
Launches a standalone master arm controller for debugging.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    can_port_arg = DeclareLaunchArgument(
        'can_port', default_value='can_left_master',
        description='CAN port for the master arm'
    )
    control_rate_arg = DeclareLaunchArgument(
        'control_rate', default_value='10',
        description='Control rate in Hz'
    )
    auto_test_arg = DeclareLaunchArgument(
        'auto_test', default_value='false',
        description='Auto test on start'
    )

    # Master arm controller node
    master_arm_controller = Node(
        package='piper',
        executable='piper_start_master_node_debug.py',
        name='master_arm_controller',
        output='screen',
        respawn=False,
        parameters=[{
            'can_port': LaunchConfiguration('can_port'),
            'control_rate': LaunchConfiguration('control_rate'),
            'auto_test': LaunchConfiguration('auto_test'),
        }],
        remappings=[
            ('/master/joint_states', '/master/joint_states'),
            ('/master/joint_cmd', '/master/joint_cmd'),
            ('/master/enable', '/master/enable'),
            ('/master/teach_mode', '/master/teach_mode'),
            ('/master/arm_status', '/master/arm_status'),
        ],
    )

    return LaunchDescription([
        can_port_arg,
        control_rate_arg,
        auto_test_arg,
        master_arm_controller,
    ])
