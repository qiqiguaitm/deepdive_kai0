"""ROS2 launch file for start_master_aloha.

Converted from ROS1 start_master_aloha.launch.
Launches left and right master arms (read and forward to ROS).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable', default_value='true',
        description='Auto-enable arms on startup'
    )

    # Left master arm
    piper_master_left = Node(
        package='piper',
        executable='piper_start_master_node.py',
        name='piper_master_left',
        output='screen',
        parameters=[{
            'can_port': 'can_left',
        }],
        remappings=[
            ('/master/joint_states', '/master/joint_left'),
        ],
    )

    # Right master arm
    piper_master_right = Node(
        package='piper',
        executable='piper_start_master_node.py',
        name='piper_master_right',
        output='screen',
        parameters=[{
            'can_port': 'can_right',
        }],
        remappings=[
            ('/master/joint_states', '/master/joint_right'),
        ],
    )

    return LaunchDescription([
        auto_enable_arg,
        piper_master_left,
        piper_master_right,
    ])
