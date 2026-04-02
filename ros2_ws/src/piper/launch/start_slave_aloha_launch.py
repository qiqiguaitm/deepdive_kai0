"""ROS2 launch file for start_slave_aloha.

Converted from ROS1 start_slave_aloha.launch.
Launches left and right slave arms.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    mode_arg = DeclareLaunchArgument(
        'mode', default_value='0',
        description='Mode: 0=read only, 1=control slave from master topics'
    )
    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable', default_value='true',
        description='Auto-enable arms on startup'
    )

    # Left slave arm
    piper_slave_left = Node(
        package='piper',
        executable='piper_start_slave_node.py',
        name='piper_slave_left',
        output='screen',
        parameters=[{
            'can_port': 'can_left',
            'mode': LaunchConfiguration('mode'),
            'auto_enable': LaunchConfiguration('auto_enable'),
        }],
        remappings=[
            ('/puppet/arm_status', '/puppet/arm_status_left'),
            ('/puppet/joint_states', '/puppet/joint_left'),
            ('/master/joint_states', '/master/joint_left'),
            ('/puppet/end_pose', '/puppet/end_pose_left'),
        ],
    )

    # Right slave arm
    piper_slave_right = Node(
        package='piper',
        executable='piper_start_slave_node.py',
        name='piper_slave_right',
        output='screen',
        parameters=[{
            'can_port': 'can_right',
            'mode': LaunchConfiguration('mode'),
            'auto_enable': LaunchConfiguration('auto_enable'),
        }],
        remappings=[
            ('/puppet/arm_status', '/puppet/arm_status_right'),
            ('/puppet/joint_states', '/puppet/joint_right'),
            ('/master/joint_states', '/master/joint_right'),
            ('/puppet/end_pose', '/puppet/end_pose_right'),
        ],
    )

    return LaunchDescription([
        mode_arg,
        auto_enable_arg,
        piper_slave_left,
        piper_slave_right,
    ])
