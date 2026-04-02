"""ROS2 launch file for start_ms_piper.

Converted from ROS1 start_ms_piper.launch.
Launches left and right piper arms in master-slave mode.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    mode_arg = DeclareLaunchArgument(
        'mode', default_value='0',
        description='Mode: 0 reads info from lead and sends to follower via ROS'
    )
    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable', default_value='true',
        description='Auto-enable arms on startup'
    )

    # Left arm node
    piper_left = Node(
        package='piper',
        executable='piper_start_ms_node.py',
        name='piper_left',
        output='screen',
        parameters=[{
            'can_port': 'can_left',
            'mode': LaunchConfiguration('mode'),
            'auto_enable': LaunchConfiguration('auto_enable'),
        }],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_left'),
            ('/master/joint_states', '/master/joint_left'),
            ('/puppet/end_pose', '/puppet/end_pose_left'),
            ('pos_cmd', '/control/end_pose_left'),
        ],
    )

    # Right arm node
    piper_right = Node(
        package='piper',
        executable='piper_start_ms_node.py',
        name='piper_right',
        output='screen',
        parameters=[{
            'can_port': 'can_right',
            'mode': LaunchConfiguration('mode'),
            'auto_enable': LaunchConfiguration('auto_enable'),
        }],
        remappings=[
            ('/puppet/joint_states', '/puppet/joint_right'),
            ('/master/joint_states', '/master/joint_right'),
            ('/puppet/end_pose', '/puppet/end_pose_right'),
            ('pos_cmd', '/control/end_pose_right'),
        ],
    )

    return LaunchDescription([
        mode_arg,
        auto_enable_arg,
        piper_left,
        piper_right,
    ])
