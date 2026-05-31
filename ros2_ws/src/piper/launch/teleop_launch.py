"""ROS2 launch file for 4-arm teleoperation.

Master arms use **arm_master_servo_node** (compatible with both 0xFA leader and
0xFC follower firmware roles): auto-switches subscribe/publish state based on
the physical freedrive button (teach_status field). Slave arms unchanged
(arm_reader / arm_teleop_node mode=1).

Workflow:
  - Press master arm freedrive button (LED bright) → firmware enters compliant
    mode → master_servo detects via teach_status poll → enters publish state →
    encoder + gripper published to /master/joint_left,right → slave's reader
    subscribes and drives slave via JointCtrl + GripperCtrl
  - Release button (LED dark) → master_servo enters subscribe state → motors
    hold last commanded position → slave keeps its last commanded pose
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    auto_enable_slave_arg = DeclareLaunchArgument(
        'auto_enable_slave', default_value='true',
        description='Auto-enable slave arms'
    )

    # --- Master arms: arm_master_servo_node (button-driven, 0xFC-compatible) ---
    # start_state='control' = motors enabled in CAN_CTRL hold (subscribe mode).
    # Pressing the physical button auto-flips to 'drag' (publish mode) at the
    # poll rate. NOTE: master needs 0xFC firmware role for this to work. If
    # role is 0xFA leader, run flash_master_to_follower.py + power cycle.
    master_remap = lambda side: [
        ('/master/joint_states',           f'/master/joint_{side}'),
        ('/puppet/joint_states',           f'/puppet_master/joint_{side}'),
        ('/master/enable',                 f'/teach/master_enable_{side}'),
        ('/master/linkage_config',         f'/teach/master_config_{side}'),
        ('/master/teach_mode',             f'/teach/teach_mode_{side}'),
        ('/master_controled/joint_states', f'/master_controled/joint_{side}'),
        ('/master/button_pressed',         f'/master_button_{side}'),
    ]

    piper_master_left = Node(
        package='piper', executable='arm_master_servo_node.py',
        name='piper_master_left', output='screen',
        parameters=[{
            'can_port': 'can_left_mas',
            'speed_percent': 30,
            'publish_rate_hz': 30.0,
            'start_state': 'control',
        }],
        remappings=master_remap('left'),
    )
    piper_master_right = Node(
        package='piper', executable='arm_master_servo_node.py',
        name='piper_master_right', output='screen',
        parameters=[{
            'can_port': 'can_right_mas',
            'speed_percent': 30,
            'publish_rate_hz': 30.0,
            'start_state': 'control',
        }],
        remappings=master_remap('right'),
    )

    # --- Slave arms: arm_teleop_node mode=1 (unchanged) ---
    piper_slave_left = Node(
        package='piper',
        executable='arm_teleop_node.py',
        name='piper_slave_left',
        output='screen',
        parameters=[{
            'can_port': 'can_left_slave',
            'mode': 1,
            'auto_enable': LaunchConfiguration('auto_enable_slave'),
        }],
        remappings=[
            ('/puppet/arm_status', '/puppet/arm_status_left'),
            ('/puppet/joint_states', '/puppet/joint_left'),
            ('/master/joint_states', '/master/joint_left'),
            ('/puppet/end_pose', '/puppet/end_pose_left'),
            ('/pos_cmd', '/puppet/pos_cmd_left'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_left'),
            ('/enable_flag', '/puppet/enable_left'),
        ],
    )
    piper_slave_right = Node(
        package='piper',
        executable='arm_teleop_node.py',
        name='piper_slave_right',
        output='screen',
        parameters=[{
            'can_port': 'can_right_slave',
            'mode': 1,
            'auto_enable': LaunchConfiguration('auto_enable_slave'),
        }],
        remappings=[
            ('/puppet/arm_status', '/puppet/arm_status_right'),
            ('/puppet/joint_states', '/puppet/joint_right'),
            ('/master/joint_states', '/master/joint_right'),
            ('/puppet/end_pose', '/puppet/end_pose_right'),
            ('/pos_cmd', '/puppet/pos_cmd_right'),
            ('/puppet/end_pose_euler', '/puppet/end_pose_euler_right'),
            ('/enable_flag', '/puppet/enable_right'),
        ],
    )

    return LaunchDescription([
        auto_enable_slave_arg,
        piper_master_left,
        piper_master_right,
        piper_slave_left,
        piper_slave_right,
    ])
