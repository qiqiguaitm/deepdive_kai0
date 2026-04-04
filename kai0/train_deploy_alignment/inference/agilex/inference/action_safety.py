"""
Piper 关节安全限位模块

[deepdive_kai0 增强] 官方 kai0 在整条推理链路中不做任何关节位置 / 速度 clamp，
仅在 agilex_policy.py 中将超出 ±π 的异常值置零，并在 gripper 上做
  max(0, gripper - RIGHT_OFFSET)
其余完全依赖 Piper 固件层硬件限位。
官方 rtc.py 中预留了 joint_actions_clip() 但为空实现 (pass)。

本模块基于官方 URDF 定义的关节限位 (piper_description.urdf) 提供:
  1. 关节位置 clamp  — 将输出动作限制在物理极限内
  2. 关节速度 clamp  — 限制相邻 step 的最大关节角变化量，防止突变 / 抖动
  3. 可通过 argparse 开关控制，默认开启

数据来源:
  /home/tim/workspace/kai0/train_deploy_alignment/inference/agilex/
    Piper_ros_private-ros-noetic/src/piper_description/urdf/piper_description.urdf
"""

import argparse
import numpy as np

# ── Piper 6-DOF + Gripper 限位 (URDF 权威值) ──────────────────────────
# joint 0~5: revolute joints (rad)
# joint 6:   gripper (rad，URDF 中左指 [0, 0.035]，右指 [-0.035, 0])
PIPER_JOINT_LOWER = np.array([-2.618,  0.0,   -2.967, -1.745, -1.22, -2.0944, 0.0   ])
PIPER_JOINT_UPPER = np.array([ 2.618,  3.14,   0.0,    1.745,  1.22,  2.0944, 0.035 ])

# URDF 速度限制 (rad/s)
PIPER_JOINT_VEL = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 1.0])


class ActionSafety:
    """关节安全限位器。每条手臂独立实例化。

    用法:
        safety_left  = ActionSafety(publish_rate=30.0)
        safety_right = ActionSafety(publish_rate=30.0)

        # 在 puppet_arm_publish 之前:
        left_action  = safety_left(left_action)
        right_action = safety_right(right_action)
    """

    def __init__(
        self,
        publish_rate: float = 30.0,
        enable_position_clamp: bool = True,
        enable_velocity_clamp: bool = True,
        velocity_scale: float = 1.0,
        joint_lower: np.ndarray = PIPER_JOINT_LOWER,
        joint_upper: np.ndarray = PIPER_JOINT_UPPER,
        joint_vel: np.ndarray = PIPER_JOINT_VEL,
    ):
        """
        Args:
            publish_rate: 动作发布频率 (Hz)，用于计算每步最大 delta
            enable_position_clamp: 启用关节位置限幅
            enable_velocity_clamp: 启用关节速度限幅
            velocity_scale: 速度限制的缩放系数 (< 1.0 更保守，> 1.0 更宽松)
            joint_lower/upper/vel: 可自定义限位，默认使用 URDF 值
        """
        self.enable_position_clamp = enable_position_clamp
        self.enable_velocity_clamp = enable_velocity_clamp
        self.joint_lower = joint_lower
        self.joint_upper = joint_upper
        # 每步最大角度变化量 = 速度 / 频率 * 缩放
        self.max_delta_per_step = joint_vel * velocity_scale / max(publish_rate, 1.0)
        self._prev_action: np.ndarray | None = None

    def __call__(self, action_7: np.ndarray) -> np.ndarray:
        """对单臂 7 维动作 [j0..j5, gripper] 执行安全 clamp。"""
        action_7 = action_7.copy()

        if self.enable_position_clamp:
            action_7 = np.clip(action_7, self.joint_lower, self.joint_upper)

        if self.enable_velocity_clamp and self._prev_action is not None:
            delta = action_7 - self._prev_action
            delta = np.clip(delta, -self.max_delta_per_step, self.max_delta_per_step)
            action_7 = self._prev_action + delta

        self._prev_action = action_7.copy()
        return action_7

    def reset(self):
        """重置速度 clamp 的历史状态 (新 episode 开始时调用)。"""
        self._prev_action = None


def add_safety_args(parser: argparse.ArgumentParser) -> None:
    """向 argparse 注入安全相关参数。"""
    group = parser.add_argument_group(
        "Action Safety",
        "[deepdive_kai0 增强] 关节安全限位，官方 kai0 无此功能"
    )
    group.add_argument(
        "--enable_joint_safety", action="store_true", default=True,
        help="启用关节位置 + 速度 clamp (default: True)",
    )
    group.add_argument(
        "--disable_joint_safety", action="store_true", default=False,
        help="禁用关节安全限位 (与官方行为一致)",
    )
    group.add_argument(
        "--safety_position_clamp", action="store_true", default=True,
        help="启用关节位置限幅 (default: True)",
    )
    group.add_argument(
        "--no_safety_position_clamp", action="store_true", default=False,
        help="禁用关节位置限幅",
    )
    group.add_argument(
        "--safety_velocity_clamp", action="store_true", default=True,
        help="启用关节速度限幅 (default: True)",
    )
    group.add_argument(
        "--no_safety_velocity_clamp", action="store_true", default=False,
        help="禁用关节速度限幅",
    )
    group.add_argument(
        "--safety_velocity_scale", type=float, default=1.0,
        help="速度限制缩放系数: < 1.0 更保守, > 1.0 更宽松 (default: 1.0)",
    )


def create_safety_pair(args) -> tuple["ActionSafety | None", "ActionSafety | None"]:
    """根据 argparse 结果创建左右臂安全限位器。

    Returns:
        (safety_left, safety_right) — 若禁用则为 (None, None)
    """
    if getattr(args, "disable_joint_safety", False):
        return None, None

    publish_rate = getattr(args, "publish_rate", 30.0)
    enable_pos = not getattr(args, "no_safety_position_clamp", False)
    enable_vel = not getattr(args, "no_safety_velocity_clamp", False)
    vel_scale = getattr(args, "safety_velocity_scale", 1.0)

    kwargs = dict(
        publish_rate=publish_rate,
        enable_position_clamp=enable_pos,
        enable_velocity_clamp=enable_vel,
        velocity_scale=vel_scale,
    )
    return ActionSafety(**kwargs), ActionSafety(**kwargs)
