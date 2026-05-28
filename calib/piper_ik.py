"""
Piper 机械臂逆运动学 (IK) 封装 + URDF-基准 FK

基于 ikpy.chain.Chain.from_urdf_file(piper_local.urdf) 实现:
  ─ active_links_mask 屏蔽 base/gripper_base/夹爪 prismatic, 只解 6 个 revolute
  ─ initial_position 用调用者传入的 q_seed → 选取离当前位形最近的 IK 分支
  ─ 解算失败/超出关节限位时返回 (q_seed, ok=False), 调用者决定 hold / 重试

EE 定义说明:
  URDF 链末端是 gripper_base (link6 后还有一个 joint6_to_gripper_base 的
  +z 0.1358m 固定偏移), 与 piper_fk.PiperFK 的 link6 EE 定义相差 13.58 cm。
  本模块同时暴露 ik_fk() 使用 URDF 链 → 保证 FK/IK 互为逆。policy 端发布的
  EE pose 一律用 URDF 约定 (gripper_base), 与外部 VLA 模型的常见标定一致。
  Rerun viz (policy_inference_node L1620+) 仍走 PiperFK link6 不变,
  那是 6 个连杆都需要画出来的可视化路径, 不影响 obs/action 数值。

单位: 平移 m, 旋转 rad, 四元数 wxyz。

Usage:
    ik = PiperIK()
    # FK
    xyz, quat_wxyz = ik.fk_xyz_quat(np.zeros(6))
    # IK
    q6, ok = ik.solve(target_pos=np.array([0.3, 0.0, 0.3]),
                       target_quat_wxyz=np.array([1, 0, 0, 0]),
                       q_seed=np.zeros(6))
"""
import os
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# ikpy emits a "fixed link in active_links_mask" UserWarning on import-time
# chain construction; we set the mask explicitly so it's expected — silence it.
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning, module='ikpy.chain')
    from ikpy.chain import Chain


_DEFAULT_URDF = Path(__file__).resolve().parent / 'piper_local.urdf'


class PiperIK:
    """Piper 6-DOF 逆运动学, 输入末端 4×4 齐次 (或 xyz+quat_wxyz), 输出 6 关节角。

    ikpy 链结构 (piper_local.urdf):
        [0] Base link        fixed   ← mask out
        [1] joint1           revolute
        [2] joint2           revolute
        [3] joint3           revolute
        [4] joint4           revolute
        [5] joint5           revolute
        [6] joint6           revolute
        [7] joint6_to_gripper_base  fixed  ← mask out
        [8] joint7           prismatic (左指爪) ← mask out (IK 不解夹爪)

    active_links_mask 长度 = 9, 只对 1..6 启用; ikpy 内部仍会把 mask=False 的
    link 当作可调但不优化, 我们传入时把这些位置填 0 即可。
    """

    # 链中可解的 6 个 revolute joint 在 ikpy 9-link 数组里的下标
    _ACTIVE_IDX = (1, 2, 3, 4, 5, 6)

    def __init__(self, urdf_path: str | os.PathLike | None = None):
        urdf_path = Path(urdf_path) if urdf_path else _DEFAULT_URDF
        if not urdf_path.is_file():
            raise FileNotFoundError(f'piper URDF not found: {urdf_path}')

        # active_links_mask 必须显式给出, 否则 ikpy 默认对所有 link 启用 → 对 fixed/prismatic 报 warning
        mask = [False] * 9
        for i in self._ACTIVE_IDX:
            mask[i] = True

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='ikpy.chain')
            self._chain = Chain.from_urdf_file(
                str(urdf_path),
                active_links_mask=mask,
            )

        self._n_links = len(self._chain.links)  # 9
        assert self._n_links == 9, f'piper URDF chain unexpected length: {self._n_links}'

    # ── 内部工具 ──────────────────────────────────────────────────────

    def _expand_q6(self, q6: np.ndarray) -> np.ndarray:
        """[6] revolute joints → [9] ikpy 全链数组 (mask=False 的位置填 0)."""
        full = np.zeros(self._n_links, dtype=np.float64)
        full[list(self._ACTIVE_IDX)] = q6
        return full

    def _extract_q6(self, full: np.ndarray) -> np.ndarray:
        return np.asarray(full)[list(self._ACTIVE_IDX)].astype(np.float64)

    @staticmethod
    def _xyz_quat_wxyz_to_mat(xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        # scipy Rotation 用 xyzw 顺序
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                             dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
        T[:3, 3] = np.asarray(xyz, dtype=np.float64)
        return T

    # ── FK (URDF 约定, gripper_base) ──────────────────────────────────

    def fk_homogeneous(self, q6_rad: np.ndarray) -> np.ndarray:
        """6 关节角 (rad) → T_base_ee (4×4 齐次, URDF 链末端 gripper_base).

        与 solve() 互为逆。
        """
        q6 = np.asarray(q6_rad, dtype=np.float64).reshape(-1)
        if q6.size != 6:
            raise ValueError(f'q6_rad must be length-6, got {q6.shape}')
        return self._chain.forward_kinematics(self._expand_q6(q6))

    def fk_xyz_quat(self, q6_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """6 关节角 → (xyz [3], quat_wxyz [4])."""
        T = self.fk_homogeneous(q6_rad)
        xyz = T[:3, 3].copy()
        q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
        quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
                             dtype=np.float64)
        return xyz, quat_wxyz

    # ── IK ────────────────────────────────────────────────────────────

    def solve(self,
              target_pos: np.ndarray,
              target_quat_wxyz: np.ndarray,
              q_seed: np.ndarray,
              tol_pos: float = 5e-3,
              tol_rot: float = 5e-2) -> tuple[np.ndarray, bool]:
        """解 EE pose (xyz + quat_wxyz, 单位 m / 单位四元数) → 6 关节角 (rad).

        Args:
            target_pos: [3] 末端在 base 系下的目标位置 (m)
            target_quat_wxyz: [4] 目标姿态 (单位四元数 wxyz)
            q_seed: [6] 当前关节角 (rad), 用作初值, 选取最近的 IK 分支
            tol_pos: 位置容差 (m), 解算后 FK 校验差大于此值视为失败
            tol_rot: 旋转容差 (rad, frobenius 等价角), 超出视为失败

        Returns:
            q6 (rad, [6]) - 若 ok=False 则等于 q_seed, 调用者决定 hold-vs-extrapolate
            ok - 解算是否成功 (位置 & 姿态都在容差内)
        """
        T_target = self._xyz_quat_wxyz_to_mat(target_pos, target_quat_wxyz)
        q_seed = np.asarray(q_seed, dtype=np.float64).reshape(-1)
        if q_seed.size != 6:
            raise ValueError(f'q_seed must be length-6, got {q_seed.shape}')

        initial_full = self._expand_q6(q_seed)

        try:
            sol_full = self._chain.inverse_kinematics_frame(
                T_target,
                initial_position=initial_full,
                orientation_mode='all',  # 6-DOF (位置 + 姿态)
            )
        except Exception:
            return q_seed.copy(), False

        q6 = self._extract_q6(sol_full)

        # FK 校验: 解出的关节再正向求 EE, 检查与 target 的偏差
        T_fk = self._chain.forward_kinematics(sol_full)
        pos_err = float(np.linalg.norm(T_fk[:3, 3] - T_target[:3, 3]))
        # 旋转误差: trace 法
        R_err = T_fk[:3, :3] @ T_target[:3, :3].T
        cos_a = max(min((np.trace(R_err) - 1) / 2.0, 1.0), -1.0)
        rot_err = float(np.arccos(cos_a))

        ok = (pos_err < tol_pos) and (rot_err < tol_rot)
        if not ok:
            return q_seed.copy(), False
        return q6, True

    def solve_mat(self,
                  T_target: np.ndarray,
                  q_seed: np.ndarray,
                  tol_pos: float = 5e-3,
                  tol_rot: float = 5e-2) -> tuple[np.ndarray, bool]:
        """直接接受 4×4 齐次矩阵的 IK 入口 (单位 m, rad)."""
        q_seed = np.asarray(q_seed, dtype=np.float64).reshape(-1)
        if q_seed.size != 6:
            raise ValueError(f'q_seed must be length-6, got {q_seed.shape}')

        initial_full = self._expand_q6(q_seed)
        try:
            sol_full = self._chain.inverse_kinematics_frame(
                T_target,
                initial_position=initial_full,
                orientation_mode='all',
            )
        except Exception:
            return q_seed.copy(), False

        q6 = self._extract_q6(sol_full)
        T_fk = self._chain.forward_kinematics(sol_full)
        pos_err = float(np.linalg.norm(T_fk[:3, 3] - T_target[:3, 3]))
        R_err = T_fk[:3, :3] @ T_target[:3, :3].T
        cos_a = max(min((np.trace(R_err) - 1) / 2.0, 1.0), -1.0)
        rot_err = float(np.arccos(cos_a))

        ok = (pos_err < tol_pos) and (rot_err < tol_rot)
        if not ok:
            return q_seed.copy(), False
        return q6, True


if __name__ == '__main__':
    # FK → IK 自检: 用 PiperIK 自己的 FK 喂回 solve(), 验证互为逆.
    # 一并对比 PiperFK 看 URDF 链与 DH 链的固定 +z 0.1358m 偏差是否符合预期.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from calib.piper_fk import PiperFK  # noqa: E402

    ik = PiperIK()
    fk_dh = PiperFK()

    print('=== Frame check at q=zero ===')
    T_ik_zero = ik.fk_homogeneous(np.zeros(6))
    T_dh_zero = fk_dh.fk_homogeneous(np.zeros(6))
    dz = T_ik_zero[:3, 3] - T_dh_zero[:3, 3]
    print(f'  ikpy gripper_base xyz = {T_ik_zero[:3, 3]}')
    print(f'  PiperFK link6 xyz     = {T_dh_zero[:3, 3]}')
    print(f'  delta (expected ~+0.1358 m on local +z): {dz}')

    print('\n=== FK -> IK round-trip (trajectory tracking, q_seed = prev step) ===')
    # URDF 实际硬限位 (非对称 — joint2/3 都是单边). 真机 puppet joint_states
    # 也是这套约定, 不需要额外坐标变换.
    LO = np.array([-2.618,  0.0, -2.967, -1.745, -1.22, -2.0944])
    HI = np.array([ 2.618,  3.14,  0.0,    1.745,  1.22,  2.0944])
    # 真实部署场景 = 30Hz action chunk 顺序解 IK, 每步 q_seed 都是前一步的解:
    # 相邻两步 joint 差 < 0.05 rad, IK 全局唯一且快速收敛. 测试模拟此行为.
    rng = np.random.default_rng(0)
    # 起点取 URDF 中点 + 小扰动 (确保在 workspace 内且远离边界)
    mid = (LO + HI) / 2
    q = mid + rng.uniform(-0.3, 0.3, size=6) * (HI - LO) * 0.3
    q = np.clip(q, LO + 1e-3, HI - 1e-3)
    n_pass, n_fail = 0, 0
    max_pos_err, max_rot_err = 0.0, 0.0
    seed = q.copy()
    for step in range(100):
        # 每步关节空间走 ≤0.05 rad (模拟 chunk 内相邻帧)
        q = np.clip(q + rng.uniform(-0.05, 0.05, size=6), LO + 1e-3, HI - 1e-3)
        xyz, quat_wxyz = ik.fk_xyz_quat(q)
        q_solve, ok = ik.solve(xyz, quat_wxyz, seed,
                               tol_pos=10e-3, tol_rot=0.2)
        if not ok:
            n_fail += 1
            continue
        xyz2, quat2 = ik.fk_xyz_quat(q_solve)
        pos_err = float(np.linalg.norm(xyz - xyz2))
        # rot err via quat dot product
        cos_half = abs(float(np.clip(np.dot(quat_wxyz, quat2), -1, 1)))
        rot_err = 2 * np.arccos(cos_half)
        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)
        seed = q_solve  # use solution as next step's seed (trajectory tracking)
        n_pass += 1
    print(f'  pass={n_pass}/100  max pos err = {max_pos_err * 1000:.2f} mm  '
          f'max rot err = {np.degrees(max_rot_err):.2f} deg')
