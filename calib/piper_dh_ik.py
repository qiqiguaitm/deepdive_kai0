"""Piper 6-DOF IK against the **DH model** (piper_sdk C_PiperForwardKinematics, CalFK).

为什么不用 calib/piper_ik.py (ikpy/URDF): X-VLA EE6D 训练用 `CalFK(0x01)` 的 link6
位姿编码 action (train_scripts/xvla/data/joint_to_ee6d.py)。部署反解必须用**同一 DH
模型**, 否则 ikpy(URDF) 与 DH 有系统性偏差 (实测 link6 round-trip 达 ~5.7cm / 6.7°,
见 docs/deployment/inference/xvla_inference_bringup.md §4 C)。

本类用 `scipy.optimize.least_squares` 在 CalFK link6 上做带界数值 IK, seed=当前/上一步
关节。round-trip 实测精确还原 (joint err ~0, pos residual ~0)。仅供 EE-mode 部署
(serve_policy_xvla 输出 16D ee → policy_inference_node 反解到关节) 使用。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

# 官方 URDF piper_description 关节限位 (rad), 与 test_inference_server.JOINT_LIMITS 一致
_LIM = [(-2.618, 2.618), (0.0, 3.14), (-2.967, 0.0),
        (-1.745, 1.745), (-1.22, 1.22), (-2.0944, 2.0944)]


class PiperDHIK:
    def __init__(self, dh_is_offset: int = 0x01, rot_weight: float = 0.5):
        # 0x01 = 2° j2/j3 offset, 与 joint_to_ee6d 训练编码一致
        self._fk = C_PiperForwardKinematics(dh_is_offset)
        self._rw = float(rot_weight)
        self._lo = np.array([l for l, _ in _LIM], dtype=np.float64)
        self._hi = np.array([h for _, h in _LIM], dtype=np.float64)
        # 诊断: 最近一次 solve 的残差 (m / rad),供调用方判断失败是位置还是姿态超容差
        self.last_pos_res = 0.0
        self.last_rot_res = 0.0

    def fk_link6(self, q6) -> np.ndarray:
        """6 关节 (rad) → T_base_link6 (4×4), 与训练 EE6D 同一约定 (CalFK link6)."""
        ee = self._fk.CalFK(list(np.asarray(q6, dtype=np.float64).reshape(-1)[:6]))[-1]
        T = np.eye(4)
        T[:3, 3] = np.asarray(ee[:3], dtype=np.float64) / 1000.0  # mm → m
        T[:3, :3] = Rotation.from_euler("xyz", np.radians(ee[3:])).as_matrix()
        return T

    def _residual(self, q, R_t, p_t):
        T = self.fk_link6(q)
        rot = Rotation.from_matrix(R_t.T @ T[:3, :3]).as_rotvec()
        return np.concatenate([T[:3, 3] - p_t, self._rw * rot])

    def solve(self, T_base_link6: np.ndarray, q_seed,
              tol_pos: float = 5e-3, tol_rot: float = 5e-2,
              max_nfev: int = 60, seed_weight: float = 0.0) -> tuple[np.ndarray, bool]:
        """解 T_base_link6 (4×4, m/rad) → (q6 [rad], ok). ok=False 时返回收敛但超容差的解。

        seed_weight>0: 在残差里加 seed_weight*(q - seed) 软约束 → 解优先停在 seed 附近,
        够不到时不翻到远分支(避免相邻步 IK 解跳 ~100° 的姿态不协调)。代价是对可达目标
        引入极小位姿偏差(weight 小则可忽略)。last_pos_res/last_rot_res 始终是纯位姿残差。
        """
        T = np.asarray(T_base_link6, dtype=np.float64)
        R_t, p_t = T[:3, :3], T[:3, 3]
        s = np.clip(np.asarray(q_seed, dtype=np.float64).reshape(-1)[:6], self._lo, self._hi)
        w = float(seed_weight)
        if w > 0.0:
            def resid(q):
                return np.concatenate([self._residual(q, R_t, p_t), w * (q - s)])
        else:
            def resid(q):
                return self._residual(q, R_t, p_t)
        sol = least_squares(resid, s, bounds=(self._lo, self._hi), method="trf",
                            max_nfev=max_nfev, ftol=1e-8, xtol=1e-8)
        q = sol.x
        r = self._residual(q, R_t, p_t)  # 纯位姿残差 (不含 seed 项), 供容差判定与诊断
        self.last_pos_res = float(np.linalg.norm(r[:3]))
        self.last_rot_res = float(np.linalg.norm(r[3:] / max(self._rw, 1e-9)))
        ok = (self.last_pos_res <= tol_pos and self.last_rot_res <= tol_rot)
        return q, ok


if __name__ == "__main__":
    # round-trip self-test: q → CalFK link6 → IK → q' (应精确还原)
    ik = PiperDHIK()
    rng = np.random.default_rng(0)
    jerr = perr = 0.0
    for _ in range(200):
        q = np.array([rng.uniform(l * 0.7, h * 0.7) for l, h in _LIM])
        T = ik.fk_link6(q)
        seed = q + rng.normal(0, 0.1, 6)
        qr, ok = ik.solve(T, seed)
        jerr = max(jerr, float(np.abs(qr - q).max()))
        perr = max(perr, float(np.linalg.norm(ik.fk_link6(qr)[:3, 3] - T[:3, 3])))
    print(f"round-trip: max joint err={jerr:.2e} rad ({np.degrees(jerr):.4f}°), "
          f"max pos err={perr:.2e} m  → {'PASS' if jerr < 1e-2 and perr < 1e-3 else 'FAIL'}")
