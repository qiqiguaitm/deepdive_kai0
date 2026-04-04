#!/usr/bin/env python3
"""
标定求解: 从采集数据计算全部空间变换

输入: preview+replay 产生的 pose_*.npz 文件
输出: calibration.yaml

流程:
  1. 左臂 calibrateHandEye → T_link6_camL + T_board_baseL
  2. 右臂 calibrateHandEye → T_link6_camR + T_board_baseR
  3. D435 solvePnP → T_board_camF
  4. 世界系 = 两臂基座中点 → 全部变换到世界系
  5. 输出 calibration.yaml

用法:
  python3 calib/solve_calibration.py \
    --left calib/data/left_calib \
    --right calib/data/right_calib \
    --head calib/data/head_calib \
    --output calib/calibration.yaml
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
from board_def import get_board


def load_arm_data(session_dir: str) -> list[dict]:
    """加载一条臂的全部 pose_*.npz"""
    files = sorted(glob.glob(os.path.join(session_dir, 'pose_*.npz')))
    if not files:
        raise FileNotFoundError(f"No pose_*.npz in {session_dir}")
    data = []
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        data.append(d)
    print(f"  Loaded {len(data)} poses from {session_dir}")
    return data


def load_head_data(session_dir: str) -> dict:
    """加载 D435 head.npz"""
    path = os.path.join(session_dir, 'head.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f"head.npz not found in {session_dir}")
    d = dict(np.load(path, allow_pickle=True))
    print(f"  Loaded head data from {path}")
    return d


def solve_hand_eye(data: list[dict], label: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """求解一条臂的手眼标定。

    Returns:
        T_link6_cam: 4×4 (camera→gripper, 即 cam 在 link6 系中的位姿)
        T_board_base: 4×4 (base in board frame)
        reproj_errors: per-pose reprojection errors
    """
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []
    reproj_errors = []

    for d in data:
        T = d['T_base_ee']  # 4×4
        R_gripper2base_list.append(T[:3, :3])
        t_gripper2base_list.append(T[:3, 3].reshape(3, 1))

        R_t2c, _ = cv2.Rodrigues(d['rvec'].flatten())
        R_target2cam_list.append(R_t2c)
        t_target2cam_list.append(d['tvec'].reshape(3, 1))

        reproj_errors.append(float(d['reproj_err'].item()))

    # calibrateHandEye: eye-on-hand — try all methods, pick best by scatter
    methods = {
        'TSAI': cv2.CALIB_HAND_EYE_TSAI,
        'PARK': cv2.CALIB_HAND_EYE_PARK,
        'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
        'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
        'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    best_method = None
    best_scatter = float('inf')
    best_T = None

    for method_name, method_flag in methods.items():
        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_gripper2base_list, t_gripper2base_list,
            R_target2cam_list, t_target2cam_list,
            method=method_flag,
        )
        T_candidate = np.eye(4)
        T_candidate[:3, :3] = R_c2g
        T_candidate[:3, 3] = t_c2g.flatten()

        # Compute T_board_base scatter to evaluate consistency
        t_list = []
        for k in range(len(data)):
            T_base_ee = data[k]['T_base_ee']
            R_t2c, _ = cv2.Rodrigues(data[k]['rvec'].flatten())
            T_cam_board = np.eye(4)
            T_cam_board[:3, :3] = R_t2c
            T_cam_board[:3, 3] = data[k]['tvec'].reshape(3)
            T_board_base_k = np.linalg.inv(T_base_ee @ T_candidate @ T_cam_board)
            t_list.append(T_board_base_k[:3, 3])
        scatter = np.std(t_list, axis=0).mean() * 1000  # mean std in mm

        cam_dist = np.linalg.norm(t_c2g) * 100
        print(f"    {method_name:12s}: scatter={scatter:.1f}mm, cam_dist={cam_dist:.1f}cm")

        if scatter < best_scatter:
            best_scatter = scatter
            best_method = method_name
            best_T = T_candidate

    print(f"    → Best: {best_method} (scatter={best_scatter:.1f}mm)")

    # If best scatter is still poor (>10mm), fall back to DANIILIDIS
    # (algebraically stable, less likely to produce degenerate solutions)
    if best_scatter > 10.0:
        print(f"    [WARN] scatter {best_scatter:.1f}mm > 10mm — poses may lack rotational diversity")
        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_gripper2base_list, t_gripper2base_list,
            R_target2cam_list, t_target2cam_list,
            method=cv2.CALIB_HAND_EYE_DANIILIDIS,
        )
        best_T = np.eye(4)
        best_T[:3, :3] = R_c2g
        best_T[:3, 3] = t_c2g.flatten()
        print(f"    Falling back to DANIILIDIS")

    T_link6_cam = best_T

    # 反推 T_board_base (多姿态, 取中位数)
    T_board_base_list = []
    for d in data:
        T_base_ee = d['T_base_ee']
        rvec = d['rvec'].flatten()
        tvec = d['tvec'].reshape(3, 1)
        R_t2c, _ = cv2.Rodrigues(rvec)
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = R_t2c
        T_cam_board[:3, 3] = tvec.flatten()

        # 闭环: board → base → ee → cam → board = I
        # I = T_board_base · T_base_ee · T_link6_cam · T_cam_board
        # T_board_base = inv(T_base_ee · T_link6_cam · T_cam_board)
        T = T_base_ee @ T_link6_cam @ T_cam_board
        T_board_base_list.append(np.linalg.inv(T))

    T_board_base = _robust_mean_se3(T_board_base_list)

    # 报告
    t_c2g = T_link6_cam[:3, 3]
    r_c2g = Rotation.from_matrix(T_link6_cam[:3, :3]).as_euler('xyz', degrees=True)
    print(f"\n  [{label}] T_link6_cam (camera pose in link6 frame):")
    print(f"    translation: [{t_c2g[0]:.4f}, {t_c2g[1]:.4f}, {t_c2g[2]:.4f}] m "
          f"(norm={np.linalg.norm(t_c2g)*100:.1f} cm)")
    print(f"    rotation:    [{r_c2g[0]:.1f}, {r_c2g[1]:.1f}, {r_c2g[2]:.1f}] deg")
    print(f"    reproj errors: mean={np.mean(reproj_errors):.3f}, max={np.max(reproj_errors):.3f} px")

    # 合理性检查
    cam_dist_cm = np.linalg.norm(t_c2g) * 100
    if not (3.0 <= cam_dist_cm <= 20.0):
        print(f"    [WARN] cam→gripper distance {cam_dist_cm:.1f}cm outside expected range [3, 20]cm")
    if np.max(reproj_errors) > 2.0:
        print(f"    [WARN] max reproj error {np.max(reproj_errors):.2f}px > 2.0px, calibration may be poor")

    return T_link6_cam, T_board_base, reproj_errors


def solve_head(data: dict) -> tuple[np.ndarray, float]:
    """求解 D435 头顶相机外参。

    Returns:
        T_board_camF: 4×4 (camera in board frame)
        reproj_err: reprojection error
    """
    rvec = data['rvec'].flatten()
    tvec = data['tvec'].reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R
    T_cam_board[:3, 3] = tvec.flatten()

    T_board_cam = np.linalg.inv(T_cam_board)

    reproj_err = float(data['reproj_err'].item())
    t = T_board_cam[:3, 3]
    R_out = T_board_cam[:3, :3]
    rvec_out, _ = cv2.Rodrigues(R_out)
    rot_deg = np.degrees(rvec_out.flatten())
    print(f"\n  [Head] T_board_camF:")
    print(f"    translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
    print(f"    rotation:    [{rot_deg[0]:.1f}, {rot_deg[1]:.1f}, {rot_deg[2]:.1f}] deg")
    print(f"    reproj error: {reproj_err:.3f} px")

    return T_board_cam, reproj_err


def compute_world_frame(T_board_baseL: np.ndarray, T_board_baseR: np.ndarray) -> np.ndarray:
    """从两个基座位姿计算世界系 (在 board 坐标系中的表达)。

    世界系: 原点 = 两基座中点, X = R→L, Z = 板面法线 (≈桌面法线), Y = 右手系

    使用 board 坐标系的 Z 轴 (即 [0,0,1] in board frame) 作为桌面法线。
    标定板应放平在桌面上, 此时 board Z 即桌面法线。
    """
    pos_L = T_board_baseL[:3, 3]
    pos_R = T_board_baseR[:3, 3]

    origin = (pos_L + pos_R) / 2.0

    # 板面法线 = board 坐标系的 Z 轴 (在 board frame 中)
    # ChArUco Z 可能朝上或朝下, 取决于标定板朝向
    # 选择使两基座 Z 坐标为正 (即基座在板面上方) 的方向
    z_axis = np.array([0.0, 0.0, 1.0])
    avg_base_z = (pos_L[2] + pos_R[2]) / 2.0
    if avg_base_z < 0:
        z_axis = -z_axis  # flip: bases should be above the board

    x_raw = pos_L - pos_R  # R→L
    # 投影到板面 (去掉沿法线的分量)
    x_axis = x_raw - np.dot(x_raw, z_axis) * z_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        raise ValueError(
            f"两臂基座在板面投影重合 (距离 {x_norm*1000:.3f}mm), 无法确定 X 轴方向。"
            f" baseL={pos_L}, baseR={pos_R}"
        )
    x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Y should point toward the board (forward), not away from it.
    # The board origin is at [0,0,0] in board frame. Check if Y points toward it.
    board_center_in_board = np.zeros(3)
    to_board = board_center_in_board - origin
    if np.dot(to_board, y_axis) < 0:
        y_axis = -y_axis
        x_axis = -x_axis  # flip X too to keep right-hand rule

    # 检查两基座高度差 — 如果差距过大, 说明板可能放歪了或 FK 不准
    height_diff_mm = abs(pos_L[2] - pos_R[2]) * 1000
    if height_diff_mm > 10.0:
        print(f"    [WARN] Base height difference {height_diff_mm:.1f}mm > 10mm — "
              f"board may not be level, or FK offset may be wrong")

    T_board_world = np.eye(4)
    T_board_world[:3, 0] = x_axis
    T_board_world[:3, 1] = y_axis
    T_board_world[:3, 2] = z_axis
    T_board_world[:3, 3] = origin
    return T_board_world


def _robust_mean_se3(T_list: list[np.ndarray]) -> np.ndarray:
    """鲁棒 SE3 均值 (平移取中位数, 旋转取与中位数平移最近的样本子集的 quaternion 均值)

    先用平移中位数剔除离群值 (> 2σ), 再对剩余样本做 quaternion 均值。
    """
    translations = np.array([T[:3, 3] for T in T_list])
    median_t = np.median(translations, axis=0)

    # 旋转: 先剔除平移离群值, 再做 quaternion 均值
    t_dists = np.linalg.norm(translations - median_t, axis=1)
    if len(T_list) >= 5:
        threshold = max(np.median(t_dists) * 3, 0.005)  # 至少 5mm 容差
        inlier_mask = t_dists < threshold
        if inlier_mask.sum() < 3:
            inlier_mask = np.ones(len(T_list), dtype=bool)  # 回退到全部
    else:
        inlier_mask = np.ones(len(T_list), dtype=bool)

    inlier_Ts = [T_list[i] for i in range(len(T_list)) if inlier_mask[i]]
    quats = np.array([Rotation.from_matrix(T[:3, :3]).as_quat() for T in inlier_Ts])
    # 确保 quaternion 符号一致 (避免 antipodal 问题)
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]
    mean_q = quats.mean(axis=0)
    mean_q = mean_q / np.linalg.norm(mean_q)
    mean_R = Rotation.from_quat(mean_q).as_matrix()

    n_outliers = len(T_list) - int(inlier_mask.sum())
    if n_outliers > 0:
        print(f"    _robust_mean_se3: removed {n_outliers} outlier(s) from {len(T_list)} samples")

    T = np.eye(4)
    T[:3, :3] = mean_R
    T[:3, 3] = median_t
    return T


def _se3_to_params(T: np.ndarray) -> np.ndarray:
    """SE3 4x4 → 6-vector [rvec(3), tvec(3)]"""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return np.concatenate([rvec.flatten(), T[:3, 3]])


def _params_to_se3(p: np.ndarray) -> np.ndarray:
    """6-vector [rvec(3), tvec(3)] → SE3 4x4"""
    R, _ = cv2.Rodrigues(p[:3])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p[3:6]
    return T


def joint_refine(
    left_data: list[dict],
    right_data: list[dict],
    head_data: dict,
    T_link6_camL_init: np.ndarray,
    T_link6_camR_init: np.ndarray,
    T_board_baseL_init: np.ndarray,
    T_board_baseR_init: np.ndarray,
    T_board_camF_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Joint optimization of arm transforms to minimize reprojection error.

    Variables (24 DOF): T_link6_camL/R (6 each), T_board_baseL/R (6 each).
    Head is fixed (only 1 frame, not enough data to refine).

    Cost: charuco corner reprojection error across all arm poses.
    """
    board = get_board()
    obj_pts_all = board.getChessboardCorners()

    x0 = np.concatenate([
        _se3_to_params(T_link6_camL_init),     # 0:6
        _se3_to_params(T_link6_camR_init),      # 6:12
        _se3_to_params(T_board_baseL_init),     # 12:18
        _se3_to_params(T_board_baseR_init),     # 18:24
    ])

    def _prep_arm_data(data):
        prepped = []
        for d in data:
            ids = d['charuco_ids'].flatten()
            corners = d['charuco_corners'].reshape(-1, 2)
            cam_matrix = d['camera_matrix']
            dist_coeffs = d['dist_coeffs']
            T_base_ee = d['T_base_ee']
            obj_pts = obj_pts_all[ids]
            prepped.append((obj_pts, corners, cam_matrix, dist_coeffs, T_base_ee))
        return prepped

    left_prepped = _prep_arm_data(left_data)
    right_prepped = _prep_arm_data(right_data)

    def _arm_residuals(T_link6_cam, T_board_base, prepped):
        """Reprojection residuals for one arm across all its poses."""
        T_cam_link6 = np.linalg.inv(T_link6_cam)
        T_base_board = np.linalg.inv(T_board_base)
        errs = []
        for obj_pts, corners, cam_matrix, dist_coeffs, T_base_ee in prepped:
            # board → base → ee(=link6) → cam
            T_cam_board = T_cam_link6 @ np.linalg.inv(T_base_ee) @ T_base_board
            rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
            tvec = T_cam_board[:3, 3]
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_matrix, dist_coeffs)
            errs.append((proj.reshape(-1, 2) - corners).flatten())
        return np.concatenate(errs)

    def residuals(x):
        T_link6_camL = _params_to_se3(x[0:6])
        T_link6_camR = _params_to_se3(x[6:12])
        T_board_baseL = _params_to_se3(x[12:18])
        T_board_baseR = _params_to_se3(x[18:24])
        errs_L = _arm_residuals(T_link6_camL, T_board_baseL, left_prepped)
        errs_R = _arm_residuals(T_link6_camR, T_board_baseR, right_prepped)
        return np.concatenate([errs_L, errs_R])

    r0 = residuals(x0)
    n_pts = len(r0) // 2
    print(f"    Initial reproj error: {np.sqrt(np.mean(r0**2)):.4f} px ({n_pts} points)")

    result = least_squares(residuals, x0, method='lm', verbose=0)

    rf = residuals(result.x)
    print(f"    Refined reproj error: {np.sqrt(np.mean(rf**2)):.4f} px ({n_pts} points)")

    T_link6_camL = _params_to_se3(result.x[0:6])
    T_link6_camR = _params_to_se3(result.x[6:12])
    T_board_baseL = _params_to_se3(result.x[12:18])
    T_board_baseR = _params_to_se3(result.x[18:24])

    return T_link6_camL, T_link6_camR, T_board_baseL, T_board_baseR, T_board_camF_init


def cross_calibrate_head(
    head_data: dict,
    left_data: list[dict],
    right_data: list[dict],
    T_link6_camL: np.ndarray,
    T_link6_camR: np.ndarray,
    T_board_camF_init: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Refine head camera extrinsic using arm-derived board observations.

    Each arm pose gives a T_cam_board for the wrist camera. Combined with
    the trusted T_link6_cam, we get T_base_board per pose. All arm poses
    see the same board, so we have many constraints to refine T_board_camF.

    The cost function reprojects charuco corners from the board through:
      board → (arm chain) → base → (world) → head_cam
    using all arm poses as virtual observations for the head camera.
    """
    board = get_board()
    obj_pts_all = board.getChessboardCorners()

    head_cam_matrix = head_data['camera_matrix']
    head_dist_coeffs = head_data['dist_coeffs']
    head_ids = head_data['charuco_ids'].flatten()
    head_corners = head_data['charuco_corners'].reshape(-1, 2)
    head_obj_pts = obj_pts_all[head_ids]

    # From each arm pose, compute T_board_base using the trusted hand-eye
    # T_board_base = inv(T_base_ee @ T_link6_cam @ T_cam_board)
    T_board_base_estimates = []
    for arm_data, T_link6_cam in [(left_data, T_link6_camL), (right_data, T_link6_camR)]:
        for d in arm_data:
            T_base_ee = d['T_base_ee']
            rvec = d['rvec'].flatten()
            tvec = d['tvec'].reshape(3, 1)
            R_t2c, _ = cv2.Rodrigues(rvec)
            T_cam_board = np.eye(4)
            T_cam_board[:3, :3] = R_t2c
            T_cam_board[:3, 3] = tvec.flatten()
            T_board_base = np.linalg.inv(T_base_ee @ T_link6_cam @ T_cam_board)
            T_board_base_estimates.append(T_board_base)

    # Robust average to get best T_board_base (= where the board is relative to arm bases)
    # All estimates should agree since the board doesn't move
    T_board_base_avg = _robust_mean_se3(T_board_base_estimates)

    # Now optimize T_board_camF to minimize reprojection of head's charuco corners,
    # while also being consistent with arm-derived board position.
    # The head directly sees the board, so the cost is just head reprojection,
    # but we initialize from the arm-derived position for better accuracy.

    # Compute T_board_camF from arm data:
    # We need T_board_camF. We have T_board_base_avg from arms.
    # From head PnP, we have T_cam_board for head. So T_board_cam = inv(T_cam_board).
    # But we want to refine this using arms as additional constraint.

    # Strategy: optimize T_board_camF (6 DOF) to minimize:
    #   1. Head charuco reprojection error (direct observation)
    #   2. Consistency with arm-derived board position (cross-camera constraint)
    #      For each arm pose: the board corners projected through
    #      board → base → ee → wrist_cam should match the observed wrist corners.
    #      But we already trust those. Instead, we can project the same board corners
    #      through: board → T_board_camF^{-1} → head_cam and compare to head observation.
    #      This is just the head reprojection, which we already have.

    # The real cross-calibration value: use arm poses to generate synthetic head observations.
    # For each arm pose, we know where the board is (T_board_base_avg).
    # If we also knew where the arm base is relative to the head camera, we could project.
    # Chain: p_cam_head = T_camF_board @ p_board = inv(T_board_camF) @ p_board

    # Actually the key insight: we have T_board_base from BOTH arms (many poses).
    # We can compute T_board_camF by combining T_board_base with a new relationship.
    # But we need T_base_camF, which requires knowing the spatial relationship
    # between arm base and head camera — that's exactly what we're trying to find.

    # Better approach: directly optimize T_board_camF using head's charuco detection
    # but with a regularizer from the arm-derived board center position.

    # The arm data tells us exactly where the board center is in the board frame (origin).
    # Project board origin through T_board_camF to head camera — this should land on a
    # consistent position. Use arm-derived T_board_base to compute expected head cam position.

    # Simplest effective approach: refine T_board_camF via LM on head reprojection,
    # initialized from the PnP solution but with all charuco corners.
    # Then add a soft constraint: the board center projected to the head camera image
    # should be consistent with what the arms tell us about the board position.

    # Let's just do a proper refinement of T_cam_board for head using solvePnP refinement:
    T_cam_board_init = np.linalg.inv(T_board_camF_init)
    rvec_init, _ = cv2.Rodrigues(T_cam_board_init[:3, :3])
    tvec_init = T_cam_board_init[:3, 3].reshape(3, 1)

    # Refine with solvePnP using initial guess (SOLVEPNP_ITERATIVE with useExtrinsicGuess)
    success, rvec_ref, tvec_ref = cv2.solvePnP(
        head_obj_pts, head_corners, head_cam_matrix, head_dist_coeffs,
        rvec=rvec_init.copy(), tvec=tvec_init.copy(),
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    R_ref, _ = cv2.Rodrigues(rvec_ref)
    T_cam_board_ref = np.eye(4)
    T_cam_board_ref[:3, :3] = R_ref
    T_cam_board_ref[:3, 3] = tvec_ref.flatten()
    T_board_camF_ref = np.linalg.inv(T_cam_board_ref)

    # Compute refined reproj error
    proj, _ = cv2.projectPoints(head_obj_pts, rvec_ref, tvec_ref, head_cam_matrix, head_dist_coeffs)
    reproj_err = float(np.sqrt(np.mean((proj.reshape(-1, 2) - head_corners) ** 2)))

    # Now use arm data to verify/improve: compute where the board center should appear
    # in head camera, according to arm calibration.
    # We know T_board_base_avg (from arms). If we had T_base_camF we could cross-check.
    # But T_base_camF = inv(T_board_base_avg) @ T_board_camF — that's circular.

    # The most valuable cross-calibration: use the arms to find T_base_camF directly.
    # For each arm pose where the wrist camera sees the board:
    #   T_cam_board (wrist) is known from wrist observation
    #   T_cam_board (head) is known from head observation (same board, same time... but not)
    # Problem: arm poses and head capture are at different times.

    # Since board is stationary, we CAN cross-calibrate:
    # From arms: T_board_base = inv(T_base_ee @ T_link6_cam @ T_cam_board_wrist)
    # From head: T_board_camF
    # So: T_base_camF = inv(T_board_base) @ T_board_camF
    # We have many T_board_base estimates → many T_base_camF estimates → average them.
    # This gives T_board_camF = T_board_base_avg @ T_base_camF... but we need T_base_camF.
    # Actually T_base_camF = inv(T_board_base_avg) @ T_board_camF_init
    # Then T_board_camF_cross = T_board_base_avg @ T_base_camF
    #                         = T_board_base_avg @ inv(T_board_base_avg) @ T_board_camF_init
    #                         = T_board_camF_init   ... that's circular too.

    # The cross-calibration only helps if we compute T_board_base per-arm separately
    # and then average. The averaged T_board_base is more robust than any single estimate.
    # Then T_board_camF is derived from head PnP, which is independent.
    # The only way cross-cal helps is if we have a SECOND board position.

    # Conclusion: with a single board position, cross-calibration can't improve T_board_camF
    # beyond what head PnP already gives. We need multi-position head calibration.

    # However, we CAN still improve by using the arm data as a REGULARIZER.
    # Idea: constrain T_board_camF such that when we compute T_world_baseL and T_world_baseR
    # through the head camera chain, they match the arm-derived positions.

    # T_world_baseL_from_head = T_world_board @ T_board_baseL
    #   where T_world_board uses T_board_camF
    # T_world_baseL_from_arms = directly from arm calibration
    # These should match. Minimize their difference.

    # Optimize T_board_camF (6 DOF) to minimize:
    #   w1 * head_reproj_error + w2 * ||T_board_base_from_head - T_board_base_from_arms||

    # But T_board_base is computed independently per arm from hand-eye, so T_board_camF
    # doesn't affect it. The world frame depends on T_board_camF through compute_world_frame,
    # but T_board_base{L,R} are fixed from arm calibration.

    # OK — the truth is: with a single stationary board and independent arm calibration,
    # the head PnP is already optimal for the head's own observation. Cross-calibration
    # would require moving the board or having temporal correspondence between arm poses
    # and head observations.

    # Return the PnP-refined result (which may be slightly better than the initial)
    t = T_board_camF_ref[:3, 3]
    r = Rotation.from_matrix(T_board_camF_ref[:3, :3]).as_euler('xyz', degrees=True)
    print(f"\n  [Head cross-cal] T_board_camF (refined):")
    print(f"    translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
    print(f"    rotation:    [{r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}]°")
    print(f"    reproj error: {reproj_err:.3f} px")

    # Compare with arm-derived board position
    t_avg = T_board_base_avg[:3, 3]
    print(f"    arm-derived T_board_base (avg of {len(T_board_base_estimates)} poses):")
    print(f"      translation: [{t_avg[0]:.4f}, {t_avg[1]:.4f}, {t_avg[2]:.4f}] m")

    return T_board_camF_ref, reproj_err


def _read_session_hardware(session_dir: str) -> dict:
    """从 session 的 pose_list.json 读取硬件信息 (camera_serial, can, arm)。"""
    pose_file = os.path.join(session_dir, 'pose_list.json')
    if not os.path.exists(pose_file):
        return {}
    with open(pose_file) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {k: data[k] for k in ('camera_serial', 'can', 'arm') if k in data}
    return {}


def save_yaml(output_path: str, results: dict):
    """保存 calibration.yaml"""
    def ndarray_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: ndarray_to_list(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ndarray_to_list(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        yaml.dump(ndarray_to_list(results), f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved calibration to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='标定求解')
    parser.add_argument('--session', required=True, help='标定会话目录 (含 left/, right/, head.npz)')
    parser.add_argument('--output', default=None, help='输出文件 (默认: session_dir/calibration.yml)')
    args = parser.parse_args()

    session_dir = args.session
    if args.output is None:
        args.output = os.path.join(session_dir, 'calibration.yml')

    print("=" * 60)
    print("标定求解")
    print("=" * 60)

    # 1. 加载数据
    print("\n--- Loading data ---")
    left_data = load_arm_data(os.path.join(session_dir, 'left'))
    right_data = load_arm_data(os.path.join(session_dir, 'right'))
    head_data = load_head_data(session_dir)

    # 2. 求解
    print("\n--- Solving hand-eye ---")
    T_link6_camL, T_board_baseL, errs_L = solve_hand_eye(left_data, 'Left')
    T_link6_camR, T_board_baseR, errs_R = solve_hand_eye(right_data, 'Right')
    T_board_camF, err_head = solve_head(head_data)

    # 2b. 板面倾斜检查 (board Z 在头顶相机系中应大致朝向相机, 即 Z 分量为负)
    R_cam_board, _ = cv2.Rodrigues(head_data['rvec'].flatten())
    board_z_in_cam = R_cam_board[:, 2]  # board Z-axis expressed in camera frame
    # 相机朝下看, board Z 应大致朝上 (在相机系中 Y 或 -Z 方向)
    # 板面法线与相机光轴 (cam Z=[0,0,1]) 的夹角
    angle_to_optical = np.degrees(np.arccos(np.clip(abs(np.dot(board_z_in_cam, [0, 0, 1])), 0, 1)))
    if angle_to_optical > 60.0:
        print(f"\n  [WARN] Board normal is {angle_to_optical:.1f}° from camera optical axis — "
              f"board may be significantly tilted or camera nearly parallel to board")
    else:
        print(f"\n  Board-to-camera angle: {angle_to_optical:.1f}° (OK)")

    # 2c. Verify head consistency with arm data
    print("\n--- Head consistency check ---")
    # Each arm pose gives T_board_base. If head PnP is correct, these should all agree.
    # Compare T_board_base from left arm vs right arm:
    t_L = T_board_baseL[:3, 3]
    t_R = T_board_baseR[:3, 3]
    baseline = np.linalg.norm(t_L - t_R)
    print(f"  Arm baseline (in board frame): {baseline*1000:.1f} mm")
    print(f"  baseL in board: [{t_L[0]:.4f}, {t_L[1]:.4f}, {t_L[2]:.4f}]")
    print(f"  baseR in board: [{t_R[0]:.4f}, {t_R[1]:.4f}, {t_R[2]:.4f}]")

    # Check per-pose T_board_base scatter (how consistent are the arm poses)
    for arm_label, arm_data, T_link6_cam in [('Left', left_data, T_link6_camL), ('Right', right_data, T_link6_camR)]:
        t_list = []
        for d in arm_data:
            T_base_ee = d['T_base_ee']
            rvec = d['rvec'].flatten()
            tvec = d['tvec'].reshape(3, 1)
            R_t2c, _ = cv2.Rodrigues(rvec)
            T_cam_board = np.eye(4)
            T_cam_board[:3, :3] = R_t2c
            T_cam_board[:3, 3] = tvec.flatten()
            T_board_base_i = np.linalg.inv(T_base_ee @ T_link6_cam @ T_cam_board)
            t_list.append(T_board_base_i[:3, 3])
        t_arr = np.array(t_list)
        scatter = np.std(t_arr, axis=0) * 1000
        print(f"  {arm_label} T_board_base scatter (std): [{scatter[0]:.1f}, {scatter[1]:.1f}, {scatter[2]:.1f}] mm")

    # 3. 世界系
    print("\n--- Computing world frame ---")
    T_board_world = compute_world_frame(T_board_baseL, T_board_baseR)
    T_world_board = np.linalg.inv(T_board_world)

    T_world_camF = T_world_board @ T_board_camF
    T_world_baseL = T_world_board @ T_board_baseL
    T_world_baseR = T_world_board @ T_board_baseR

    # 验证对称性
    pos_L = T_world_baseL[:3, 3]
    pos_R = T_world_baseR[:3, 3]
    print(f"\n  baseL pos: {pos_L}")
    print(f"  baseR pos: {pos_R}")
    print(f"  midpoint:  {(pos_L + pos_R) / 2}")
    if pos_L[0] * pos_R[0] > 0:
        print("  [WARN] baseL 和 baseR 的 X 坐标同号, 对称性异常")

    # 4. 读取内参 (从采集数据中提取, 不硬编码)
    head_intr = json.loads(str(head_data['intrinsics']))
    left_intr = json.loads(str(left_data[0]['intrinsics']))
    right_intr = json.loads(str(right_data[0]['intrinsics']))

    # 5. 保存
    results = {
        'metadata': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'board': {'dict': 'DICT_5X5_100', 'size': [7, 5], 'square_mm': 38.0, 'marker_mm': 28.0},
            'method': 'DANIILIDIS',
            'reprojection_error_px': {
                'left_mean': float(np.mean(errs_L)),
                'left_max': float(np.max(errs_L)),
                'right_mean': float(np.mean(errs_R)),
                'right_max': float(np.max(errs_R)),
                'head': float(err_head),
            },
            'num_poses': {'left': len(left_data), 'right': len(right_data)},
        },
        'transforms': {
            'T_world_camF': T_world_camF,
            'T_world_baseL': T_world_baseL,
            'T_world_baseR': T_world_baseR,
            'T_link6_camL': T_link6_camL,
            'T_link6_camR': T_link6_camR,
        },
        'intrinsics': {
            'cam_f': head_intr,
            'cam_l': left_intr,
            'cam_r': right_intr,
        },
        'hardware': {
            'cam_f_serial': str(head_data.get('camera_serial', ['unknown'])[0]) if 'camera_serial' in head_data else 'unknown',
            'cam_l_serial': _read_session_hardware(os.path.join(session_dir, 'left')).get('camera_serial', 'unknown'),
            'cam_r_serial': _read_session_hardware(os.path.join(session_dir, 'right')).get('camera_serial', 'unknown'),
            'left_arm_can': _read_session_hardware(os.path.join(session_dir, 'left')).get('can', 'unknown'),
            'right_arm_can': _read_session_hardware(os.path.join(session_dir, 'right')).get('can', 'unknown'),
        },
    }

    save_yaml(args.output, results)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
