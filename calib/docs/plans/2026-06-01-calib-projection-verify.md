# 标定投影验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 写一个纯离线脚本，用完整标定链把 board 角点投回 `data/calib_/` 每帧图像，量化像素误差并分层定位"不准"在内参/hand-eye/FK/世界系哪一段，产出交互式 Plotly HTML 报告。

**Architecture:** 单脚本 `calib/verify_projection.py` + 测试 `calib/test_verify_projection.py`。核心几何（投影、SE3 链路、误差）走 TDD 合成数据测试；编排与可视化用真实数据 smoke。复用 `board_def.get_board()` 取 board 3D 角点、`solve_calibration._robust_mean_se3` 做 SE3 鲁棒均值。

**Tech Stack:** Python（`kai0/.venv/bin/python`，cv2 4.11）、numpy、opencv、scipy、pyyaml、plotly。

**运行环境（重要）：** 一律用 conda 环境 **e3d**。已实测：cv2 4.11.0、scipy、pyyaml、pytest 9.0.3、**plotly 6.6.0 全部就绪**，`board_def.get_board()` 正常。系统 `/usr/bin/python3`（cv2 4.6）跑 `board_def` 会 **segfault**，禁用。下文所有命令的 `PY` 指：
```
PY=/data1/miniconda3/envs/e3d/bin/python
```

**坐标约定（已核对，勿改）：**
- `T_world_cam = T_world_base · T_base_ee · T_link6_cam`（与 `verify_calibration.py` 一致）
- 臂帧 `T_world_base`/`T_link6_cam` 按 arm 选 `baseL/camL` 或 `baseR/camR`
- 每帧 PnP：`rvec,tvec` 即 `T_cam_board`（`P_cam = R·P_board + t`）
- 反推 `T_world_board(i) = T_world_cam(i) · T_cam_board(i)`；head 用 `T_world_board = T_world_camF · T_camF_board`
- board 角点 id→坐标：`id=row*13+col`，`((col+1)·0.02,(row+1)·0.02,0)`（实测确认，104 个）

---

### Task 0: 确认 e3d 依赖（已满足，无需安装）

**Files:** 无

- [ ] **Step 1: 确认 plotly 已在 e3d**

Run: `$PY -c "import plotly,cv2,scipy,pytest; print('plotly',plotly.__version__,'cv2',cv2.__version__)"`
Expected: `plotly 6.6.0 cv2 4.11.0`。已实测满足，**无需任何安装**。

---

### Task 1: 数据加载 + board 角点编号 sanity（最关键）

先做这个：用真实数据确认"自取 board 3D 角点 + 该帧 rvec/tvec 投影" 能落回 npz 存的 `charuco_corners`，证明角点编号正确，整条链才可信。

**Files:**
- Create: `calib/verify_projection.py`
- Test: `calib/test_verify_projection.py`

- [ ] **Step 1: 写 loader 与 board 角点函数（最小实现）**

`calib/verify_projection.py`：
```python
#!/usr/bin/env python3
"""Offline projection verification for hand-eye + intrinsics calibration.

Reuses captured frames in data/calib_/ (no hardware). For each frame, projects
the known charuco board 3D corners back into the image via the full calibration
chain and compares against detected corners, then localizes error by layer.
"""
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_def import BoardSpec, get_board


@dataclass
class FrameData:
    """One captured pose. Pixel arrays are (N,2)/(N,) with matching charuco ids."""
    label: str                 # e.g. "left/pose_03", "head"
    arm: str | None            # "left" | "right" | None (head)
    rgb: np.ndarray            # (H,W,3) uint8 BGR
    K: np.ndarray              # (3,3)
    dist: np.ndarray           # (5,)
    corners_2d: np.ndarray     # (N,2) detected charuco corners
    ids: np.ndarray            # (N,) int charuco ids
    T_cam_board: np.ndarray    # (4,4) from per-frame solvePnP (rvec,tvec)
    T_base_ee: np.ndarray | None   # (4,4) FK, None for head
    pnp_err: float             # stored per-frame solvePnP residual (px)


def _rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """(rvec,tvec) -> 4x4 homogeneous T_cam_board."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def load_frame(npz_path: str, label: str, arm: str | None) -> FrameData:
    """Load one pose npz into a FrameData."""
    d = np.load(npz_path, allow_pickle=True)
    return FrameData(
        label=label,
        arm=arm,
        rgb=d["rgb_image"],
        K=d["camera_matrix"].astype(np.float64),
        dist=d["dist_coeffs"].astype(np.float64).reshape(-1),
        corners_2d=d["charuco_corners"].reshape(-1, 2).astype(np.float64),
        ids=d["charuco_ids"].reshape(-1).astype(int),
        T_cam_board=_rt_to_T(d["rvec"], d["tvec"]),
        T_base_ee=(d["T_base_ee"].astype(np.float64) if "T_base_ee" in d.files else None),
        pnp_err=float(np.asarray(d["reproj_err"]).reshape(-1)[0]),
    )


def load_session(session_dir: str) -> dict:
    """Load all left/right pose npz + head.npz from a calib session dir."""
    frames: list[FrameData] = []
    for arm in ("left", "right"):
        adir = os.path.join(session_dir, arm)
        if not os.path.isdir(adir):
            continue
        for fn in sorted(f for f in os.listdir(adir) if f.startswith("pose_") and f.endswith(".npz")):
            frames.append(load_frame(os.path.join(adir, fn), f"{arm}/{fn[:-4]}", arm))
    head = None
    head_path = os.path.join(session_dir, "head.npz")
    if os.path.exists(head_path):
        head = load_frame(head_path, "head", None)
    return {"frames": frames, "head": head}


def board_corners_3d(board, ids: np.ndarray) -> np.ndarray:
    """Charuco board interior corners (meters, board frame) for given ids -> (N,3)."""
    all_corners = np.array(board.getChessboardCorners(), dtype=np.float64)
    return all_corners[np.asarray(ids).reshape(-1)]
```

- [ ] **Step 2: 写 sanity 测试（真实数据，先跑应通过）**

`calib/test_verify_projection.py`：
```python
import os
import numpy as np
import cv2
import pytest

import verify_projection as vp
from board_def import BoardSpec, get_board

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION = os.path.join(HERE, "data", "calib_")
BOARD_YAML = os.path.join(HERE, "board_9x14.yaml")


@pytest.fixture(scope="module")
def board():
    return get_board(BoardSpec.from_yaml(BOARD_YAML))


def test_board_corner_ids_match_detection(board):
    """Self-取 3D 角点 + 该帧 PnP 投影,应落回 npz 存的检测角点 ~PnP 残差级别.

    证明 board 角点编号与采集时一致——整条验证链的前提.
    """
    fr = vp.load_frame(os.path.join(SESSION, "left", "pose_01.npz"), "left/pose_01", "left")
    P_board = vp.board_corners_3d(board, fr.ids)            # (N,3)
    rvec, _ = cv2.Rodrigues(fr.T_cam_board[:3, :3])
    tvec = fr.T_cam_board[:3, 3]
    proj, _ = cv2.projectPoints(P_board, rvec, tvec, fr.K, fr.dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - fr.corners_2d, axis=1)
    assert err.mean() < 1.0, f"mean reproj {err.mean():.3f}px — board id 约定可能不符"
```

- [ ] **Step 3: 跑测试验证通过**

Run: `cd /data1/tim/workspace/deepdive_kai0/calib && $PY -m pytest test_verify_projection.py -v`
Expected: `test_board_corner_ids_match_detection` PASS（mean err 应 ~0.1px）。
若 FAIL：board id 约定不符，停下排查，勿继续。

- [ ] **Step 4: Commit**

```bash
git add calib/verify_projection.py calib/test_verify_projection.py
git commit -m "feat(calib): 投影验证 loader + board 角点编号 sanity"
```
（注：若 calib 不在版本控制内，`git add` 会报 ignored；届时改为提交到可写位置或跳过 commit，先与用户确认。）

---

### Task 2: 世界系链路 + 投影核心

**Files:**
- Modify: `calib/verify_projection.py`
- Test: `calib/test_verify_projection.py`

- [ ] **Step 1: 写失败测试（合成针孔 + 链路 roundtrip）**

追加到 `test_verify_projection.py`：
```python
def _synthetic_T(tx, ty, tz, rx=0, ry=0, rz=0):
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
    return T


def test_project_world_pinhole():
    """无畸变针孔: 相机原点看 z=1m 平面上的点,投影落在预期像素."""
    K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
    dist = np.zeros(5)
    T_world_cam = np.eye(4)                      # cam == world
    P_world = np.array([[0, 0, 1.0], [0.1, 0, 1.0]])  # 光轴上 + 右移 0.1m
    px = vp.project_world_to_pixels(P_world, T_world_cam, K, dist)
    assert np.allclose(px[0], [320, 240], atol=1e-6)
    assert np.allclose(px[1], [320 + 600 * 0.1, 240], atol=1e-6)


def test_arm_chain_roundtrip_zero_error():
    """已知 base/ee/link6_cam/board -> 真值相机位姿投影得'检测',链路预测应零误差."""
    K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
    dist = np.zeros(5)
    T_world_base = _synthetic_T(0.3, 0, 0)
    T_base_ee = _synthetic_T(0.2, 0.1, 0.4, rz=0.3)
    T_link6_cam = _synthetic_T(-0.08, 0, 0.04, rx=1.5)
    T_world_board = _synthetic_T(0.0, 0.0, 0.5)
    P_board = np.array([[0.02, 0.02, 0], [0.26, 0.16, 0], [0.1, 0.08, 0]])

    T_world_cam = vp.arm_cam_pose_world(T_world_base, T_base_ee, T_link6_cam)
    P_world = (T_world_board @ np.c_[P_board, np.ones(len(P_board))].T).T[:, :3]
    det = vp.project_world_to_pixels(P_world, T_world_cam, K, dist)
    pred = vp.project_world_to_pixels(P_world, T_world_cam, K, dist)
    assert np.allclose(det, pred, atol=1e-9)
    # 反推 T_world_board 应还原真值
    T_cam_board = np.linalg.inv(T_world_cam) @ T_world_board
    T_rec = T_world_cam @ T_cam_board
    assert np.allclose(T_rec, T_world_board, atol=1e-9)
```

- [ ] **Step 2: 跑测试验证失败**

Run: `cd /data1/tim/workspace/deepdive_kai0/calib && $PY -m pytest test_verify_projection.py -k "pinhole or roundtrip" -v`
Expected: FAIL（`project_world_to_pixels` / `arm_cam_pose_world` 未定义）。

- [ ] **Step 3: 实现链路与投影函数**

追加到 `verify_projection.py`：
```python
def arm_cam_pose_world(T_world_base, T_base_ee, T_link6_cam) -> np.ndarray:
    """Full hand-eye chain -> camera pose in world (4x4)."""
    return T_world_base @ T_base_ee @ T_link6_cam


def project_world_to_pixels(P_world, T_world_cam, K, dist) -> np.ndarray:
    """Project world 3D points into a camera image -> (N,2) pixels (with distortion)."""
    P_world = np.asarray(P_world, dtype=np.float64).reshape(-1, 3)
    T_cam_world = np.linalg.inv(T_world_cam)
    P_cam = (T_cam_world @ np.c_[P_world, np.ones(len(P_world))].T).T[:, :3]
    px, _ = cv2.projectPoints(P_cam, np.zeros(3), np.zeros(3), K, dist)
    return px.reshape(-1, 2)


def frame_T_world_board(fr: FrameData, T_world_base, T_link6_cam) -> np.ndarray:
    """反推该臂帧观测到的 board 世界位姿 T_world_board(i)."""
    T_world_cam = arm_cam_pose_world(T_world_base, fr.T_base_ee, T_link6_cam)
    return T_world_cam @ fr.T_cam_board


def head_T_world_board(fr: FrameData, T_world_camF) -> np.ndarray:
    """Head 帧观测到的 board 世界位姿."""
    return T_world_camF @ fr.T_cam_board
```

- [ ] **Step 4: 跑测试验证通过**

Run: `cd /data1/tim/workspace/deepdive_kai0/calib && $PY -m pytest test_verify_projection.py -k "pinhole or roundtrip" -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add calib/verify_projection.py calib/test_verify_projection.py
git commit -m "feat(calib): 世界系 hand-eye 链路 + 投影核心"
```

---

### Task 3: SE3 一致性度量

**Files:**
- Modify: `calib/verify_projection.py`
- Test: `calib/test_verify_projection.py`

- [ ] **Step 1: 写失败测试**

追加：
```python
def test_se3_spread_zero_for_identical():
    T = _synthetic_T(0.1, 0.2, 0.3, rz=0.5)
    tr_mm, rot_deg = vp.se3_spread([T.copy() for _ in range(5)])
    assert tr_mm < 1e-6 and rot_deg < 1e-6


def test_se3_spread_detects_offset():
    Ts = [_synthetic_T(0.1, 0, 0), _synthetic_T(0.1 + 0.01, 0, 0)]  # 差 10mm
    tr_mm, _ = vp.se3_spread(Ts)
    assert 3 < tr_mm < 8   # 关于均值的 std,两点对称 -> ~5mm
```

- [ ] **Step 2: 跑测试验证失败**

Run: `cd /data1/tim/workspace/deepdive_kai0/calib && $PY -m pytest test_verify_projection.py -k se3_spread -v`
Expected: FAIL（`se3_spread` 未定义）。

- [ ] **Step 3: 实现（复用 solve_calibration 的鲁棒均值）**

追加到 `verify_projection.py`（顶部 import 区加 `from scipy.spatial.transform import Rotation`，并 `import solve_calibration`）：
```python
import solve_calibration  # for _robust_mean_se3
from scipy.spatial.transform import Rotation


def se3_spread(T_list: list[np.ndarray]) -> tuple[float, float]:
    """Spread of a set of SE3 poses about their robust mean.

    Returns (translation_std_mm, rotation_std_deg).
    """
    T_mean = solve_calibration._robust_mean_se3(T_list)
    t_dev = [np.linalg.norm(T[:3, 3] - T_mean[:3, 3]) for T in T_list]
    R_mean_inv = T_mean[:3, :3].T
    r_dev = [np.linalg.norm(Rotation.from_matrix(R_mean_inv @ T[:3, :3]).as_rotvec())
             for T in T_list]
    return float(np.sqrt(np.mean(np.square(t_dev))) * 1000.0), \
        float(np.degrees(np.sqrt(np.mean(np.square(r_dev)))))
```

- [ ] **Step 4: 跑测试验证通过**

Run: `cd /data1/tim/workspace/deepdive_kai0/calib && $PY -m pytest test_verify_projection.py -k se3_spread -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add calib/verify_projection.py calib/test_verify_projection.py
git commit -m "feat(calib): SE3 一致性度量 se3_spread"
```

---

### Task 4: 分层诊断编排（L0–L3）+ report.json

**Files:**
- Modify: `calib/verify_projection.py`
- Test: `calib/test_verify_projection.py`

- [ ] **Step 1: 写失败测试（真实 session,跑通 + 阈值断言）**

追加：
```python
def test_run_layers_on_real_session(board):
    calib = vp.load_calibration(os.path.join(SESSION, "calibration.yml"))
    session = vp.load_session(SESSION)
    rep = vp.run_layers(session, calib, board)
    # 结构完整
    for k in ("L0_intrinsics", "L1_handeye", "L2_reproj", "L3_cross"):
        assert k in rep
    # 已知 PnP ~0.12px,内参层必过
    assert rep["L0_intrinsics"]["pnp_err_mean_px"] < 0.5
    # 每帧全链路误差有值
    assert rep["L2_reproj"]["err_mean_px"] >= 0
```

- [ ] **Step 2: 跑测试验证失败**

Run: `cd /data1/tim/workspace/deepdive_kai0/calib && $PY -m pytest test_verify_projection.py -k run_layers -v`
Expected: FAIL（`load_calibration` / `run_layers` 未定义）。

- [ ] **Step 3: 实现编排**

追加到 `verify_projection.py`：
```python
def load_calibration(path: str) -> dict:
    """Load calibration.yml transforms (list->ndarray) + intrinsics."""
    with open(path) as f:
        data = yaml.safe_load(f)
    for k in data["transforms"]:
        data["transforms"][k] = np.array(data["transforms"][k], dtype=np.float64)
    return data


def _arm_extrinsics(calib: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
    t = calib["transforms"]
    if arm == "left":
        return t["T_world_baseL"], t["T_link6_camL"]
    return t["T_world_baseR"], t["T_link6_camR"]


def run_layers(session: dict, calib: dict, board) -> dict:
    """Run L0–L3 diagnostics. Returns a JSON-serializable report dict.

    Also stashes per-frame projections under rep['_frames'] for visualization
    (not serialized to report.json by the caller).
    """
    frames = session["frames"]
    head = session["head"]
    t = calib["transforms"]

    # --- 反推每帧 T_world_board + 全链路重投影 ---
    world_boards: list[np.ndarray] = []
    per_frame = []
    for fr in frames:
        T_world_base, T_link6_cam = _arm_extrinsics(calib, fr.arm)
        T_world_board_i = frame_T_world_board(fr, T_world_base, T_link6_cam)
        world_boards.append(T_world_board_i)
        per_frame.append((fr, arm_cam_pose_world(T_world_base, fr.T_base_ee, T_link6_cam)))
    head_world_board = head_T_world_board(head, t["T_world_camF"]) if head else None

    # 参考板位姿: 所有臂帧反推的鲁棒均值
    T_ref = solve_calibration._robust_mean_se3(world_boards)

    # L0: 内参 (PnP 残差汇总)
    pnp = np.array([fr.pnp_err for fr in frames] + ([head.pnp_err] if head else []))

    # L1: hand-eye 自洽 (反推 T_world_board 的离散度)
    tr_mm, rot_deg = se3_spread(world_boards)

    # L2: 全链路重投影 (用参考板位姿投回每帧)
    all_err = []
    frame_viz = []
    iter_list = list(per_frame) + ([(head, t["T_world_camF"])] if head else [])
    for fr, T_world_cam in iter_list:
        P_board = board_corners_3d(board, fr.ids)
        P_world = (T_ref @ np.c_[P_board, np.ones(len(P_board))].T).T[:, :3]
        pred = project_world_to_pixels(P_world, T_world_cam, fr.K, fr.dist)
        err = np.linalg.norm(pred - fr.corners_2d, axis=1)
        all_err.append(err)
        frame_viz.append({"label": fr.label, "det": fr.corners_2d, "pred": pred,
                          "err_mean": float(err.mean()), "rgb": fr.rgb})
    flat = np.concatenate(all_err)

    # L3: 跨相机/世界系
    posL = t["T_world_baseL"][:3, 3]; posR = t["T_world_baseR"][:3, 3]
    sym_mm = float(np.linalg.norm((posL + posR) / 2.0) * 1000.0)
    head_arm_mm = (float(np.linalg.norm(head_world_board[:3, 3] - T_ref[:3, 3]) * 1000.0)
                   if head_world_board is not None else None)

    rep = {
        "session": None,  # caller (main) fills the real path
        "n_frames": len(frames) + (1 if head else 0),
        "L0_intrinsics": {"pnp_err_mean_px": float(pnp.mean()),
                          "pnp_err_max_px": float(pnp.max()),
                          "pass": bool(pnp.mean() < 0.5)},
        "L1_handeye": {"world_board_trans_std_mm": tr_mm,
                       "world_board_rot_std_deg": rot_deg,
                       "pass": bool(tr_mm < 3.0 and rot_deg < 0.3)},
        "L2_reproj": {"err_mean_px": float(flat.mean()),
                      "err_p95_px": float(np.percentile(flat, 95)),
                      "err_max_px": float(flat.max()),
                      "pass": bool(flat.mean() < 2.0 and np.percentile(flat, 95) < 5.0)},
        "L3_cross": {"base_symmetry_mm": sym_mm,
                     "head_vs_arm_board_mm": head_arm_mm,
                     "pass": bool(sym_mm < 5.0 and (head_arm_mm is None or head_arm_mm < 10.0))},
    }
    rep["_frames"] = frame_viz
    rep["_world_boards"] = world_boards
    rep["_T_ref"] = T_ref
    rep["_calib"] = calib
    return rep


def verdict(rep: dict) -> str:
    """One-line localization conclusion based on which layer first fails."""
    if not rep["L0_intrinsics"]["pass"]:
        return "内参/板检测异常 (L0): 重新标定内参"
    if not rep["L1_handeye"]["pass"]:
        return "外参不自洽 (L1): 嫌疑 hand-eye / FK / base-in-board"
    if not rep["L3_cross"]["pass"]:
        return "跨相机/世界系不一致 (L3): 嫌疑 T_world_camF / 世界系定义"
    if not rep["L2_reproj"]["pass"]:
        return "全链路重投影偏大 (L2) 但各层自洽: 嫌疑 FK/URDF 与部署不匹配"
    return "全部通过: 标定在采集数据上自洽且重投影良好"
```

- [ ] **Step 4: 跑测试验证通过**

Run: `cd /data1/tim/workspace/deepdive_kai0/calib && $PY -m pytest test_verify_projection.py -k run_layers -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add calib/verify_projection.py calib/test_verify_projection.py
git commit -m "feat(calib): L0-L3 分层诊断编排 + 定位结论"
```

---

### Task 5: Plotly HTML 可视化 + CLI main

**Files:**
- Modify: `calib/verify_projection.py`

无单测（可视化）；用真实数据 smoke run 验证产物存在。

- [ ] **Step 1: 实现叠加图 + 3D + HTML + main**

追加到 `verify_projection.py`（顶部加 `import json`, `import base64`, `import argparse`；plotly 在函数内 import，缺失时给清晰报错）：
```python
def _overlay_png_b64(rgb, det, pred) -> str:
    """Draw det(green)/pred(red)+连线 on RGB, return base64 PNG."""
    img = rgb.copy()
    for (du, dv), (pu, pv) in zip(det.astype(int), pred.astype(int)):
        cv2.line(img, (du, dv), (pu, pv), (0, 255, 255), 1)
        cv2.circle(img, (du, dv), 3, (0, 255, 0), -1)   # detected green
        cv2.circle(img, (pu, pv), 3, (0, 0, 255), -1)   # predicted red
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


def build_report_html(rep: dict, out_html: str) -> None:
    """Assemble the self-contained Plotly HTML report."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 1) 3D world-frame consistency: 每帧反推 board 角点 + base/cam 位姿
    board = get_board(BoardSpec.from_yaml(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "board_9x14.yaml")))
    all_corners = np.array(board.getChessboardCorners(), dtype=np.float64)
    fig3d = go.Figure()
    for Twb in rep["_world_boards"]:
        Pw = (Twb @ np.c_[all_corners, np.ones(len(all_corners))].T).T[:, :3]
        fig3d.add_trace(go.Scatter3d(x=Pw[:, 0], y=Pw[:, 1], z=Pw[:, 2],
                                     mode="markers", marker=dict(size=1.5),
                                     opacity=0.4, showlegend=False))
    t = rep["_calib"]["transforms"]
    for name, T in (("baseL", t["T_world_baseL"]), ("baseR", t["T_world_baseR"]),
                    ("camF", t["T_world_camF"])):
        p = T[:3, 3]
        fig3d.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text",
                                     marker=dict(size=5), text=[name], name=name))
    fig3d.update_layout(scene=dict(aspectmode="data"), title="世界系一致性 (board 角点应聚拢)")

    # 2) 每帧误差柱状
    labels = [f["label"] for f in rep["_frames"]]
    errs = [f["err_mean"] for f in rep["_frames"]]
    figbar = go.Figure(go.Bar(x=labels, y=errs))
    figbar.update_layout(title="每帧全链路重投影误差 (px)", xaxis_tickangle=-60)

    # 3) 叠加图 (base64 内嵌)
    overlay_html = ""
    for f in rep["_frames"]:
        b64 = _overlay_png_b64(f["rgb"], f["det"], f["pred"])
        overlay_html += (f'<div style="display:inline-block;margin:4px;text-align:center">'
                         f'<img src="data:image/png;base64,{b64}" width="320"><br>'
                         f'<small>{f["label"]} — {f["err_mean"]:.2f}px</small></div>')

    # 报告头
    head_html = (f'<h2>标定投影验证报告</h2><p><b>结论:</b> {verdict(rep)}</p>'
                 f'<pre>{json.dumps({k: v for k, v in rep.items() if not k.startswith("_")}, ensure_ascii=False, indent=2)}</pre>')

    with open(out_html, "w") as fp:
        fp.write("<html><head><meta charset='utf-8'></head><body>")
        fp.write(head_html)
        fp.write(fig3d.to_html(full_html=False, include_plotlyjs="cdn"))
        fp.write(figbar.to_html(full_html=False, include_plotlyjs=False))
        fp.write("<h3>叠加图 (绿=检测, 红=标定预测)</h3>")
        fp.write(overlay_html)
        fp.write("</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description="离线标定投影验证")
    ap.add_argument("--session", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "calib_"))
    ap.add_argument("--board", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "board_9x14.yaml"))
    args = ap.parse_args()

    board = get_board(BoardSpec.from_yaml(args.board))
    calib = load_calibration(os.path.join(args.session, "calibration.yml"))
    session = load_session(args.session)
    rep = run_layers(session, calib, board)

    out_dir = os.path.join(args.session, "verify")
    os.makedirs(out_dir, exist_ok=True)
    clean = {k: v for k, v in rep.items() if not k.startswith("_")}
    clean["session"] = os.path.abspath(args.session)
    clean["verdict"] = verdict(rep)
    with open(os.path.join(out_dir, "report.json"), "w") as fp:
        json.dump(clean, fp, ensure_ascii=False, indent=2)
    build_report_html(rep, os.path.join(out_dir, "verify_report.html"))

    print("=" * 60)
    print(f"结论: {verdict(rep)}")
    for k in ("L0_intrinsics", "L1_handeye", "L2_reproj", "L3_cross"):
        print(f"  {k}: {'PASS' if rep[k]['pass'] else 'FAIL'}  {clean[k]}")
    print(f"报告: {os.path.join(out_dir, 'verify_report.html')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 真实数据 smoke run**

Run: `cd /data1/tim/workspace/deepdive_kai0/calib && $PY verify_projection.py`
Expected: 打印结论 + 4 层 PASS/FAIL；生成 `data/calib_/verify/verify_report.html` 与 `report.json`。

- [ ] **Step 3: 验证产物**

Run: `ls -la /data1/tim/workspace/deepdive_kai0/calib/data/calib_/verify/`
Expected: `verify_report.html`（应 > 100KB，含内嵌图）、`report.json` 存在。
浏览器打开 html 应见 3D 散点 + 柱状 + 叠加图。

- [ ] **Step 4: 全部测试回归**

Run: `cd /data1/tim/workspace/deepdive_kai0/calib && $PY -m pytest test_verify_projection.py -v`
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add calib/verify_projection.py
git commit -m "feat(calib): Plotly HTML 报告 (3D 世界系/误差/叠加图) + CLI"
```

---

## 验收

- `$PY -m pytest calib/test_verify_projection.py -v` 全绿（含真实数据 board id sanity）
- `data/calib_/verify/verify_report.html` 浏览器可打开,含可旋转 3D + 叠加图 + 误差柱状
- `report.json` 给出 L0–L3 数字 + 一句定位结论
- 现有 `verify_calibration.py` 未改动

## 风险/注意

- **环境**: 用 e3d (plotly/cv2/pytest 全就绪,无需装包)。勿用系统 python3 (cv2 4.6 segfault)。
- **git 归属**: `docs/` 属 tim 不可写,计划/spec 放在 `calib/docs/`;若 `calib/` 不在版本控制内,commit 步骤需与用户确认替代方案。
- **结论解读**: 标定是从本 session 解出的,L2 在训练帧上偏小是预期;若 L0/L1/L3 全过但下游仍报不准,最可能是 **FK/URDF 或坐标约定在部署侧与标定侧不一致**——这超出本脚本范围,需另查部署代码。
