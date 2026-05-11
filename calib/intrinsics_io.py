"""Shared helpers for camera-intrinsics I/O and the per-camera RealSense
serial lookup. Keeps capture / solve / verify scripts thin and consistent.

Output canonical file: `config/intrinsics.yaml`. One block per camera role
(top_head / hand_left / hand_right). Schema:

    cameras:
      top_head:
        method: charuco_intrinsic
        resolution: [640, 480]
        K:                # 3×3 row-major
          - [fx, 0,  cx]
          - [0,  fy, cy]
          - [0,  0,  1 ]
        dist: [k1, k2, p1, p2, k3]   # OpenCV plumb-bob, 5 dofs
        reprojection_error_px:
          mean: 0.18
          max:  0.45
        num_frames: 32
        captured_at: '2026-05-08 22:14'
        session: top_head_intr_2026-05-08
        factory_K_delta_px:           # max ‖K - K_factory‖ over fx/fy/cx/cy
          fx: -1.32
          fy: -0.96
          cx: +0.41
          cy: +0.18

The `factory_K_delta_px` block makes regressions obvious — if a recalibration
shifts cx/cy by tens of pixels, something's wrong with the procedure (skew,
small sample, all-corners-in-center). Should normally be ≤ a few px.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_INTRINSICS_PATH = CONFIG_DIR / "intrinsics.yaml"
CAMERAS_YML = CONFIG_DIR / "cameras.yml"


def load_cameras_yml() -> dict:
    with open(CAMERAS_YML) as f:
        return yaml.safe_load(f)


def camera_serial(role: str) -> str:
    cams = load_cameras_yml()["cameras"]
    if role not in cams:
        raise KeyError(f"unknown camera role {role!r}; available: {list(cams)}")
    return str(cams[role]["serial_number"])


def camera_resolution(role: str) -> tuple[int, int]:
    """Returns (W, H) from cameras.yml."""
    cams = load_cameras_yml()["cameras"]
    w, h = (int(x) for x in cams[role]["resolution"].lower().split("x"))
    return w, h


def load_intrinsics_yaml(path: Path | str = DEFAULT_INTRINSICS_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"cameras": {}}
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    d.setdefault("cameras", {})
    return d


def save_intrinsics_yaml(role: str, entry: dict, path: Path | str = DEFAULT_INTRINSICS_PATH) -> Path:
    """Merge `entry` under cameras[role] and rewrite the file (preserving other
    roles untouched). Creates parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = load_intrinsics_yaml(path)
    cur["cameras"][role] = entry
    with open(path, "w") as f:
        yaml.safe_dump(cur, f, default_flow_style=None, sort_keys=False)
    return path


def get_intrinsics(role: str, path: Path | str = DEFAULT_INTRINSICS_PATH) -> dict | None:
    """Returns the cameras[role] entry or None if absent."""
    cur = load_intrinsics_yaml(path)
    return cur["cameras"].get(role)


def k_dist_from_entry(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """yaml entry → (K 3×3, dist 1×5) ndarray for cv2 routines."""
    K = np.array(entry["K"], dtype=np.float64)
    dist = np.array(entry["dist"], dtype=np.float64).reshape(-1)
    return K, dist


def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
