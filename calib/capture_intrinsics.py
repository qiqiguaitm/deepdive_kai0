#!/usr/bin/env python3
"""ChArUco-based camera **intrinsic** calibration — capture phase.

Captures N frames of the CharuCo board (board_def.py specs) from a single
RealSense camera, with live coverage feedback so the operator can ensure the
board fills every image quadrant + a range of tilt angles. Persists raw .png
frames + per-frame charuco detection JSON under a session dir, ready to be
consumed by `solve_intrinsics.py`.

Usage:
    python3 calib/capture_intrinsics.py \\
        --camera top_head --session top_head_intr_2026-05-08 \\
        [--target-frames 32] [--corners-min 12]

Key handling:
    SPACE  accept current frame (only if board fully detected + min corners
           met, and the cell it occupies has not yet been sampled enough).
    R      undo last accepted frame.
    Q/ESC  quit (saves whatever was captured so far).

Coverage policy:
    The image is split into a 4×3 grid (12 cells). The script favors capturing
    in under-represented cells — it shows the cell occupancy live and rejects
    SPACE presses that would oversample (>3 per cell unless every cell has ≥2).
    Aim: every cell has ≥1, most have ≥2.

Output layout:
    calib/data/<session>/
        meta.json                 # camera role, serial, resolution, board specs
        frames/000.png …          # raw BGR captures
        detections/000.json       # {charuco_corners, charuco_ids, n_aruco}
        coverage.png              # final cell-occupancy heatmap (for posterity)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from board_def import (BoardSpec, add_board_cli_args, detect_charuco)
from intrinsics_io import camera_resolution, camera_serial

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise SystemExit("pyrealsense2 not installed; activate the kai0 venv") from e


# ── Coverage grid ───────────────────────────────────────────────────────────
GRID_COLS = 4   # 4 columns × 3 rows = 12 cells
GRID_ROWS = 3


def _cell_of(point_xy: np.ndarray, w: int, h: int) -> tuple[int, int]:
    """(x, y) pixel → (col, row) in coverage grid."""
    cx = int(np.clip(point_xy[0] * GRID_COLS / w, 0, GRID_COLS - 1))
    cy = int(np.clip(point_xy[1] * GRID_ROWS / h, 0, GRID_ROWS - 1))
    return cx, cy


def _board_centroid_cell(charuco_corners: np.ndarray, w: int, h: int) -> tuple[int, int]:
    centroid = charuco_corners.reshape(-1, 2).mean(axis=0)
    return _cell_of(centroid, w, h)


def _render_overlay(img: np.ndarray, corners, ids, cov: Counter,
                    target: int, accepted: int, w: int, h: int,
                    last_msg: str = "", spec=None) -> np.ndarray:
    """Draw the detected corners + coverage grid + HUD onto a copy of the
    frame. Returns the annotated frame (BGR uint8)."""
    n_corners_max = spec.n_corners if spec is not None else 24
    out = img.copy()
    # grid lines
    for c in range(1, GRID_COLS):
        x = int(c * w / GRID_COLS)
        cv2.line(out, (x, 0), (x, h), (60, 60, 60), 1, cv2.LINE_AA)
    for r in range(1, GRID_ROWS):
        y = int(r * h / GRID_ROWS)
        cv2.line(out, (0, y), (w, y), (60, 60, 60), 1, cv2.LINE_AA)
    # cell occupancy heat
    for cx in range(GRID_COLS):
        for cy in range(GRID_ROWS):
            n = cov.get((cx, cy), 0)
            x0 = int(cx * w / GRID_COLS)
            y0 = int(cy * h / GRID_ROWS)
            x1 = int((cx + 1) * w / GRID_COLS)
            y1 = int((cy + 1) * h / GRID_ROWS)
            # green more saturated as samples accumulate (cap at 4)
            intensity = min(4, n) / 4.0
            color = (0, int(50 + 130 * intensity), 0)
            overlay = out.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
            cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
            cv2.putText(out, str(n), (x0 + 6, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    # detected corners
    if corners is not None and len(corners) > 0:
        cv2.aruco.drawDetectedCornersCharuco(out, corners, ids,
                                             cornerColor=(0, 255, 255))
    # HUD
    hud = [
        f"accepted={accepted}/{target}    corners={0 if corners is None else len(corners)}/{n_corners_max}",
        f"keys: SPACE accept | R undo | Q/ESC quit",
    ]
    if last_msg:
        hud.append(last_msg)
    y = 20
    for line in hud:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return out


def _save_coverage_png(cov: Counter, w: int, h: int, path: Path):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cx in range(GRID_COLS):
        for cy in range(GRID_ROWS):
            n = cov.get((cx, cy), 0)
            color = (0, int(40 + 50 * min(4, n)), 0)
            x0 = int(cx * w / GRID_COLS); y0 = int(cy * h / GRID_ROWS)
            x1 = int((cx + 1) * w / GRID_COLS); y1 = int((cy + 1) * h / GRID_ROWS)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
            cv2.putText(img, str(n), (x0 + 6, y0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def _open_camera(serial: str, w: int, h: int, fps: int = 30) -> rs.pipeline:
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    pipe = rs.pipeline()
    pipe.start(cfg)
    # warm up: drop first 30 frames so AE/WB settles
    for _ in range(30):
        pipe.wait_for_frames()
    return pipe


def _factory_intrinsics(pipe: rs.pipeline) -> dict:
    """Snapshot RealSense's factory color intrinsics for the running stream."""
    prof = pipe.get_active_profile().get_stream(rs.stream.color)
    intr = prof.as_video_stream_profile().get_intrinsics()
    return {
        "fx": float(intr.fx), "fy": float(intr.fy),
        "cx": float(intr.ppx), "cy": float(intr.ppy),
        "dist": [float(c) for c in intr.coeffs],
        "model": str(intr.model),
        "width": int(intr.width), "height": int(intr.height),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", required=True,
                    choices=("top_head", "hand_left", "hand_right"))
    ap.add_argument("--session", required=True,
                    help="session name; output goes to calib/data/<session>/")
    ap.add_argument("--target-frames", type=int, default=32)
    ap.add_argument("--corners-min", type=int, default=12,
                    help="reject frames with fewer than this many charuco corners")
    ap.add_argument("--per-cell-cap", type=int, default=3,
                    help="reject capture if its grid cell already has ≥ this many samples (until every cell has ≥2)")
    ap.add_argument("--out-root", default=str(THIS_DIR / "data"),
                    help="parent dir for the session folder")
    add_board_cli_args(ap)
    args = ap.parse_args()

    role = args.camera
    serial = camera_serial(role)
    w, h = camera_resolution(role)
    spec = BoardSpec.from_cli(args)
    session_dir = Path(args.out_root) / args.session
    frames_dir = session_dir / "frames"
    det_dir = session_dir / "detections"
    frames_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== intrinsic capture — {role} ({serial}) {w}×{h} ===")
    print(f"  board:   {spec}")
    print(f"  board physical: {spec.cols * spec.square_mm:.0f}×{spec.rows * spec.square_mm:.0f} mm")
    print(f"  session: {session_dir}")
    print(f"  target:  {args.target_frames} frames, ≥{args.corners_min} corners each")
    print()

    pipe = _open_camera(serial, w, h, fps=30)
    factory_k = _factory_intrinsics(pipe)
    print(f"  factory K: fx={factory_k['fx']:.2f} fy={factory_k['fy']:.2f} "
          f"cx={factory_k['cx']:.2f} cy={factory_k['cy']:.2f}")
    print()

    # Charuco detection uses factory K as initial guess for pose-aware refine.
    # Final intrinsics are NOT this — they come from `solve_intrinsics.py`.
    K_init = np.array([
        [factory_k["fx"], 0, factory_k["cx"]],
        [0, factory_k["fy"], factory_k["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_init = np.array(factory_k["dist"], dtype=np.float64)

    cov: Counter[tuple[int, int]] = Counter()
    accepted_frames: list[dict] = []  # {idx, cell, n_corners}
    msg = ""
    win = f"intrinsic capture — {role}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())

            ch_corners, ch_ids, _, _, reproj, n_aruco = detect_charuco(
                img, K_init, dist_init, spec)

            overlay = _render_overlay(img, ch_corners, ch_ids, cov,
                                      args.target_frames, len(accepted_frames),
                                      w, h, last_msg=msg, spec=spec)
            cv2.imshow(win, overlay)
            k = cv2.waitKey(1) & 0xFF

            if k in (27, ord("q")):
                msg = "quitting…"
                break
            if k == ord("r"):
                if accepted_frames:
                    last = accepted_frames.pop()
                    cov[last["cell"]] -= 1
                    if cov[last["cell"]] <= 0:
                        del cov[last["cell"]]
                    # delete files
                    (frames_dir / f"{last['idx']:03d}.png").unlink(missing_ok=True)
                    (det_dir / f"{last['idx']:03d}.json").unlink(missing_ok=True)
                    msg = f"undone frame {last['idx']}"
                else:
                    msg = "(no frames to undo)"
                continue
            if k == ord(" "):
                # Acceptance gate
                if ch_corners is None or len(ch_corners) < args.corners_min:
                    msg = (f"rejected: only {0 if ch_corners is None else len(ch_corners)} "
                           f"charuco corners (need ≥{args.corners_min})")
                    continue
                cell = _board_centroid_cell(ch_corners, w, h)
                min_cell_count = min(cov.values()) if len(cov) == GRID_COLS * GRID_ROWS else 0
                if cov[cell] >= args.per_cell_cap and min_cell_count >= 2:
                    msg = (f"rejected: cell {cell} already has {cov[cell]} samples — "
                           f"point at an under-sampled cell first")
                    continue
                # save
                idx = len(accepted_frames)
                cv2.imwrite(str(frames_dir / f"{idx:03d}.png"), img)
                det = {
                    "frame_index": idx,
                    "n_aruco": int(n_aruco),
                    "n_charuco": int(len(ch_corners)),
                    "charuco_ids": ch_ids.reshape(-1).tolist(),
                    "charuco_corners": ch_corners.reshape(-1, 2).tolist(),
                    "reprojection_error_init": (float(reproj)
                                                if reproj is not None else None),
                }
                (det_dir / f"{idx:03d}.json").write_text(json.dumps(det, indent=2))
                cov[cell] += 1
                accepted_frames.append({"idx": idx, "cell": cell,
                                         "n_corners": int(len(ch_corners))})
                msg = (f"accepted #{idx} in cell {cell} — "
                       f"{len(accepted_frames)}/{args.target_frames}, "
                       f"corners={len(ch_corners)}")
                if len(accepted_frames) >= args.target_frames:
                    msg += "  (target reached — press Q to finalize)"
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    # Write coverage png + meta
    _save_coverage_png(cov, w, h, session_dir / "coverage.png")
    meta = {
        "camera_role": role,
        "serial": serial,
        "resolution": [w, h],
        "board": {
            "cols": spec.cols, "rows": spec.rows,
            "square_mm": spec.square_mm, "marker_mm": spec.marker_mm,
            "dict": spec.dict_name,
        },
        "factory_intrinsics": factory_k,
        "frames": accepted_frames,
        "coverage_cells": {f"{cx},{cy}": int(n) for (cx, cy), n in cov.items()},
        "captured_with": "calib/capture_intrinsics.py",
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print()
    print(f"✓ captured {len(accepted_frames)} frames → {session_dir}")
    print(f"  coverage cells filled: {len(cov)}/{GRID_COLS * GRID_ROWS}")
    print(f"  next: python3 calib/solve_intrinsics.py --session {args.session} --camera {role}")


if __name__ == "__main__":
    main()
