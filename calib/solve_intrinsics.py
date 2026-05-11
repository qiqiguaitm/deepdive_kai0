#!/usr/bin/env python3
"""ChArUco-based camera **intrinsic** calibration — solve phase.

Consumes a session dir produced by `capture_intrinsics.py`, runs OpenCV's
`cv2.aruco.calibrateCameraCharucoExtended` with two passes:

  1. all-frames calibration → per-frame reprojection error
  2. drop top-K outliers (per-frame error above (mean + 2σ) by default),
     re-calibrate → final K, dist

Writes the result into `config/intrinsics.yaml` under the camera role and
prints a side-by-side diff with the RealSense factory K (saved at capture).

Usage:
    python3 calib/solve_intrinsics.py --session top_head_intr_2026-05-08 \\
        [--camera top_head]  [--outlier-sigma 2.0]  [--min-keep 20]

If `--camera` is omitted it is read from session meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from board_def import BoardSpec, dict_id_from_name, get_board
from intrinsics_io import (DEFAULT_INTRINSICS_PATH, now_str, save_intrinsics_yaml)


def _spec_from_meta(meta: dict) -> BoardSpec:
    b = meta["board"]
    return BoardSpec(cols=int(b["cols"]), rows=int(b["rows"]),
                     square_mm=float(b["square_mm"]),
                     marker_mm=float(b["marker_mm"]),
                     dict_id=dict_id_from_name(str(b["dict"])))


def _load_session(session_dir: Path) -> tuple[dict, list[Path], list[dict]]:
    meta = json.loads((session_dir / "meta.json").read_text())
    det_dir = session_dir / "detections"
    img_dir = session_dir / "frames"
    detections = []
    img_paths = []
    for entry in meta["frames"]:
        idx = entry["idx"]
        det_file = det_dir / f"{idx:03d}.json"
        img_file = img_dir / f"{idx:03d}.png"
        if not det_file.is_file() or not img_file.is_file():
            print(f"  skip frame {idx}: missing files")
            continue
        detections.append(json.loads(det_file.read_text()))
        img_paths.append(img_file)
    return meta, img_paths, detections


def _det_to_cv(det: dict) -> tuple[np.ndarray, np.ndarray]:
    """JSON detection → (charuco_corners[N,1,2] float32, charuco_ids[N,1] int32)
    in the shapes opencv expects."""
    corners = np.asarray(det["charuco_corners"], dtype=np.float32).reshape(-1, 1, 2)
    ids = np.asarray(det["charuco_ids"], dtype=np.int32).reshape(-1, 1)
    return corners, ids


def _per_frame_reproj_error_from_matched(K, dist, rvecs, tvecs, obj_pts_list, img_pts_list):
    """Mean reprojection error per frame from pre-matched (objp, imgp) pairs
    (output of board.matchImagePoints). Returns ndarray of length N."""
    errs = []
    for obj, img, rvec, tvec in zip(obj_pts_list, img_pts_list, rvecs, tvecs):
        img_proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        img_proj = img_proj.reshape(-1, 2)
        det = img.reshape(-1, 2)
        errs.append(float(np.mean(np.linalg.norm(img_proj - det, axis=1))))
    return np.asarray(errs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--camera", default=None,
                    help="camera role; if omitted, read from session meta.json")
    ap.add_argument("--data-root", default=str(THIS_DIR / "data"))
    ap.add_argument("--outlier-sigma", type=float, default=2.0,
                    help="drop frames whose per-frame error > mean + σ·std (set 0 to disable)")
    ap.add_argument("--min-keep", type=int, default=20,
                    help="never drop below this many frames in outlier rejection")
    ap.add_argument("--output", type=str, default=str(DEFAULT_INTRINSICS_PATH))
    args = ap.parse_args()

    session_dir = Path(args.data_root) / args.session
    if not session_dir.is_dir():
        raise SystemExit(f"session dir not found: {session_dir}")

    meta, img_paths, detections = _load_session(session_dir)
    role = args.camera or meta["camera_role"]
    w, h = meta["resolution"]
    image_size = (w, h)
    spec = _spec_from_meta(meta)
    board = get_board(spec)
    print(f"  board: {spec}")

    print(f"=== solve intrinsics — {role} ({len(detections)} frames, {w}×{h}) ===")
    if len(detections) < 8:
        raise SystemExit("need ≥ 8 frames; capture more")

    # OpenCV calibration inputs.
    # In OpenCV 4.7+ the aruco.calibrateCameraCharuco* API was removed in favor
    # of the modern workflow: board.matchImagePoints() → (objpts, imgpts) per
    # frame, then standard cv2.calibrateCamera(). This is what we use.
    corners_list = [_det_to_cv(d)[0] for d in detections]
    ids_list = [_det_to_cv(d)[1] for d in detections]

    object_points = []
    image_points = []
    for corners, ids in zip(corners_list, ids_list):
        objp, imgp = board.matchImagePoints(corners, ids)
        if objp is None or len(objp) < 4:
            continue
        object_points.append(objp.astype(np.float32))
        image_points.append(imgp.astype(np.float32))

    if len(object_points) < 8:
        raise SystemExit(f"only {len(object_points)} usable frames after matchImagePoints; "
                          f"check board geometry / detection quality")

    factory = meta["factory_intrinsics"]
    K0 = np.array([
        [factory["fx"], 0, factory["cx"]],
        [0, factory["fy"], factory["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    d0 = np.array(factory["dist"], dtype=np.float64).reshape(-1)
    if d0.shape[0] != 5:
        d0 = np.pad(d0, (0, max(0, 5 - len(d0))))[:5]

    flags = cv2.CALIB_USE_INTRINSIC_GUESS   # 5-dof Brown-Conrady (k1..k3, p1, p2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, K0.copy(), d0.copy(),
        flags=flags, criteria=criteria)
    per_frame = _per_frame_reproj_error_from_matched(
        K, dist, rvecs, tvecs, object_points, image_points)
    mean, std = float(per_frame.mean()), float(per_frame.std())
    print(f"pass 1: overall RMS={ret:.4f}px  mean={mean:.4f}px  std={std:.4f}px  "
          f"max={per_frame.max():.4f}px  n={len(per_frame)}")

    # Pass 2 — outlier rejection.
    if args.outlier_sigma > 0 and len(per_frame) > args.min_keep:
        thr = mean + args.outlier_sigma * std
        keep = per_frame <= thr
        if keep.sum() < args.min_keep:
            order = np.argsort(per_frame)
            keep = np.zeros_like(per_frame, dtype=bool)
            keep[order[: args.min_keep]] = True
        n_drop = int((~keep).sum())
        if n_drop > 0:
            objs_k = [o for o, k in zip(object_points, keep) if k]
            imgs_k = [i for i, k in zip(image_points, keep) if k]
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                objs_k, imgs_k, image_size, K.copy(), dist.copy(),
                flags=flags, criteria=criteria)
            per_frame = _per_frame_reproj_error_from_matched(
                K, dist, rvecs, tvecs, objs_k, imgs_k)
            mean = float(per_frame.mean()); std = float(per_frame.std())
            print(f"pass 2: dropped {n_drop} outliers (err > {thr:.4f}px) → "
                  f"RMS={ret:.4f}px  mean={mean:.4f}px  max={per_frame.max():.4f}px  "
                  f"kept={int(keep.sum())}")
        else:
            print(f"pass 2: no outliers (all ≤ {thr:.4f}px)")

    # Factory delta (max abs over fx/fy/cx/cy)
    delta = {
        "fx": float(K[0, 0] - factory["fx"]),
        "fy": float(K[1, 1] - factory["fy"]),
        "cx": float(K[0, 2] - factory["cx"]),
        "cy": float(K[1, 2] - factory["cy"]),
    }

    entry = {
        "method": "charuco_intrinsic",
        "resolution": [w, h],
        "board": meta["board"],   # propagate the board specs that produced this K
        "K": [[float(K[i, j]) for j in range(3)] for i in range(3)],
        "dist": [float(x) for x in dist.reshape(-1)[:5]],
        "reprojection_error_px": {
            "rms": float(ret),
            "mean": mean,
            "std": std,
            "max": float(per_frame.max()),
            "per_frame": [float(x) for x in per_frame],
        },
        "num_frames": int(len(per_frame)),
        "captured_at": now_str(),
        "session": args.session,
        "factory_K_delta_px": delta,
        "factory_K": {
            "fx": factory["fx"], "fy": factory["fy"],
            "cx": factory["cx"], "cy": factory["cy"],
            "dist": factory["dist"],
        },
    }

    path = save_intrinsics_yaml(role, entry, args.output)
    # Also save a copy alongside the session for archival.
    (session_dir / "intrinsics.json").write_text(json.dumps(entry, indent=2))

    print()
    print(f"  K =\n{K}")
    print(f"  dist = {dist.reshape(-1)[:5]}")
    print(f"  Δ from factory:  fx{delta['fx']:+.3f}  fy{delta['fy']:+.3f}  "
          f"cx{delta['cx']:+.3f}  cy{delta['cy']:+.3f}")
    print(f"✓ wrote {path}  (cameras.{role})")
    print(f"  archive: {session_dir / 'intrinsics.json'}")


if __name__ == "__main__":
    main()
