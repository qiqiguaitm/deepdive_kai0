#!/usr/bin/env python3
"""ChArUco-based camera intrinsic calibration — verify phase.

Loads a session + the solved intrinsics from `config/intrinsics.yaml` and:

  1. Re-detects the board on every captured frame
  2. Reprojects corners with the calibrated K + dist; renders side-by-side
     (detected vs reprojected) overlays into `<session>/verify/<NNN>.png`
  3. Generates `<session>/verify/error_hist.png` and prints per-frame error
     stats
  4. Computes max absolute pixel shift between factory-undistorted and
     calibrated-undistorted images on a small sample — large shift (>3 px in
     corners) means the new intrinsics significantly differ from factory and
     downstream consumers that assumed factory K need refresh

Usage:
    python3 calib/verify_intrinsics.py --session top_head_intr_2026-05-08

Optional: `--camera top_head` (else read from session meta.json).
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
from intrinsics_io import get_intrinsics, k_dist_from_entry


def _spec_from_meta(meta: dict) -> BoardSpec:
    b = meta["board"]
    return BoardSpec(cols=int(b["cols"]), rows=int(b["rows"]),
                     square_mm=float(b["square_mm"]),
                     marker_mm=float(b["marker_mm"]),
                     dict_id=dict_id_from_name(str(b["dict"])))


def _draw_compare(img, detected, reprojected, idx, err):
    out = img.copy()
    for d, r in zip(detected.reshape(-1, 2), reprojected.reshape(-1, 2)):
        cv2.circle(out, tuple(d.astype(int)), 5, (0, 255, 255), 1)
        cv2.circle(out, tuple(r.astype(int)), 3, (0, 0, 255), -1)
        cv2.line(out, tuple(d.astype(int)), tuple(r.astype(int)),
                 (255, 0, 255), 1, cv2.LINE_AA)
    label = f"frame {idx}  per-corner mean = {err:.3f}px  (cyan=detected, red=reprojected)"
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _hist_png(errs: np.ndarray, out_path: Path):
    """Tiny dependency-free histogram (cv2 only — avoid matplotlib pull)."""
    h, w = 320, 640
    img = np.full((h, w, 3), 250, dtype=np.uint8)
    if len(errs) == 0:
        cv2.imwrite(str(out_path), img); return
    nbins = 24
    hist, edges = np.histogram(errs, bins=nbins)
    max_count = max(1, int(hist.max()))
    bar_w = w // nbins
    for i, c in enumerate(hist):
        bar_h = int(c / max_count * (h - 60))
        x0 = i * bar_w + 1
        y0 = h - 30 - bar_h
        cv2.rectangle(img, (x0, y0), (x0 + bar_w - 2, h - 30), (80, 140, 200), -1)
    cv2.putText(img, f"per-frame mean reproj err (px)  "
                     f"mean={errs.mean():.3f}  max={errs.max():.3f}  n={len(errs)}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, f"{edges[0]:.2f}", (4, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, f"{edges[-1]:.2f}", (w - 60, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def _undistort_drift(K_new, d_new, K_fac, d_fac, w, h) -> float:
    """Max pixel shift on a coarse grid between two undistort maps. A small
    drift (≤1 px) means consumers using factory K won't notice; a large drift
    means we changed the geometry meaningfully and dependents need re-running."""
    xs, ys = np.meshgrid(np.linspace(0, w - 1, 17), np.linspace(0, h - 1, 13))
    pts = np.stack([xs, ys], axis=-1).astype(np.float32).reshape(-1, 1, 2)
    a = cv2.undistortPoints(pts, K_fac, d_fac, P=K_fac).reshape(-1, 2)
    b = cv2.undistortPoints(pts, K_new, d_new, P=K_new).reshape(-1, 2)
    return float(np.linalg.norm(a - b, axis=1).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--data-root", default=str(THIS_DIR / "data"))
    args = ap.parse_args()

    session_dir = Path(args.data_root) / args.session
    if not session_dir.is_dir():
        raise SystemExit(f"session not found: {session_dir}")

    meta = json.loads((session_dir / "meta.json").read_text())
    role = args.camera or meta["camera_role"]
    entry = get_intrinsics(role)
    if entry is None:
        raise SystemExit(f"no intrinsics found for role {role!r}; run solve_intrinsics.py first")

    K, dist = k_dist_from_entry(entry)
    fac = meta["factory_intrinsics"]
    K_fac = np.array([[fac["fx"], 0, fac["cx"]],
                       [0, fac["fy"], fac["cy"]],
                       [0, 0, 1]], dtype=np.float64)
    d_fac = np.array(fac["dist"], dtype=np.float64).reshape(-1)
    if d_fac.shape[0] != 5:
        d_fac = np.pad(d_fac, (0, max(0, 5 - len(d_fac))))[:5]

    w, h = meta["resolution"]
    spec = _spec_from_meta(meta)
    board = get_board(spec)
    obj_all = board.getChessboardCorners()
    print(f"  board: {spec}")

    verify_dir = session_dir / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== verify intrinsics — {role}  ({entry['num_frames']} frames in DB) ===")
    print(f"  K  = fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"  dist = {dist[:5]}")
    print()

    errs = []
    for fdef in meta["frames"]:
        idx = fdef["idx"]
        img_path = session_dir / "frames" / f"{idx:03d}.png"
        det_path = session_dir / "detections" / f"{idx:03d}.json"
        if not img_path.is_file() or not det_path.is_file():
            continue
        det = json.loads(det_path.read_text())
        corners = np.asarray(det["charuco_corners"], dtype=np.float32).reshape(-1, 1, 2)
        ids = np.asarray(det["charuco_ids"], dtype=np.int32).reshape(-1)
        obj = obj_all[ids]
        ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))
        errs.append(err)
        img = cv2.imread(str(img_path))
        out = _draw_compare(img, corners, proj, idx, err)
        cv2.imwrite(str(verify_dir / f"{idx:03d}.png"), out)

    errs = np.asarray(errs)
    _hist_png(errs, verify_dir / "error_hist.png")
    drift = _undistort_drift(K, dist, K_fac, d_fac, w, h)
    print(f"per-frame mean reproj error: n={len(errs)}  mean={errs.mean():.4f}px  "
          f"max={errs.max():.4f}px")
    print(f"undistort drift vs factory  (max over 17×13 grid): {drift:.3f} px")
    if drift > 3.0:
        print("  ⚠ drift > 3 px — downstream code using factory K (rerun FK, depth)")
        print("    may need to consume config/intrinsics.yaml explicitly.")
    print(f"✓ per-frame overlays + histogram → {verify_dir}")


if __name__ == "__main__":
    main()
