#!/usr/bin/env python3
"""Auto-identify a CharuCo / ArUco board's dictionary by trying every common
predefined ArUco dict against a live RealSense frame (or a single image
file). Reports the dict that yields the most detected markers, plus the
observed id range so you can pin down (cols × rows) and marker count.

Use this when you have an unknown board and need to figure out its specs
before running `capture_intrinsics.py`. It does NOT calibrate — it just IDs.

Usage:
    # Live mode — point at board, press Q to exit
    python3 calib/detect_board.py --camera top_head

    # File mode — analyze a saved image
    python3 calib/detect_board.py --image /path/to/board.png

Reports per dict:
    name              # e.g. DICT_5X5_100
    n_markers         # how many detected in this frame
    id_range          # [min, max] observed marker IDs
    likely_capacity   # 2 × max_id rounded → fits how many board squares

The dict with the highest n_markers + id_range that fits your board's
expected marker count (cols × rows ÷ 2 for a CharuCo board) is the winner.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from board_def import BoardSpec, dict_id_from_name, get_board
from intrinsics_io import camera_resolution, camera_serial


CANDIDATE_DICTS = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]


def _scan_image(img: np.ndarray) -> list[dict]:
    """Try every candidate dict on a single image. Returns rows sorted by
    n_markers descending."""
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    rows = []
    for name in CANDIDATE_DICTS:
        if not hasattr(cv2.aruco, name):
            continue
        d_id = getattr(cv2.aruco, name)
        ad = cv2.aruco.getPredefinedDictionary(d_id)
        det = cv2.aruco.ArucoDetector(ad, params)
        corners, ids, rejected = det.detectMarkers(img)
        n = 0 if ids is None else int(len(ids))
        id_range = None
        if ids is not None and len(ids) > 0:
            id_range = (int(ids.min()), int(ids.max()))
        rows.append({
            "dict": name,
            "n_markers": n,
            "id_range": id_range,
            "n_rejected": int(len(rejected)) if rejected is not None else 0,
        })
    rows.sort(key=lambda r: (-r["n_markers"], r["dict"]))
    return rows


def _charuco_test(img: np.ndarray, dict_name: str, cols: int, rows: int,
                  square_mm: float, marker_mm: float,
                  combos: list[tuple[int, int]] | None = None) -> list[dict]:
    """For each candidate (cols, rows) × legacy_pattern combo, count how many
    charuco corners CharucoDetector.detectBoard interpolates. Goal: tell user
    which board geometry actually matches their physical board.

    If `combos` is None, expands `(cols, rows)` to a small set of nearby
    (cols, rows) variants — covers the common label confusion where a "9×15"
    board is actually 9×14 (with one black-square row at the bottom missing
    a marker) or vice versa.

    Returns list of {combo, n_charuco, n_aruco}; n_charuco>0 means working combo."""
    dict_id = dict_id_from_name(dict_name)
    if combos is None:
        combos = set()
        for c in (cols - 1, cols, cols + 1):
            for r in (rows - 1, rows, rows + 1):
                if c >= 3 and r >= 3:
                    combos.add((c, r))
                    combos.add((r, c))  # both orientations
        combos = sorted(combos)
    results = []
    for cols_eff, rows_eff in combos:
        for legacy in (False, True):
            spec = BoardSpec(cols=cols_eff, rows=rows_eff,
                             square_mm=square_mm, marker_mm=marker_mm,
                             dict_id=dict_id, legacy_pattern=legacy)
            try:
                board = get_board(spec)
                params = cv2.aruco.DetectorParameters()
                params.adaptiveThreshWinSizeMin = 3
                params.adaptiveThreshWinSizeMax = 23
                params.adaptiveThreshWinSizeStep = 10
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
                cd = cv2.aruco.CharucoDetector(board, detectorParams=params)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                charuco_corners, charuco_ids, marker_corners, marker_ids = cd.detectBoard(gray)
                n_ch = 0 if charuco_corners is None else int(len(charuco_corners))
                n_ar = 0 if marker_ids is None else int(len(marker_ids))
                n_exp_markers = (cols_eff * rows_eff) // 2
            except Exception:
                n_ch, n_ar, n_exp_markers = -1, -1, -1
            results.append({
                "combo": f"({cols_eff}×{rows_eff}, legacy={legacy})",
                "expected_markers": n_exp_markers,
                "n_charuco": n_ch, "n_aruco": n_ar,
            })
    # Sort: n_charuco desc, then n_aruco desc
    results.sort(key=lambda r: (-r["n_charuco"], -r["n_aruco"]))
    return results


def _print_table(rows: list[dict]):
    print(f"  {'dict':<24} n_markers  id_range       n_rejected")
    print(f"  {'-' * 24} ---------  -------------  ----------")
    for r in rows[:8]:
        ir = "—" if r["id_range"] is None else f"[{r['id_range'][0]:>3d},{r['id_range'][1]:>3d}]"
        print(f"  {r['dict']:<24} {r['n_markers']:>9d}  {ir:<13}  {r['n_rejected']:>10d}")
    print()
    best = rows[0]
    if best["n_markers"] > 0:
        print(f"→ best guess: {best['dict']}  ({best['n_markers']} markers, ids {best['id_range']})")
        print()
        print(f"  for a 9×15 CharuCo board you expect ~67 markers (9*15//2 = 67).")
        print(f"  if {best['dict']} gives ~67 with id range fitting in dict size, that's your board.")
    else:
        print("→ no dict detected ANY markers — board out of focus, too far, glare, or all marker IDs")
        print("  fall outside the candidate set above. Move board closer + flatter to the camera.")


def _print_charuco_table(results: list[dict]):
    print(f"  {'combo (cols×rows, legacy)':<32}  expected_mk  n_charuco  n_aruco")
    print(f"  {'-' * 32}  -----------  ---------  -------")
    shown = 0
    for r in results:
        flag = " ← USE THIS" if r["n_charuco"] > 0 and shown == 0 else ""
        print(f"  {r['combo']:<32}  {r['expected_markers']:>11d}  "
              f"{r['n_charuco']:>9d}  {r['n_aruco']:>7d}{flag}")
        shown += 1
        if shown >= 12 and r["n_charuco"] == 0:
            print(f"  …({len(results) - shown} more combos all gave 0 charuco corners)")
            break


def _run_live(role: str, args):
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise SystemExit("pyrealsense2 not installed; activate kai0 venv")
    w, h = camera_resolution(role)
    serial = camera_serial(role)
    print(f"=== live detect on {role} ({serial}) {w}×{h} ===")
    cfg = rs.config(); cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
    pipe = rs.pipeline(); pipe.start(cfg)
    for _ in range(30): pipe.wait_for_frames()  # warm-up

    last_scan = 0.0
    rows = []
    last_dict = ""
    win = f"board detect — {role}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    last_img = None
    try:
        while True:
            try:
                fr = pipe.wait_for_frames()
                c = fr.get_color_frame()
                if not c: continue
                img = np.asanyarray(c.get_data())
                last_img = img
                now = time.monotonic()
                if now - last_scan > 1.0:   # scan once per second to keep UI responsive
                    rows = _scan_image(img)
                    last_scan = now
                    last_dict = rows[0]["dict"] if rows[0]["n_markers"] > 0 else ""
                disp = img.copy()
                y = 24
                for r in rows[:5]:
                    ir = "—" if r["id_range"] is None else f"[{r['id_range'][0]}, {r['id_range'][1]}]"
                    line = f"{r['dict']:<22} n={r['n_markers']:<3} ids={ir}"
                    cv2.putText(disp, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(disp, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255) if r["dict"] != last_dict else (0, 255, 0),
                                1, cv2.LINE_AA)
                    y += 22
                cv2.putText(disp, "Q = exit and run probe", (8, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(win, disp)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
            except KeyboardInterrupt:
                # let Ctrl+C also fall through to probe
                print("\n[detect_board] interrupted — exiting live loop, running probe…")
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
    print()
    if rows:
        _print_table(rows)
    # Optional second pass: probe charuco geometry on the LAST captured image.
    if args.charuco_probe and rows and rows[0]["n_markers"] > 0:
        dict_name = args.dict or rows[0]["dict"]
        print()
        print(f"=== charuco geometry probe (dict={dict_name}, "
              f"square={args.square_mm}mm, marker={args.marker_mm}mm) ===")
        if last_img is None:
            print("  (no frame captured during live loop — re-opening camera)")
            pipe2 = rs.pipeline()
            cfg2 = rs.config(); cfg2.enable_device(camera_serial(role))
            cfg2.enable_stream(rs.stream.color,
                               *camera_resolution(role), rs.format.bgr8, 30)
            pipe2.start(cfg2)
            for _ in range(15): pipe2.wait_for_frames()
            fr = pipe2.wait_for_frames()
            last_img = np.asanyarray(fr.get_color_frame().get_data())
            pipe2.stop()
        results = _charuco_test(last_img, dict_name, args.cols, args.rows,
                                args.square_mm, args.marker_mm)
        _print_charuco_table(results)


def _run_file(image_path: Path, args):
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"cannot read image: {image_path}")
    print(f"=== file detect on {image_path}  ({img.shape[1]}×{img.shape[0]}) ===")
    print()
    rows = _scan_image(img)
    _print_table(rows)
    if args.charuco_probe and rows and rows[0]["n_markers"] > 0:
        dict_name = args.dict or rows[0]["dict"]
        print()
        print(f"=== charuco geometry probe (dict={dict_name}, "
              f"square={args.square_mm}mm, marker={args.marker_mm}mm) ===")
        results = _charuco_test(img, dict_name, args.cols, args.rows,
                                args.square_mm, args.marker_mm)
        _print_charuco_table(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", choices=("top_head", "hand_left", "hand_right"),
                   help="live mode: open this RealSense camera + scan every 1s")
    g.add_argument("--image", type=Path, help="file mode: scan a saved image")
    ap.add_argument("--charuco-probe", action="store_true",
                    help="after dict ID, also probe (cols×rows × legacy_pattern) "
                         "to find which CharucoBoard geometry actually matches")
    ap.add_argument("--cols", type=int, default=15, help="for --charuco-probe")
    ap.add_argument("--rows", type=int, default=9, help="for --charuco-probe")
    ap.add_argument("--square-mm", type=float, default=20.0, help="for --charuco-probe")
    ap.add_argument("--marker-mm", type=float, default=15.0, help="for --charuco-probe")
    ap.add_argument("--dict", default=None,
                    help="for --charuco-probe; if omitted, uses winner from dict scan")
    args = ap.parse_args()
    if args.camera:
        _run_live(args.camera, args)
    else:
        _run_file(args.image, args)


if __name__ == "__main__":
    main()
