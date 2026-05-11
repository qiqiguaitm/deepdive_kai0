# Camera Intrinsic Calibration (CharuCo) — Procedure

> Goal: solve each camera's pinhole + Brown-Conrady model
> `(fx, fy, cx, cy, k1, k2, p1, p2, k3)` from CharuCo board observations,
> replacing the RealSense factory K (which is typically accurate within
> ~1-2 px but worth tightening for high-precision tasks).

## 0. What this framework does (and doesn't)

This is the **intrinsic** (in-camera-geometry) calibration pipeline. It is
**separate** from the existing hand-eye + head-pose extrinsic pipeline
(`capture_handeye.py` / `solve_calibration.py`).

The two are orthogonal and the order matters: **always update intrinsics
first**, then re-run extrinsic calibration if you care about millimeter-scale
end-effector pose accuracy. Extrinsic solve consumes the K from
`config/intrinsics.yaml` if present, else falls back to RealSense factory K.

Supports all three RealSense cameras (`top_head` D435, `hand_left` /
`hand_right` D405). Same scripts, just change `--camera`.

## 1. Hardware setup

- CharuCo board: **7×5 squares, 38 mm square, 28 mm marker, dict
  `DICT_5X5_100`** (same as the hand-eye board; specs in
  `calib/board_def.py`). Print A4 at 100 % scale, glue to a rigid flat
  panel (foamboard or 3 mm acrylic). **Verify** square size with a caliper
  before use — a 1 % print scaling error puts every focal length ~1 % off.
- Cameras: ensure the camera you're calibrating is the only RealSense
  plugged in, or use the per-camera serial (handled automatically by
  reading `config/cameras.yml`).
- Lighting: even, diffuse, no glare on the board. AE/WB should settle
  (the capture script discards the first 30 frames for warm-up).

## 2. Three-phase workflow

```
  ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
  │  capture_intrinsics  │ → │  solve_intrinsics    │ → │  verify_intrinsics   │
  │  (interactive)       │   │  (batch, ~5 s)       │   │  (per-frame overlay) │
  └──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

All three scripts live in `calib/`. Each one's `--help` documents its full
flag set.

## 3. Phase 1 — capture (`capture_intrinsics.py`)

```bash
python3 calib/capture_intrinsics.py \
    --camera top_head \
    --session top_head_intr_2026-05-08 \
    --target-frames 32
```

Live window opens. The HUD shows:
- a 4×3 cell occupancy grid (cells get greener as samples accumulate)
- detected ChArUco corners (yellow circles)
- current corner count, accepted count

Operator workflow:
1. Hold the board so it fills ~25-60 % of the image.
2. Move it through **every cell of the 4×3 grid**, varying:
   - tilt (yaw ±30°, pitch ±30°, roll ±20°)
   - distance (near: ~30 cm, far: ~1 m)
   - rotation in image (board ±90° rotated)
3. Press **SPACE** to accept a frame. The script rejects:
   - frames with < 12 ChArUco corners visible (raise `--corners-min` to
     enforce higher quality)
   - capture in a cell that already has ≥3 samples *until* every cell has
     at least 2 (forces good coverage before oversampling)
4. Press **R** to undo the last accepted frame.
5. Press **Q** / ESC to finalize.

Aim: **30-40 frames, every cell ≥1 (most ≥2), no missing corners**. Too
few or too uniform → unstable distortion estimate (k3 can blow up).

Output: `calib/data/<session>/`
- `frames/000.png` … (raw BGR)
- `detections/000.json` (charuco corners + ids)
- `coverage.png` (final occupancy snapshot)
- `meta.json` (serial, resolution, factory K, accepted frame list)

## 4. Phase 2 — solve (`solve_intrinsics.py`)

```bash
python3 calib/solve_intrinsics.py \
    --session top_head_intr_2026-05-08
```

Two-pass calibration:
1. all frames → record per-frame reprojection error
2. drop outliers (default: error > mean + 2σ; never drop below 20 frames)
   → re-solve

Prints:
- overall RMS, mean, max reprojection error (both passes)
- final K matrix + 5-dof distortion `[k1, k2, p1, p2, k3]`
- Δ vs factory K in pixels

Writes:
- `config/intrinsics.yaml` (canonical, merged under `cameras.top_head`)
- `calib/data/<session>/intrinsics.json` (archival)

Reasonable acceptance targets:
| metric | OK | great | bad |
|---|---|---|---|
| RMS reproj err | ≤ 0.40 px | ≤ 0.25 px | > 0.6 px |
| max reproj err | ≤ 0.80 px | ≤ 0.50 px | > 1.5 px |
| Δfx, Δfy vs factory | < 5 px | < 2 px | > 15 px |
| Δcx, Δcy vs factory | < 3 px | < 1 px | > 8 px |

If "bad": more frames, better coverage, check board print size with caliper.

## 5. Phase 3 — verify (`verify_intrinsics.py`)

```bash
python3 calib/verify_intrinsics.py \
    --session top_head_intr_2026-05-08
```

For each captured frame, runs `cv2.solvePnP` with the new K and renders a
side-by-side overlay of detected (cyan circles) vs reprojected (red dots)
corners with magenta connecting lines into `<session>/verify/<NNN>.png`.

Also emits:
- `error_hist.png` — histogram of per-frame mean reprojection error
- console: `undistort drift vs factory` — max pixel shift on a 17×13
  sample grid between factory and new undistort map. **>3 px** is a hint
  that downstream consumers (rerun_viz_node depth back-projection,
  hand-eye solve) must be re-run to stay coherent.

Spot-check 3-5 frames: red dots should sit visually on top of cyan circles
across the full image. Systematic radial offset in the corners → distortion
is undercaptured; capture more frames with the board at the image edges.

## 6. Output schema (`config/intrinsics.yaml`)

See top of `calib/intrinsics_io.py` for the canonical schema. Brief:

```yaml
cameras:
  top_head:
    method: charuco_intrinsic
    resolution: [640, 480]
    K:    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    dist: [k1, k2, p1, p2, k3]
    reprojection_error_px:
      rms:  0.21
      mean: 0.18
      max:  0.42
      std:  0.07
    num_frames: 32
    captured_at: '2026-05-08 22:14'
    session: top_head_intr_2026-05-08
    factory_K_delta_px:
      fx: -1.32  # new - factory
      fy: -0.96
      cx: +0.41
      cy: +0.18
    factory_K:        # snapshot for diff
      fx: 605.42
      fy: 605.42
      cx: 318.13
      cy: 245.71
      dist: [0.0, 0.0, 0.0, 0.0, 0.0]
```

Helpers `intrinsics_io.get_intrinsics(role)` and `k_dist_from_entry(entry)`
load these in scripts.

## 7. What changes downstream

The intrinsics produced here are not auto-consumed by any node yet — the
existing extrinsic + ROS2 inference pipeline reads RealSense factory K
directly from `pyrealsense2.intrinsics`. **By design**: changing K silently
could break rerun_viz depth projection and hand-eye numbers. To wire it in
(future PR):

1. `ros2_ws/.../multi_camera_node.py`: when publishing `camera_info`,
   substitute K + dist from `config/intrinsics.yaml` if present.
2. `calib/capture_handeye.py` + `solve_calibration.py`: load K from
   `config/intrinsics.yaml` instead of `pyrealsense2.intrinsics` when
   present; re-run head-pose / hand-eye solve.
3. `rerun_viz_node.py`: depth back-projection uses pinhole K — switch to
   loading from `intrinsics.yaml`.

For now (top_head only first), the file just records the calibration —
verify the numbers look right, then we can wire the consumers in a follow-up.

## 8. Why CharuCo over plain checkerboard

- Partial occlusion safe: a checkerboard either is fully detected or
  drops the frame; CharuCo recognizes the 17 ArUco markers individually
  so a partially-visible board is still usable.
- Sub-pixel corner refinement is built into the OpenCV detector
  (`CORNER_REFINE_SUBPIX`, win=5, see `board_def.get_detector_params`).
- Same board doubles for the existing hand-eye flow — one printout, two
  pipelines.

## 9. Common failure modes

| symptom | cause | fix |
|---|---|---|
| RMS > 0.6 px after 30 frames | poor board print, motion blur, glare | re-print on rigid panel; static board, move camera |
| max err >> mean err | one frame is way off (motion blur, edge corner) | rerun `solve_intrinsics --outlier-sigma 1.5` (drops more) |
| Δfx, Δfy ≫ 10 px | wrong board square size, or non-square pixels | caliper-check the print scale; recompute SQUARE_MM in board_def.py |
| undistort drift > 5 px | k3 over-fit (typical with poor edge coverage) | capture more frames with board near image corners; consider `--flags FIX_K3` in solve (TODO flag) |
| solve_intrinsics rejects all frames | min-keep > captured | lower `--min-keep` or capture more |

## 10. One-liner for top_head right now

```bash
# capture (interactive)
python3 calib/capture_intrinsics.py \
    --camera top_head --session top_head_intr_$(date +%Y-%m-%d)

# solve
python3 calib/solve_intrinsics.py \
    --session top_head_intr_$(date +%Y-%m-%d)

# verify
python3 calib/verify_intrinsics.py \
    --session top_head_intr_$(date +%Y-%m-%d)

# inspect output
cat config/intrinsics.yaml
```
