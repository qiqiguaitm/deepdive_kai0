"""Per-camera depth-stream toggle (macro-style).

This module is the *single source of truth* for whether each RealSense
camera's depth output is consumed downstream — both during teleoperation
data collection (web/data_manager + multi_camera_node) and during model
testing / inference (rerun_viz_node, autonomy_launch).

Flip a flag below to turn that camera's depth path on/off everywhere; no
other file needs editing. Setting a flag to False means:

  * multi_camera_node.py     skips enabling the depth stream on the device
  * launch_3cam.py           sets enable_depth=False on that realsense2 node
  * web/data_manager.recorder.py  doesn't allocate / write a zarr for it
  * web/data_manager.sync.py      doesn't push its zarr to remote stores
  * web/data_manager.ros_bridge.py doesn't subscribe to its depth topic
  * rerun_viz_node.py        skips its depth subscription, point cloud,
                             and foreground mesh pass

Color (RGB) processing is unaffected — these flags only gate depth.

Why D405 wrist depth is currently off:
  * Bandwidth contention on the shared USB hub: enabling 3x depth streams
    pushed the wrist cameras into "Incomplete frame" drop loops at 30 fps.
  * Storage cost: 16-bit 640x480 zarr per arm doubled per-episode size
    without measurable downstream gain — the policy doesn't consume depth,
    and the wrist FK pose is already known so wrist depth was redundant.
  * D405 global-shutter color requires a fixed 20 ms exposure for anti-
    flicker, which conflicts with auto-exposure that depth filtering
    expects.
"""

# ── Macros: edit these to enable/disable each camera's depth path ──────
ENABLE_DEPTH_TOP_HEAD: bool = True   # D435 (top/俯视全局)
ENABLE_DEPTH_HAND_LEFT: bool = False  # D405 (左手腕)
ENABLE_DEPTH_HAND_RIGHT: bool = False  # D405 (右手腕)

# ── Derived helpers (do not edit; computed from the macros above) ──────
CAMERA_DEPTH_ENABLED: dict[str, bool] = {
    "top_head": ENABLE_DEPTH_TOP_HEAD,
    "hand_left": ENABLE_DEPTH_HAND_LEFT,
    "hand_right": ENABLE_DEPTH_HAND_RIGHT,
}

DEPTH_CAMERAS: tuple[str, ...] = tuple(
    name for name, enabled in CAMERA_DEPTH_ENABLED.items() if enabled
)


def is_depth_enabled(camera: str) -> bool:
    """Return True iff the given camera's depth stream should be consumed.

    Unknown camera names default to False so a typo can't accidentally
    turn on a stream that was meant to stay off.
    """
    return CAMERA_DEPTH_ENABLED.get(camera, False)
