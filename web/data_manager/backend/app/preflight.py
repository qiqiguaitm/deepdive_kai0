"""后端版 preflight —— 与前端 StatusBar.collectFailures 保持一致的故障列表。

独立模块而非 copy-paste 到 toggle endpoint, 是为了 (a) 方便将来前端改调
/api/recorder/preflight 统一逻辑, (b) 可单元测试。
"""
from __future__ import annotations

EXPECTED_CAMERAS = ("top_head", "hand_left", "hand_right")
_CAM_LABEL = {"top_head": "俯视相机", "hand_left": "左手相机", "hand_right": "右手相机"}
_CAM_FPS_SLACK = 3  # 与前端一致: 允许实际 fps 比 target 低 3


def collect_failures(snap: dict) -> list[str]:
    """Mirror of frontend collectFailures() — returns empty list iff system OK."""
    fails: list[str] = []
    h = snap.get("health") or {}
    if not h.get("ros2"):
        fails.append("ROS2")
    if not h.get("can_left"):
        fails.append("CAN-L")
    if not h.get("can_right"):
        fails.append("CAN-R")
    if not h.get("teleop"):
        fails.append("Teleop")

    cams = snap.get("cameras") or {}
    for cam in EXPECTED_CAMERAS:
        s = cams.get(cam)
        if not s:
            fails.append(f"{_CAM_LABEL[cam]}(缺失)")
            continue
        target = s.get("target_fps") or 30
        fps = s.get("fps") or 0
        if fps < target - _CAM_FPS_SLACK:
            fails.append(f"{_CAM_LABEL[cam]}(fps {fps}/{target})")

    rec = snap.get("recorder") or {}
    if rec.get("state") == "ERROR":
        fails.append(f"录制错误: {rec.get('error') or ''}")

    for w in snap.get("warnings") or []:
        fails.append(str(w))
    return fails
