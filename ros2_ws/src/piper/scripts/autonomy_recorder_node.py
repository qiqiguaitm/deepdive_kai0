#!/usr/bin/env python3
"""Autonomy LeRobot dataset recorder — drops binary-identical episodes (parquet
+ mp4 + zarr) into Task_X/autonomy/<date>/... so deployment data round-trips
through start_autonomy.sh --replay just like teleop output.

Trigger model:
  - Starts when first /master/joint_* message arrives (= policy outputting).
  - Stops on rclpy shutdown / SIGINT, finalizing the episode atomically.
  - One run = one episode. Next run picks ep+1 from existing files.

state, action layout (matches data_manager backend's ros_bridge):
  state  = puppet  left[7] + puppet  right[7]   # real arm sensor readback
  action = master left[7] + master right[7]    # policy commands (autonomy)
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState


# ── locate web/data_manager/backend so we can reuse the teleop writer ──
# Probes upward from this file for the dataset_writer.py module so a single
# implementation produces the on-disk bytes (mp4 / parquet / zarr meta).
def _bootstrap_backend_path() -> None:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "web" / "data_manager" / "backend" / "app" / "dataset_writer.py"
        if candidate.is_file():
            sys.path.insert(0, str(candidate.parent.parent))  # adds .../backend
            return
    raise RuntimeError("could not locate web/data_manager/backend; "
                       "autonomy recorder needs dataset_writer.py")


# KAI0_DATA_ROOT is overloaded: setup_env.sh sets it to the training root
# (.../deepdive_kai0/kai0) but data_manager.config.DATA_ROOT — which dataset_writer
# resolves to at import time — expects the recordings root (default
# /data1/DATA_IMP/KAI0, where teleop writes). Override KAI0_DATA_ROOT for this
# subprocess only; the kai0 training/inference codepaths run in separate procs
# (policy_inference, train.py, etc.) so they're unaffected.
# Set KAI0_RECORDING_ROOT to point recordings elsewhere.
os.environ["KAI0_DATA_ROOT"] = os.environ.get(
    "KAI0_RECORDING_ROOT", "/data1/DATA_IMP/KAI0"
)

_bootstrap_backend_path()
from app.dataset_writer import (  # noqa: E402
    CAMERAS,
    DEPTH_CAMERAS,
    EpisodeWriter,
    FPS,
    HEIGHT,
    WIDTH,
    next_episode_id,
    update_info_json,
    write_episode_meta,
)


# Topics published by multi_camera_node (hardcoded in that node, NOT taken
# from cameras.yml — the yaml is consumed by teleop/data_manager only).
CAM_RGB_TOPIC = {
    "top_head":   "/camera_f/camera/color/image_raw",
    "hand_left":  "/camera_l/camera/color/image_raw",
    "hand_right": "/camera_r/camera/color/image_raw",
}
CAM_DEPTH_TOPIC = {
    "top_head":   "/camera_f/camera/aligned_depth_to_color/image_raw",
    "hand_left":  "/camera_l/camera/aligned_depth_to_color/image_raw",
    "hand_right": "/camera_r/camera/aligned_depth_to_color/image_raw",
}
JOINT_TOPIC = {
    "left_slave":   "/puppet/joint_left",
    "right_slave":  "/puppet/joint_right",
    "left_master":  "/master/joint_left",
    "right_master": "/master/joint_right",
}


def _normalize_topic(topic: str) -> str:
    """Collapse adjacent duplicate path segments: ``/camera_f/camera_f/color/x``
    → ``/camera_f/color/x``. Mirrors the same step in ros_bridge so this node
    subscribes to the topic that the realsense launcher actually publishes on.
    """
    parts = [p for p in topic.split("/") if p]
    out = []
    for p in parts:
        if not out or out[-1] != p:
            out.append(p)
    return "/" + "/".join(out)


def _decode_image_rgb(msg: Image) -> Optional[np.ndarray]:
    """sensor_msgs/Image → uint8 H×W×3 RGB. None on unsupported encoding."""
    w, h, enc = msg.width, msg.height, msg.encoding
    data = bytes(msg.data)
    if enc == "rgb8":
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).copy()
    if enc == "bgr8":
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(arr[:, :, ::-1])
    return None


def _decode_image_depth(msg: Image) -> Optional[np.ndarray]:
    """sensor_msgs/Image (16UC1 / mono16) → uint16 H×W."""
    if msg.encoding not in ("16UC1", "mono16"):
        return None
    w, h = msg.width, msg.height
    return np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h, w).copy()


def _to_7dim(msg: JointState) -> list[float]:
    pos = list(msg.position)[:7]
    pos += [0.0] * (7 - len(pos))
    return [float(x) for x in pos]


def _infer_task_from_ckpt(ckpt_dir: str) -> str:
    """`/data1/.../checkpoints/Task_A/mixed_1` → 'Task_A'.
    `/data1/.../checkpoints/task_a_new_pure2_*` → 'Task_A'.
    Falls back to 'Task_A' if nothing matches.
    """
    if not ckpt_dir:
        return "Task_A"
    s = ckpt_dir.lower()
    for letter in ("a", "b", "c", "d", "e"):
        if re.search(rf"\btask[_-]?{letter}\b", s) or f"/task_{letter}/" in s:
            return f"Task_{letter.upper()}"
    return "Task_A"


def _infer_prompt_from_ckpt(ckpt_dir: str) -> str:
    """Try ckpt_dir/train_config.json's prompt field, else degrade to a label."""
    if not ckpt_dir:
        return "deployed inference"
    cfg_path = pathlib.Path(ckpt_dir) / "train_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
            p = cfg.get("prompt") or cfg.get("task_prompt") or cfg.get("default_prompt")
            if p:
                return str(p)
        except Exception:
            pass
    return f"deployed inference of {pathlib.Path(ckpt_dir).name}"


class AutonomyRecorder(Node):
    def __init__(self) -> None:
        super().__init__("autonomy_recorder")

        # ── Parameters ──
        self.declare_parameter("task_name", "")
        self.declare_parameter("prompt", "")
        self.declare_parameter("subset", "autonomy")
        self.declare_parameter("operator", "auto")
        self.declare_parameter("checkpoint_dir", "")
        self.declare_parameter("record_enable", True)

        ckpt_dir = self.get_parameter("checkpoint_dir").value or ""
        task_param = self.get_parameter("task_name").value or ""
        prompt_param = self.get_parameter("prompt").value or ""

        self._task: str = task_param or _infer_task_from_ckpt(ckpt_dir)
        self._prompt: str = prompt_param or _infer_prompt_from_ckpt(ckpt_dir)
        self._subset: str = self.get_parameter("subset").value or "autonomy"
        self._operator: str = self.get_parameter("operator").value or "auto"
        self._record_enable: bool = bool(self.get_parameter("record_enable").value)

        # ── State ──
        self._lock = threading.Lock()
        self._writer: Optional[EpisodeWriter] = None
        self._started_at: float = 0.0
        self._rgb: dict[str, Optional[np.ndarray]] = {c: None for c in CAMERAS}
        self._depth: dict[str, Optional[np.ndarray]] = {c: None for c in DEPTH_CAMERAS}
        self._joint_pos: dict[str, list[float]] = {k: [0.0] * 7 for k in JOINT_TOPIC}
        self._tick_n = 0
        self._wrote_frames = 0

        # ── QoS for sensor streams ──
        # multi_camera_node publishes with RELIABLE+VOLATILE; match to maximize
        # delivery. KEEP_LAST depth=1 is fine — we always pick latest anyway.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribe: RGB cameras ──
        for cam, topic in CAM_RGB_TOPIC.items():
            t = _normalize_topic(topic)
            self.create_subscription(
                Image, t,
                lambda msg, k=cam: self._on_rgb(k, msg),
                sensor_qos,
            )

        # ── Subscribe: Depth (only cameras in DEPTH_CAMERAS macro) ──
        for cam in DEPTH_CAMERAS:
            topic = CAM_DEPTH_TOPIC.get(cam)
            if not topic:
                continue
            t = _normalize_topic(topic)
            self.create_subscription(
                Image, t,
                lambda msg, k=cam: self._on_depth(k, msg),
                sensor_qos,
            )

        # ── Subscribe: joints ──
        for key, topic in JOINT_TOPIC.items():
            cb = self._on_master_joint if key.endswith("_master") else self._on_slave_joint
            self.create_subscription(
                JointState, topic,
                lambda msg, k=key, fn=cb: fn(k, msg),
                10,
            )

        # ── 30 Hz capture timer ──
        self.create_timer(1.0 / FPS, self._on_tick)

        self.get_logger().info(
            f"autonomy_recorder ready: task={self._task} subset={self._subset} "
            f"fps={FPS} depth={DEPTH_CAMERAS} record_enable={self._record_enable}; "
            f"will auto-start on first /master/joint_*"
        )

    # ── Callbacks ──
    def _on_rgb(self, cam: str, msg: Image) -> None:
        arr = _decode_image_rgb(msg)
        if arr is None:
            return
        with self._lock:
            self._rgb[cam] = arr

    def _on_depth(self, cam: str, msg: Image) -> None:
        arr = _decode_image_depth(msg)
        if arr is None:
            return
        with self._lock:
            self._depth[cam] = arr

    def _on_slave_joint(self, key: str, msg: JointState) -> None:
        pos = _to_7dim(msg)
        with self._lock:
            self._joint_pos[key] = pos

    def _on_master_joint(self, key: str, msg: JointState) -> None:
        pos = _to_7dim(msg)
        # Filter out arm_reader_node stub publishes (all zeros, no real command).
        is_real = any(abs(x) > 1e-6 for x in pos[:6])
        with self._lock:
            self._joint_pos[key] = pos
            should_start = (
                is_real and self._record_enable
                and self._writer is None
            )
        if should_start:
            self._start_recording()

    # ── Episode lifecycle ──
    def _start_recording(self) -> None:
        try:
            ep = next_episode_id(self._task, self._subset)
            writer = EpisodeWriter(
                task=self._task, subset=self._subset, ep=ep,
                prompt=self._prompt, template_id="autonomy", operator=self._operator,
            )
        except Exception as e:
            self.get_logger().error(f"writer init failed: {e}")
            return
        with self._lock:
            self._writer = writer
            self._started_at = time.time()
        self.get_logger().info(
            f"recording started: task={self._task} subset={self._subset} "
            f"ep={ep} → {writer.root}"
        )

    def _on_tick(self) -> None:
        with self._lock:
            writer = self._writer
            if writer is None:
                return
            state = self._joint_pos["left_slave"] + self._joint_pos["right_slave"]
            action = self._joint_pos["left_master"] + self._joint_pos["right_master"]
            frames = {cam: self._rgb[cam] for cam in CAMERAS}
            depth_frames = {cam: self._depth.get(cam) for cam in DEPTH_CAMERAS}
            now = time.time()

        try:
            writer.write_tick(frames, state, action, now, depth_frames=depth_frames)
        except Exception as e:
            self.get_logger().error(f"write_tick failed (aborting recording): {e}")
            with self._lock:
                self._writer = None
            try:
                writer.abort()
            except Exception:
                pass
            return

        self._wrote_frames += 1
        self._tick_n += 1
        if self._tick_n % (FPS * 10) == 0:  # every 10s
            self.get_logger().info(
                f"recording tick={self._tick_n} frames={self._wrote_frames} "
                f"({self._wrote_frames / FPS:.1f}s)"
            )

    def finalize(self) -> None:
        """Flush mp4 encoders, write meta/episodes.jsonl + info.json. Idempotent."""
        with self._lock:
            writer = self._writer
            started_at = self._started_at
            self._writer = None
        if writer is None:
            return
        duration = time.time() - started_at
        try:
            writer.finalize()
            write_episode_meta(
                writer, duration, success=True,
                note=f"autonomy session ({self._wrote_frames} frames @ {FPS} Hz)",
                scene_tags=[],
            )
            update_info_json(self._task, self._subset)
            self.get_logger().info(
                f"recording saved: ep={writer.ep} frames={self._wrote_frames} "
                f"duration={duration:.1f}s → {writer.root}"
            )
        except Exception as e:
            self.get_logger().error(f"finalize failed ({e}); aborting episode")
            try:
                writer.abort()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = AutonomyRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
