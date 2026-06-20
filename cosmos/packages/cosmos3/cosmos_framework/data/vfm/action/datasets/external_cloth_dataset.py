# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""External cloth-folding LeRobot v2 dataset loader (style-1: per-episode parquet).

Supports arbitrary camera key mappings and optional action-index extraction so
a single class can ingest robocoin, unitree-z1, xvla-soft, and ALOHA-style
datasets with minimal per-dataset configuration.

Compatible with ``WamFoldLeRobotDataset``'s ``__getitem__`` contract so the
same ``ActionDataPacker`` / ``DataPackerDataLoader`` can consume it.

Only style-1 LeRobot v2 datasets are supported here (per-episode parquet files
at ``data/chunk-{NNN}/episode_{NNNNNN}.parquet``).  Style-2 file-grouped datasets
(e.g. full_folding, unitree_z1) will be added in a follow-up loader.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.action_normalization import normalize_action
from cosmos_framework.data.vfm.action.action_spec import Joint, build_action_spec
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.pose_utils import compute_idle_frames

# Reuse the bounded decoder cache from wam_fold_dataset to avoid CPU-RAM leaks.
import lerobot.datasets.video_utils as _lvu
from cosmos_framework.data.vfm.action.datasets.wam_fold_dataset import _BoundedVideoDecoderCache as _BVC

if not isinstance(_lvu._default_decoder_cache, _BVC):
    _lvu._default_decoder_cache = _BVC(int(os.environ.get("LEROBOT_DECODER_CACHE_MAX", "16")))

Viewpoint = Literal["concat_view"]
_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")
_ACTION_DIM = 14
_DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)
_ACTION_SPEC = build_action_spec(Joint(n=_ACTION_DIM, label="arm"))

# ALOHA 26-D → 14-D: keep joint positions + grippers, drop EEF cartesian.
# Input layout per arm: [j1..j6, eef_pos_xyz, eef_rot_xyz, gripper] (13 per arm)
# We keep: [j1..j6 (0:6), gripper (12)] for left + [j1..j6 (13:19), gripper (25)] for right
ALOHA_ACTION_INDICES: list[int] = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 25]

# full-folding 16-D → 14-D: drop joint_7 per arm (7-DOF arm, we keep 6+gripper)
# Input: [R_j1..R_j7, R_gripper, L_j1..L_j7, L_gripper]
# Output: [R_j1..R_j6, R_gripper, L_j1..L_j6, L_gripper]
FULL_FOLDING_ACTION_INDICES: list[int] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15]

# ── Per-dataset registry ─────────────────────────────────────────────────────
# Each entry:
#   root         : absolute path to LeRobot v2 dataset root
#   domain_name  : key into EMBODIMENT_TO_DOMAIN_ID
#   stats_path   : path to quantile norm stats JSON (same format as visrobot01.json)
#   camera_map   : {"high": <feat_key_suffix>, "left": ..., "right": ...}
#                  The suffix is appended to "observation.images."
#   action_col   : parquet column name for raw action (default "action")
#   action_indices: optional list[int] to extract 14-D from wider action; None = use as-is
#   state_col    : parquet column for proprioception anchor (for delta transform)
#   task_text    : task description string
#   fps          : recording fps (used for conditioning_fps token)

_EXT_BASE = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/external_cloth"
_STATS_ROOT = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/data/stats"

EXT_RIG_DEFAULTS: dict[str, dict[str, Any]] = {
    # ── Tier 1: 14-D, style-1, camera keys already match or need simple rename ──
    "robocoin_fold": {
        "root": f"{_EXT_BASE}/robocoin_fold_clothes",
        "domain_name": "robocoin_fold",
        "stats_path": f"{_STATS_ROOT}/robocoin_fold.json",
        "camera_map": {"high": "cam_front_rgb", "left": "cam_left_wrist_rgb", "right": "cam_right_wrist_rgb"},
        "action_col": "action",
        "action_indices": None,
        "state_col": "observation.state",
        "task_text": "Flatten and fold the cloth.",
        "fps": 30.0,
    },
    "robocoin_r1lite": {
        "root": f"{_EXT_BASE}/robocoin_r1lite_fold_clothes",
        "domain_name": "robocoin_r1lite",
        "stats_path": f"{_STATS_ROOT}/robocoin_r1lite.json",
        "camera_map": {"high": "cam_high_rgb", "left": "cam_left_wrist_rgb", "right": "cam_right_wrist_rgb"},
        "action_col": "action",
        "action_indices": None,
        "state_col": "observation.state",
        "task_text": "Flatten and fold the cloth.",
        "fps": 30.0,
    },
    # ── Tier 2: 26-D ALOHA → 14-D extraction ───────────────────────────────
    "robocoin_towel_blue": {
        "root": f"{_EXT_BASE}/robocoin_fold_towel_blue",
        "domain_name": "robocoin_aloha",
        "stats_path": f"{_STATS_ROOT}/robocoin_aloha.json",
        "camera_map": {"high": "cam_head_rgb", "left": "cam_left_wrist_rgb", "right": "cam_right_wrist_rgb"},
        "action_col": "action",
        "action_indices": ALOHA_ACTION_INDICES,
        "state_col": "observation.state",
        "task_text": "Fold the towel on the table.",
        "fps": 30.0,
    },
    "robocoin_towel_brown": {
        "root": f"{_EXT_BASE}/robocoin_fold_towel_brown",
        "domain_name": "robocoin_aloha",
        "stats_path": f"{_STATS_ROOT}/robocoin_aloha.json",
        "camera_map": {"high": "cam_head_rgb", "left": "cam_left_wrist_rgb", "right": "cam_right_wrist_rgb"},
        "action_col": "action",
        "action_indices": ALOHA_ACTION_INDICES,
        "state_col": "observation.state",
        "task_text": "Fold the towel on the table.",
        "fps": 30.0,
    },
    "robocoin_short_sleeve": {
        "root": f"{_EXT_BASE}/robocoin_fold_short_sleeve_white",
        "domain_name": "robocoin_aloha",
        "stats_path": f"{_STATS_ROOT}/robocoin_aloha.json",
        "camera_map": {"high": "cam_head_rgb", "left": "cam_left_wrist_rgb", "right": "cam_right_wrist_rgb"},
        "action_col": "action",
        "action_indices": ALOHA_ACTION_INDICES,
        "state_col": "observation.state",
        "task_text": "Fold the short-sleeve shirt.",
        "fps": 30.0,
    },
    "robocoin_tray_twice": {
        "root": f"{_EXT_BASE}/robocoin_fold_towel_tray_twice",
        "domain_name": "robocoin_aloha",
        "stats_path": f"{_STATS_ROOT}/robocoin_aloha.json",
        "camera_map": {"high": "cam_head_rgb", "left": "cam_left_wrist_rgb", "right": "cam_right_wrist_rgb"},
        "action_col": "action",
        "action_indices": ALOHA_ACTION_INDICES,
        "state_col": "observation.state",
        "task_text": "Fold the towel on the tray.",
        "fps": 30.0,
    },
    # ── Tier 3: AgiBot a2d humanoid, 14-D arm joints, cloth-folding tasks ──
    # Camera keys: head (top), hand_left/hand_right (wrists).  No _rgb suffix.
    # action_col: "actions.joint.position" (14-D, arm joints only, no waist/head).
    # state_col:  "observation.states.joint.position" (same 14-D layout).
    # Domain: reuses agibotworld=15 (same embodiment family).
    # task_362 has 8,233 eps (54% of AgiBot total); use max_episodes to cap it.
    "agibot_362": {
        "root": f"{_EXT_BASE}/agibot_task362/task_362",
        "domain_name": "agibotworld",
        "stats_path": f"{_STATS_ROOT}/agibotworld_fold.json",
        "camera_map": {"high": "head", "left": "hand_left", "right": "hand_right"},
        "action_col": "actions.joint.position",
        "action_indices": None,
        "state_col": "observation.states.joint.position",
        "task_text": "Fold the shorts on the bed.",
        "fps": 30.0,
    },
    "agibot_444": {
        "root": f"{_EXT_BASE}/agibot_task444/task_444",
        "domain_name": "agibotworld",
        "stats_path": f"{_STATS_ROOT}/agibotworld_fold.json",
        "camera_map": {"high": "head", "left": "hand_left", "right": "hand_right"},
        "action_col": "actions.joint.position",
        "action_indices": None,
        "state_col": "observation.states.joint.position",
        "task_text": "Fold the short-sleeve shirt on the bed.",
        "fps": 30.0,
    },
    "agibot_477": {
        "root": f"{_EXT_BASE}/agibot_task477/task_477",
        "domain_name": "agibotworld",
        "stats_path": f"{_STATS_ROOT}/agibotworld_fold.json",
        "camera_map": {"high": "head", "left": "hand_left", "right": "hand_right"},
        "action_col": "actions.joint.position",
        "action_indices": None,
        "state_col": "observation.states.joint.position",
        "task_text": "Fold the towel on the table.",
        "fps": 30.0,
    },
    "agibot_509": {
        "root": f"{_EXT_BASE}/agibot_task509/task_509",
        "domain_name": "agibotworld",
        "stats_path": f"{_STATS_ROOT}/agibotworld_fold.json",
        "camera_map": {"high": "head", "left": "hand_left", "right": "hand_right"},
        "action_col": "actions.joint.position",
        "action_indices": None,
        "state_col": "observation.states.joint.position",
        "task_text": "Fold the towel near the washbasin.",
        "fps": 30.0,
    },
    "agibot_520": {
        "root": f"{_EXT_BASE}/agibot_task520/task_520",
        "domain_name": "agibotworld",
        "stats_path": f"{_STATS_ROOT}/agibotworld_fold.json",
        "camera_map": {"high": "head", "left": "hand_left", "right": "hand_right"},
        "action_col": "actions.joint.position",
        "action_indices": None,
        "state_col": "observation.states.joint.position",
        "task_text": "Fold the shorts on the bed.",
        "fps": 30.0,
    },
    "agibot_555": {
        "root": f"{_EXT_BASE}/agibot_task555/task_555",
        "domain_name": "agibotworld",
        "stats_path": f"{_STATS_ROOT}/agibotworld_fold.json",
        "camera_map": {"high": "head", "left": "hand_left", "right": "hand_right"},
        "action_col": "actions.joint.position",
        "action_indices": None,
        "state_col": "observation.states.joint.position",
        "task_text": "Fold the shorts on the bed.",
        "fps": 30.0,
    },
    "agibot_561": {
        "root": f"{_EXT_BASE}/agibot_task561/task_561",
        "domain_name": "agibotworld",
        "stats_path": f"{_STATS_ROOT}/agibotworld_fold.json",
        "camera_map": {"high": "head", "left": "hand_left", "right": "hand_right"},
        "action_col": "actions.joint.position",
        "action_indices": None,
        "state_col": "observation.states.joint.position",
        "task_text": "Flatten and fold the shorts on the bed.",
        "fps": 30.0,
    },
}


class ExternalClothDataset(Dataset):
    """Style-1 LeRobot v2 cloth-folding dataset with configurable camera/action mapping."""

    def __init__(
        self,
        rig: str,
        root: str | None = None,
        domain_name: str | None = None,
        stats_path: str | None = None,
        camera_map: dict[str, str] | None = None,
        action_col: str | None = None,
        action_indices: list[int] | None = "USE_DEFAULT",
        state_col: str | None = None,
        task_text: str | None = None,
        fps: float | None = None,
        chunk_length: int = 32,
        mode: str = "forward_dynamics",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "concat_view",
        max_episodes: int | None = None,
        window_stride: int = 1,
    ) -> None:
        super().__init__()
        if rig not in EXT_RIG_DEFAULTS:
            raise KeyError(f"Unknown external rig {rig!r}. Available: {sorted(EXT_RIG_DEFAULTS)}")
        defaults = EXT_RIG_DEFAULTS[rig]

        self._rig = rig
        self._root = Path(root or defaults["root"])
        self._domain_name = domain_name or defaults["domain_name"]
        self._domain_id = get_domain_id(self._domain_name)
        self._stats_path = stats_path or defaults["stats_path"]
        self._camera_map = camera_map or defaults["camera_map"]
        self._action_col = action_col or defaults["action_col"]
        self._action_indices: list[int] | None = (
            defaults["action_indices"] if action_indices == "USE_DEFAULT" else action_indices
        )
        self._state_col = state_col or defaults["state_col"]
        self._task_text = task_text or defaults["task_text"]
        self._fps = float(fps or defaults["fps"])
        self._chunk_length = int(chunk_length)
        self._mode = mode
        self._tolerance_s = float(tolerance_s)
        self._norm_stats: dict[str, torch.Tensor] | None = None

        self._info = json.loads((self._root / "meta" / "info.json").read_text())

        self._episode_files = sorted(
            (self._root / "data").glob("chunk-*/episode_*.parquet"),
            key=lambda p: (p.parent.name, p.name),
        )
        if not self._episode_files:
            raise FileNotFoundError(f"No per-episode parquet files found under {self._root / 'data'}")
        if max_episodes is not None and max_episodes < len(self._episode_files):
            rng = random.Random(42)
            self._episode_files = rng.sample(self._episode_files, max_episodes)

        self._windows: list[tuple[int, int]] = []
        self._file_meta: list[dict[str, int]] = []
        for file_idx, path in enumerate(self._episode_files):
            episode_chunk = int(path.parent.name.split("-")[-1])
            episode_index = int(path.stem.split("_")[-1])
            num_rows = pq.read_metadata(path).num_rows
            self._file_meta.append(
                {"episode_chunk": episode_chunk, "episode_index": episode_index, "num_rows": int(num_rows)}
            )
            for start in range(0, max(0, int(num_rows) - self._chunk_length), window_stride):
                self._windows.append((file_idx, start))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = random.choice(_MODE_CHOICES) if self._mode == "joint" else self._mode
        file_idx, start = self._windows[int(idx)]
        path = self._episode_files[file_idx]
        meta = self._file_meta[file_idx]

        table = pq.read_table(path)
        obs_slice = slice(start, start + self._chunk_length + 1)
        timestamps = table.column("timestamp").to_pylist()[obs_slice]
        timestamps = [float(t[0]) if isinstance(t, (list, tuple)) else float(t) for t in timestamps]

        raw = np.asarray(
            table.column(self._action_col).to_pylist()[start : start + self._chunk_length],
            dtype=np.float32,
        )  # [chunk, D]

        # Extract 14-D from wider action (ALOHA→14, full_folding→14, etc.)
        if self._action_indices is not None:
            raw = raw[:, self._action_indices]  # [chunk, 14]

        # Delta transform: arm joints → delta vs anchor state, grippers absolute.
        state_anchor = np.asarray(
            table.column(self._state_col).to_pylist()[start], dtype=np.float32
        )
        if self._action_indices is not None:
            state_anchor = state_anchor[self._action_indices]
        raw = raw.copy()
        raw[:, _DELTA_MASK] = raw[:, _DELTA_MASK] - state_anchor[_DELTA_MASK]

        video = self._load_concat_video(meta, timestamps)
        action = torch.from_numpy(raw).float()

        cache_key = (
            f"{self._rig}_d{self._domain_id}_c{meta['episode_chunk']:03d}"
            f"_e{meta['episode_index']:06d}_s{start:04d}_L{self._chunk_length:02d}"
        )

        idle_frames = compute_idle_frames(action, _ACTION_SPEC, joint_threshold=5e-3, min_streak=3)
        normalized_action = normalize_action(action, "quantile", self._load_norm_stats())
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)

        return {
            "ai_caption": self._task_text,
            "video": formatted_video,
            "action": normalized_action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": "concat_view",
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            "cache_key": cache_key,
            "additional_view_description": (
                "The top row is from the head/front-mounted camera. "
                "The bottom row contains two horizontally concatenated wrist-camera views."
            ),
        }

    def _load_concat_video(self, meta: dict[str, int], timestamps: list[float]) -> torch.Tensor:
        def _load(suffix: str) -> torch.Tensor:
            video_key = f"observation.images.{suffix}"
            rel = self._info["video_path"].format(
                video_key=video_key,
                episode_chunk=meta["episode_chunk"],
                chunk_index=meta["episode_chunk"],
                episode_index=meta["episode_index"],
                file_index=meta["episode_index"],
            )
            return decode_video_frames(self._root / rel, timestamps, self._tolerance_s)

        high = _load(self._camera_map["high"])
        left = _load(self._camera_map["left"])
        right = _load(self._camera_map["right"])
        _, _, h_h, w_h = high.shape
        half_h, half_w = h_h // 2, w_h // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        return torch.cat([high, torch.cat([left, right], dim=-1)], dim=-2)

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is not None:
            return self._norm_stats
        raw = json.loads(Path(self._stats_path).read_text())
        action_block = raw["global"]["action"]
        self._norm_stats = {
            k: torch.tensor(action_block[k], dtype=torch.float32)
            for k in ("mean", "std", "min", "max", "q01", "q99")
            if k in action_block
        }
        return self._norm_stats

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def domain_id(self) -> int:
        return self._domain_id
