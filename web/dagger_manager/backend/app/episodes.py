"""History episode browsing + replay for dagger_manager.

Reads the Form C dual-dataset layout written by dagger_recorder_node:
    <DATA_ROOT>/<task>/<subset>/<date>-v2/
        ├── data/chunk-000/episode_NNNNNN.parquet
        ├── videos/chunk-000/<camera>/episode_NNNNNN.mp4   (AV1)
        └── meta/episodes.jsonl

episode_id resets per date dir, so an episode is uniquely keyed by
(subset, date, episode_id). subset ∈ {dagger, inference}.

All path building goes through _safe_join which (1) whitelists each URL
component and (2) verifies the resolved path stays under DATA_ROOT — same
traversal defense as data_manager/episodes.py.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from .stack import DATA_ROOT

SUBSETS = ("dagger", "inference")
C_CAM = ("top_head", "hand_left", "hand_right")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-.]+$")

def _camera_video_path(date_root: Path, cam: str, ep: int) -> Path:
    """Try `observation.images.<cam>` (v4+) then bare `<cam>` (v3)."""
    name = f"episode_{ep:06d}.mp4"
    p = date_root / "videos" / "chunk-000" / f"observation.images.{cam}" / name
    if p.exists():
        return p
    return date_root / "videos" / "chunk-000" / cam / name


def _camera_video_candidates(date_root: Path, cam: str, ep: int) -> list[Path]:
    """All possible video paths for deletion."""
    name = f"episode_{ep:06d}.mp4"
    return [
        date_root / "videos" / "chunk-000" / f"observation.images.{cam}" / name,
        date_root / "videos" / "chunk-000" / cam / name,
    ]


def _camera_depth_candidates(date_root: Path, cam: str, ep: int) -> list[Path]:
    """All possible depth paths for deletion."""
    stem, zarr_n = f"episode_{ep:06d}", f"episode_{ep:06d}.zarr"
    return [
        date_root / "videos" / "chunk-000" / f"observation.depth.{cam}" / zarr_n,
        date_root / "videos" / "chunk-000" / f"observation.depth.{cam}" / (stem + ".zarr.zip"),
        date_root / "videos" / "chunk-000" / f"observation.depth.{cam}" / (stem + ".mkv"),
        date_root / "videos" / "chunk-000" / f"{cam}_depth" / zarr_n,
        date_root / "videos" / "chunk-000" / f"{cam}_depth" / (stem + ".zarr.zip"),
        date_root / "videos" / "chunk-000" / f"{cam}_depth" / (stem + ".mkv"),
    ]


def _safe(*parts: str) -> None:
    for p in parts:
        if not SAFE_NAME.match(p):
            raise HTTPException(400, f"unsafe path component: {p!r}")


_VER_RE = re.compile(r"v\d+$")


def _iter_date_dirs(subset_root: Path):
    """Yield (date_leaf, date_dir) for BOTH layouts:
      - nested (2026-06-15+): <subset>/<vN>/<date>-vN/   (date dirs under a version dir)
      - legacy flat:          <subset>/<date>-v2/         (date dirs directly under subset)
    Only dirs containing meta/episodes.jsonl are yielded."""
    if not subset_root.is_dir():
        return
    for child in sorted(subset_root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        if _VER_RE.fullmatch(child.name):
            # version dir → its children are the date dirs
            for dd in sorted(child.iterdir(), reverse=True):
                if dd.is_dir() and (dd / "meta" / "episodes.jsonl").is_file():
                    yield dd.name, dd
        elif (child / "meta" / "episodes.jsonl").is_file():
            yield child.name, child  # legacy flat date dir


def _date_root(task: str, subset: str, date: str) -> Path:
    _safe(task, subset, date)
    if subset not in SUBSETS:
        raise HTTPException(400, f"unknown subset {subset!r}")
    subset_root = DATA_ROOT / task / subset
    # Candidates: nested <subset>/<vN>/<date> (vN parsed from the date's own -vN
    # suffix, e.g. 2026-06-15-v3 → v3), then legacy flat <subset>/<date>.
    candidates: list[Path] = []
    m = re.search(r"-(v\d+)$", date)
    if m:
        candidates.append(subset_root / m.group(1) / date)
    candidates.append(subset_root / date)
    for c in candidates:
        full = c.resolve()
        if not str(full).startswith(str(DATA_ROOT.resolve())):
            raise HTTPException(400, "path escapes DATA_ROOT")
        if full.is_dir():
            return full
    # last resort: scan version dirs for a matching leaf
    if subset_root.is_dir():
        for vdir in subset_root.iterdir():
            if vdir.is_dir() and _VER_RE.fullmatch(vdir.name):
                full = (vdir / date).resolve()
                if str(full).startswith(str(DATA_ROOT.resolve())) and full.is_dir():
                    return full
    # nothing on disk — return the primary candidate (callers 404 on missing meta)
    return candidates[0].resolve()


def list_tasks() -> list[dict]:
    """Every Task_* dir under DATA_ROOT, with a flag for whether it has any
    dagger/inference data (so the UI can grey out empty tasks)."""
    out: list[dict] = []
    if not DATA_ROOT.is_dir():
        return out
    for d in sorted(DATA_ROOT.iterdir()):
        if d.is_dir() and d.name.startswith("Task_"):
            has_data = any((d / s).is_dir() for s in SUBSETS)
            out.append({"task": d.name, "has_data": has_data})
    return out


def list_episodes(task: str = "Task_A") -> list[dict]:
    """Enumerate every episode across both subsets and all date dirs.

    Sorted newest-first by (date desc, episode_id desc) so the most recent
    captures bubble to the top of the UI list.
    """
    out: list[dict] = []
    task_root = DATA_ROOT / task
    if not task_root.is_dir():
        return out
    for subset in SUBSETS:
        for _leaf, date_dir in _iter_date_dirs(task_root / subset):
            meta_fp = date_dir / "meta" / "episodes.jsonl"
            video_dir = date_dir / "videos" / "chunk-000"
            for line in meta_fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep = int(d.get("episode_id", -1))
                if ep < 0:
                    continue
                # Has at least the head-cam mp4? (try compliant v4+ key first, then bare v3 key)
                head_mp4_v4 = video_dir / "observation.images.top_head" / f"episode_{ep:06d}.mp4"
                head_mp4_v3 = video_dir / "top_head" / f"episode_{ep:06d}.mp4"
                out.append({
                    "subset": subset,
                    "date": date_dir.name,
                    "episode_id": ep,
                    "length": int(d.get("length", 0)),
                    "duration_s": float(d.get("duration_s", 0.0)),
                    "operator": d.get("operator", ""),
                    "prompt": d.get("prompt", ""),
                    "success": bool(d.get("success", True)),
                    "note": d.get("note", ""),
                    "created_at": d.get("created_at"),
                    "has_video": head_mp4_v4.is_file() or head_mp4_v3.is_file(),
                    # 油门加速标识: 本段 rollout 是否踩过油门 (整段标记) + 峰值倍率
                    "used_throttle": bool(d.get("used_throttle", False)),
                    "speed_factor": float(d.get("speed_factor", 1.0)),
                })
    out.sort(key=lambda e: (e["date"], e["episode_id"]), reverse=True)
    return out


def episode_video_path(task: str, subset: str, date: str, episode_id: int,
                       camera: str) -> Path:
    if camera not in C_CAM:
        raise HTTPException(400, f"unknown camera {camera!r}")
    root = _date_root(task, subset, date)
    return _camera_video_path(root, camera, episode_id)


def episode_meta(task: str, subset: str, date: str, episode_id: int) -> dict:
    meta_fp = _date_root(task, subset, date) / "meta" / "episodes.jsonl"
    if not meta_fp.is_file():
        raise HTTPException(404, "meta not found")
    for line in meta_fp.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(d.get("episode_id", -1)) == episode_id:
            return d
    raise HTTPException(404, "episode not found")


def delete_episode(task: str, subset: str, date: str, episode_id: int) -> None:
    """Remove parquet + per-camera mp4 + meta line. Irreversible."""
    root = _date_root(task, subset, date)
    pq = root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
    if pq.exists():
        pq.unlink()
    for cam in C_CAM:
        for vp in _camera_video_candidates(root, cam, episode_id):
            if vp.exists():
                vp.unlink()
        import shutil as _sh
        for dp in _camera_depth_candidates(root, cam, episode_id):
            if dp.exists():
                if dp.suffix == ".zarr" and dp.is_dir():
                    _sh.rmtree(dp, ignore_errors=True)
                else:
                    dp.unlink()
    meta_fp = root / "meta" / "episodes.jsonl"
    if meta_fp.is_file():
        keep: list[str] = []
        for line in meta_fp.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                if int(d.get("episode_id", -1)) == episode_id:
                    continue
            except json.JSONDecodeError:
                pass
            keep.append(line)
        meta_fp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
