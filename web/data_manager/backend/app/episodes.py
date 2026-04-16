"""历史 episode 浏览/读取（采集员可用）。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from .config import DATA_ROOT
from .stats_service import service as stats

SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-\.]+$")


def _safe_join(*parts: str) -> Path:
    for p in parts:
        if not SAFE_NAME.match(p):
            raise HTTPException(status_code=400, detail=f"unsafe path component: {p}")
    full = (DATA_ROOT.joinpath(*parts)).resolve()
    if not str(full).startswith(str(DATA_ROOT)):
        raise HTTPException(status_code=400, detail="path escapes DATA_ROOT")
    return full


def episode_video_path(task_id: str, subset: str, episode_id: int, camera: str) -> Path:
    if camera not in ("top_head", "hand_left", "hand_right"):
        raise HTTPException(status_code=400, detail="unknown camera")
    return _safe_join(task_id, subset, "videos", "chunk-000", camera, f"episode_{episode_id:06d}.mp4")


def episode_depth_zarr_path(task_id: str, subset: str, episode_id: int, camera: str) -> Path:
    if camera not in ("top_head", "hand_left", "hand_right"):
        raise HTTPException(status_code=400, detail="unknown camera")
    return _safe_join(task_id, subset, "videos", "chunk-000", f"{camera}_depth", f"episode_{episode_id:06d}.zarr")


def episode_meta(task_id: str, subset: str, episode_id: int) -> dict:
    fp = _safe_join(task_id, subset, "meta", "episodes.jsonl")
    if not fp.exists():
        raise HTTPException(status_code=404, detail="meta not found")
    for line in fp.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(d.get("episode_id", -1)) == episode_id:
            return d
    raise HTTPException(status_code=404, detail="episode not found")


def delete_episode(task_id: str, subset: str, episode_id: int) -> None:
    import shutil
    pq = _safe_join(task_id, subset, "data", "chunk-000", f"episode_{episode_id:06d}.parquet")
    if pq.exists():
        pq.unlink()
    for cam in ("top_head", "hand_left", "hand_right"):
        v = _safe_join(task_id, subset, "videos", "chunk-000", cam, f"episode_{episode_id:06d}.mp4")
        if v.exists():
            v.unlink()
        # depth zarr 是目录, 用 rmtree
        d = _safe_join(task_id, subset, "videos", "chunk-000", f"{cam}_depth", f"episode_{episode_id:06d}.zarr")
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    # 同步从 meta 中删除该条
    meta_fp = _safe_join(task_id, subset, "meta", "episodes.jsonl")
    if meta_fp.exists():
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
    stats.remove_by_path(pq)
