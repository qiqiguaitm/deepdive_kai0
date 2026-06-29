"""历史 episode 浏览/读取（采集员可用）。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from .config import DATA_ROOT
from .layout import compound_to_subset_root
from .stats_service import service as stats

SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-\.]+$")


def _safe_tail(*parts: str) -> tuple[str, ...]:
    """URL 传进来的每段都要过白名单, 防 path traversal。返回原样 tuple 供 caller 拼。"""
    for p in parts:
        if not SAFE_NAME.match(p):
            raise HTTPException(status_code=400, detail=f"unsafe path component: {p}")
    return parts


def _subset_join(task_id: str, subset: str, *tail: str) -> Path:
    """`Task_A_2026-04-16, base, 'videos', 'chunk-000', ...` → 真实磁盘 Path,
    透明支持新层级 / 老扁平布局。同时做 traversal 校验。"""
    _safe_tail(task_id, subset, *tail)
    base = compound_to_subset_root(task_id, subset)
    full = base.joinpath(*tail).resolve()
    if not str(full).startswith(str(DATA_ROOT)):
        raise HTTPException(status_code=400, detail="path escapes DATA_ROOT")
    return full


def _camera_video_path(base: Path, cam: str, ep: int) -> Path:
    """Resolve video for a camera, trying both `observation.images.<cam>` (v4+
    compliant layout) and bare `<cam>` (v3 legacy). Returns the first hit."""
    name = f"episode_{ep:06d}.mp4"
    p = base / "videos" / "chunk-000" / f"observation.images.{cam}" / name
    if p.exists():
        return p
    return base / "videos" / "chunk-000" / cam / name


def _camera_depth_path(base: Path, cam: str, ep: int) -> Path:
    """Resolve depth zarr, trying `observation.depth.<cam>` then `<cam>_depth`."""
    name_zarr = f"episode_{ep:06d}.zarr"
    name_zip = name_zarr + ".zip"
    name_mkv = f"episode_{ep:06d}.mkv"
    # zarr directory
    p = base / "videos" / "chunk-000" / f"observation.depth.{cam}" / name_zarr
    if p.exists():
        return p
    q = base / "videos" / "chunk-000" / f"{cam}_depth" / name_zarr
    if q.exists():
        return q
    # zarr.zip (packed)
    pz = p.with_name(name_zip)
    if pz.exists():
        return pz
    qz = q.with_name(name_zip)
    if qz.exists():
        return qz
    # mkv (new ffv1)
    pm = p.with_name(name_mkv)
    if pm.exists():
        return pm
    qm = q.with_name(name_mkv)
    if qm.exists():
        return qm
    return q  # fallback (caller 404s)


def episode_video_path(task_id: str, subset: str, episode_id: int, camera: str) -> Path:
    if camera not in ("top_head", "hand_left", "hand_right"):
        raise HTTPException(status_code=400, detail="unknown camera")
    base = compound_to_subset_root(task_id, subset)
    return _camera_video_path(base, camera, episode_id)


def episode_depth_zarr_path(task_id: str, subset: str, episode_id: int, camera: str) -> Path:
    # 新录制只为 D435 头顶相机生成 depth zarr (见 recorder.DEPTH_CAMERAS);
    # hand_left / hand_right 仅为兼容历史数据保留可达路径, 新数据下这两个 cam
    # 走到 main.py 的 file_exists 检查即返回 404, 不会泄露其他目录.
    if camera not in ("top_head", "hand_left", "hand_right"):
        raise HTTPException(status_code=400, detail="unknown camera")
    base = compound_to_subset_root(task_id, subset)
    return _camera_depth_path(base, camera, episode_id)


def episode_meta(task_id: str, subset: str, episode_id: int) -> dict:
    fp = _subset_join(task_id, subset, "meta", "episodes.jsonl")
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
    base = compound_to_subset_root(task_id, subset)
    pq = base / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
    if pq.exists():
        pq.unlink()
    for cam in ("top_head", "hand_left", "hand_right"):
        for v in _candidate_video_paths(base, cam, episode_id):
            if v.exists():
                v.unlink()
        for zd in _candidate_depth_paths(base, cam, episode_id):
            if zd.exists():
                if zd.suffix == ".zarr" and zd.is_dir():
                    shutil.rmtree(zd, ignore_errors=True)
                else:
                    zd.unlink()
    # 同步从 meta 中删除该条
    meta_fp = base / "meta" / "episodes.jsonl"
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


def _candidate_video_paths(base: Path, cam: str, ep: int) -> list[Path]:
    name = f"episode_{ep:06d}.mp4"
    return [
        base / "videos" / "chunk-000" / f"observation.images.{cam}" / name,
        base / "videos" / "chunk-000" / cam / name,
    ]


def _candidate_depth_paths(base: Path, cam: str, ep: int) -> list[Path]:
    stem, zarr_n = f"episode_{ep:06d}", f"episode_{ep:06d}.zarr"
    return [
        base / "videos" / "chunk-000" / f"observation.depth.{cam}" / zarr_n,
        base / "videos" / "chunk-000" / f"observation.depth.{cam}" / (stem + ".zarr.zip"),
        base / "videos" / "chunk-000" / f"observation.depth.{cam}" / (stem + ".mkv"),
        base / "videos" / "chunk-000" / f"{cam}_depth" / zarr_n,
        base / "videos" / "chunk-000" / f"{cam}_depth" / (stem + ".zarr.zip"),
        base / "videos" / "chunk-000" / f"{cam}_depth" / (stem + ".mkv"),
    ]
    # 同步从 meta 中删除该条
    meta_fp = _subset_join(task_id, subset, "meta", "episodes.jsonl")
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
