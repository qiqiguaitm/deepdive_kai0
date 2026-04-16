"""独立统计服务：磁盘扫描 + watchdog 增量 + SQLite 索引。
不信任任何录制流的内存状态；以磁盘上真实存在的 episode_*.parquet 为唯一凭据。
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import DATA_ROOT, STATS_DB_PATH
from .models import EpisodeMeta, StatsBucket, StatsResponse

CAMERAS = ("top_head", "hand_left", "hand_right")
EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")


def _parse_episode_path(p: Path) -> Optional[dict]:
    """匹配 .../<TASK_DIR>/<subset>/data/chunk-XXX/episode_NNNNNN.parquet
    新布局 TASK_DIR = Task_<X>_<YYYY-MM-DD>; 旧布局 TASK_DIR = Task_<X>。
    我们把整段第一层目录名当作 'task_id', 让 UI/DB 自然按日期分组。"""
    try:
        rel = p.resolve().relative_to(DATA_ROOT)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 5 or parts[-3] != "data" or not parts[-2].startswith("chunk-"):
        return None
    m = EPISODE_RE.search(parts[-1])
    if not m:
        return None
    return {
        "task_id": parts[0],   # 含日期 e.g. "Task_A_2026-04-16"
        "subset": parts[1],
        "chunk": parts[-2],
        "episode_id": int(m.group(1)),
        "parquet_path": str(p),
    }


def _expected_videos(task_id: str, subset: str, chunk: str, ep_id: int) -> dict[str, Path]:
    base = DATA_ROOT / task_id / subset / "videos" / chunk
    return {cam: base / cam / f"episode_{ep_id:06d}.mp4" for cam in CAMERAS}


def _expected_depths(task_id: str, subset: str, chunk: str, ep_id: int) -> dict[str, Path]:
    """新增: depth zarr 目录路径, 同 video 平行, key 加 _depth 后缀."""
    base = DATA_ROOT / task_id / subset / "videos" / chunk
    return {cam: base / f"{cam}_depth" / f"episode_{ep_id:06d}.zarr" for cam in CAMERAS}


def _read_meta_lookup(task_id: str, subset: str) -> dict[int, dict]:
    """读 meta/episodes.jsonl，返回 episode_id -> meta dict。"""
    fp = DATA_ROOT / task_id / subset / "meta" / "episodes.jsonl"
    out: dict[int, dict] = {}
    if not fp.exists():
        return out
    try:
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "episode_id" in d:
                out[int(d["episode_id"])] = d
    except OSError:
        pass
    return out


class StatsService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._db = sqlite3.connect(STATS_DB_PATH, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS episodes (
                key TEXT PRIMARY KEY,
                task_id TEXT, subset TEXT, episode_id INTEGER,
                parquet_path TEXT, size_bytes INTEGER, duration_s REAL,
                operator TEXT, prompt TEXT, success INTEGER,
                created_at REAL, incomplete INTEGER, incomplete_reason TEXT,
                videos_json TEXT
            )"""
        )
        self._db.commit()
        self._last_scan_at = 0.0
        self._observer: Optional[Observer] = None

    # ---------- scanning ----------
    def full_rescan(self) -> int:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        rows: list[tuple] = []
        meta_cache: dict[tuple[str, str], dict[int, dict]] = {}
        # Glob 放宽到 */*: 既能匹配旧 Task_A 也能匹配新 Task_A_2026-04-16,
        # 第一层目录名整段进 task_id (见 _parse_episode_path 注释).
        for parquet in DATA_ROOT.glob("*/*/data/chunk-*/episode_*.parquet"):
            info = _parse_episode_path(parquet)
            if info is None:
                continue
            key = f"{info['task_id']}/{info['subset']}/{info['episode_id']}"
            videos = _expected_videos(info["task_id"], info["subset"], info["chunk"], info["episode_id"])
            missing = [c for c, v in videos.items() if not v.exists()]
            try:
                size = parquet.stat().st_size + sum(v.stat().st_size for v in videos.values() if v.exists())
                created = parquet.stat().st_mtime
            except OSError:
                continue
            mc = meta_cache.setdefault(
                (info["task_id"], info["subset"]),
                _read_meta_lookup(info["task_id"], info["subset"]),
            )
            m = mc.get(info["episode_id"], {})
            rows.append(
                (
                    key,
                    info["task_id"],
                    info["subset"],
                    info["episode_id"],
                    info["parquet_path"],
                    size,
                    float(m.get("duration_s", 0.0)),
                    str(m.get("operator", "")),
                    str(m.get("prompt", "")),
                    1 if m.get("success", True) else 0,
                    created,
                    1 if missing else 0,
                    ("missing_videos:" + ",".join(missing)) if missing else None,
                    json.dumps({c: str(v) for c, v in videos.items()}),
                )
            )
        with self._lock:
            cur = self._db.cursor()
            cur.execute("DELETE FROM episodes")
            cur.executemany(
                "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
            )
            self._db.commit()
            self._last_scan_at = time.time()
        return len(rows)

    def upsert_one(self, parquet: Path) -> bool:
        info = _parse_episode_path(parquet)
        if info is None or not parquet.exists():
            return False
        key = f"{info['task_id']}/{info['subset']}/{info['episode_id']}"
        videos = _expected_videos(info["task_id"], info["subset"], info["chunk"], info["episode_id"])
        missing = [c for c, v in videos.items() if not v.exists()]
        try:
            size = parquet.stat().st_size + sum(v.stat().st_size for v in videos.values() if v.exists())
            created = parquet.stat().st_mtime
        except OSError:
            return False
        m = _read_meta_lookup(info["task_id"], info["subset"]).get(info["episode_id"], {})
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key, info["task_id"], info["subset"], info["episode_id"],
                    str(parquet), size, float(m.get("duration_s", 0.0)),
                    str(m.get("operator", "")), str(m.get("prompt", "")),
                    1 if m.get("success", True) else 0, created,
                    1 if missing else 0,
                    ("missing_videos:" + ",".join(missing)) if missing else None,
                    json.dumps({c: str(v) for c, v in videos.items()}),
                ),
            )
            self._db.commit()
        return True

    def remove_by_path(self, parquet: Path) -> None:
        info = _parse_episode_path(parquet)
        if info is None:
            return
        key = f"{info['task_id']}/{info['subset']}/{info['episode_id']}"
        with self._lock:
            self._db.execute("DELETE FROM episodes WHERE key=?", (key,))
            self._db.commit()

    # ---------- queries ----------
    def stats(self) -> StatsResponse:
        now = time.time()
        today_start = time.mktime(time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d"))
        week_start = today_start - 6 * 86400
        with self._lock:
            cur = self._db.cursor()
            total = cur.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            today = cur.execute("SELECT COUNT(*) FROM episodes WHERE created_at>=?", (today_start,)).fetchone()[0]
            week = cur.execute("SELECT COUNT(*) FROM episodes WHERE created_at>=?", (week_start,)).fetchone()[0]
            incomplete = cur.execute("SELECT COUNT(*) FROM episodes WHERE incomplete=1").fetchone()[0]
            tot_dur, tot_size = cur.execute(
                "SELECT COALESCE(SUM(duration_s),0), COALESCE(SUM(size_bytes),0) FROM episodes"
            ).fetchone()
            buckets = lambda sql: [
                StatsBucket(key=str(k or ""), count=int(c))
                for k, c in cur.execute(sql).fetchall()
            ]
            by_ts = buckets(
                "SELECT task_id || '/' || subset, COUNT(*) FROM episodes GROUP BY task_id, subset ORDER BY 2 DESC"
            )
            by_op = buckets("SELECT operator, COUNT(*) FROM episodes GROUP BY operator ORDER BY 2 DESC")
            by_pr = buckets("SELECT prompt, COUNT(*) FROM episodes GROUP BY prompt ORDER BY 2 DESC")
            by_ok = [
                StatsBucket(key=("success" if k else "fail"), count=int(c))
                for k, c in cur.execute("SELECT success, COUNT(*) FROM episodes GROUP BY success").fetchall()
            ]
        return StatsResponse(
            total=total, today=today, this_week=week, incomplete=incomplete,
            total_duration_s=float(tot_dur), total_size_bytes=int(tot_size),
            by_task_subset=by_ts, by_operator=by_op, by_prompt=by_pr, by_success=by_ok,
            last_scan_at=self._last_scan_at or now,
        )

    def list_episodes(
        self,
        task_id: Optional[str] = None,
        subset: Optional[str] = None,
        operator: Optional[str] = None,
        success: Optional[bool] = None,
        prompt_kw: Optional[str] = None,
        limit: int = 200,
    ) -> list[EpisodeMeta]:
        sql = "SELECT * FROM episodes WHERE 1=1"
        args: list = []
        if task_id:
            sql += " AND task_id=?"; args.append(task_id)
        if subset:
            sql += " AND subset=?"; args.append(subset)
        if operator:
            sql += " AND operator=?"; args.append(operator)
        if success is not None:
            sql += " AND success=?"; args.append(1 if success else 0)
        if prompt_kw:
            sql += " AND prompt LIKE ?"; args.append(f"%{prompt_kw}%")
        sql += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
        with self._lock:
            cur = self._db.cursor()
            cols = [c[0] for c in cur.execute(sql, args).description] if False else None
            rows = cur.execute(sql, args).fetchall()
        out: list[EpisodeMeta] = []
        for r in rows:
            (key, task, sub, ep, pq, size, dur, op, prompt, ok, created, inc, inc_r, vjson) = r
            out.append(EpisodeMeta(
                episode_id=ep, task_id=task, subset=sub, prompt=prompt or "",
                operator=op or "", success=bool(ok), note="",
                duration_s=float(dur), size_bytes=int(size), created_at=float(created),
                parquet_path=pq, video_paths=json.loads(vjson),
                incomplete=bool(inc), incomplete_reason=inc_r,
            ))
        return out

    def next_episode_id(self, task_dir: str, subset: str) -> int:
        """直接扫盘目录下已存在的 episode_NNNNNN.parquet, 返回 max+1。

        以前是查 SQLite 里 task_id 的最大值; 但 SQLite 的 task_id 是 *已带日期的目录名*
        (e.g. 'Task_A_2026-04-16'), 而 recorder 一开始传进来的是 *裸 task* (e.g. 'Task_A'),
        匹配不上 → 永远返回 0 → 每次保存覆盖 episode_000000.

        改为扫真实目录: 不依赖 DB 一致性, 也能正确处理外部脚本删除/搬运 episode 的情况.
        task_dir 必须是已带日期后缀的目录名 (recorder.dated_task_name 计算)."""
        parquet_dir = DATA_ROOT / task_dir / subset / "data" / "chunk-000"
        max_id = -1
        if parquet_dir.is_dir():
            for p in parquet_dir.glob("episode_*.parquet"):
                m = EPISODE_RE.search(p.name)
                if m:
                    max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    # ---------- watchdog ----------
    def start_watcher(self) -> None:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        handler = _EpisodeWatcher(self)
        obs = Observer()
        obs.schedule(handler, str(DATA_ROOT), recursive=True)
        obs.daemon = True
        obs.start()
        self._observer = obs

    def stop_watcher(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)


class _EpisodeWatcher(FileSystemEventHandler):
    def __init__(self, svc: StatsService) -> None:
        self.svc = svc

    def on_any_event(self, event):
        p = Path(event.src_path)
        if p.suffix == ".parquet" and EPISODE_RE.search(p.name):
            if event.event_type == "deleted":
                self.svc.remove_by_path(p)
            else:
                self.svc.upsert_one(p)
        elif p.suffix == ".jsonl" and p.name == "episodes.jsonl":
            # meta 变更：触发该 task/subset 下的轻量重扫
            self.svc.full_rescan()


service = StatsService()
