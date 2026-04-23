"""Post-save 数据同步: rsync 录完的一条 episode 到 gf0/gf1。

设计点:
  * 异步: recorder.save() 返回前 spawn 后台线程, 不阻塞 UI
  * 幂等: rsync -a 只传递差异; 失败不影响下一次触发
  * 可观测: 所有 rsync 结果写 web/data_manager/logs/sync.log, 带 remote / 用时 / rc
  * 可关闭: KAI0_SYNC_ENABLED=0 整体禁用
  * 可重定向: KAI0_SYNC_REMOTES (JSON list) 覆盖默认 gf0/gf1
  * 可补偿: sync_remaining() 手动一次性同步已存在的 task/date/subset (见 bottom)

rsync 策略:
  - src:  {DATA_ROOT}/{task}/{date}/{subset}    (无 trailing slash, src 目录本身被推)
  - dst:  {user}@{host}:{dest_root}/{task}/{date}/
  - --mkpath 自动建远端父目录 (rsync 3.2.3+, sim01 是 3.2.7 ✓)
  - 不加 --delete: 本地误删时不传播到云端
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import DATA_ROOT


# ----------------------------- config -------------------------------------
@dataclass(frozen=True)
class Remote:
    name: str
    user: str
    host: str
    port: int
    dest_root: str


# Default push targets. Adding a remote only needs (name, user, host, port, dest_root).
#
# gf0-vepfs: gf0 和 gf1 的 /vePFS/visrobot01 是同一块 gpfs 卷 (fs_vepfs-cnsh075262e1f815),
#   推 gf0 = 两台同时到。rsync 3.2.7. 一次性: sudo chown -R tim:tim /vePFS/visrobot01/KAI0.
# bja2-vla:  单独的开发机, root SSH, rsync 3.1.3 (不支持 --mkpath, 见 _rsync_cmd 里用
#   --rsync-path 兼容老版本).
DEFAULT_REMOTES: list[Remote] = [
    Remote(name="gf0-vepfs", user="tim", host="14.103.44.161", port=55555,
           dest_root="/vePFS/visrobot01/KAI0"),
    Remote(name="bja2-vla", user="root", host="115.190.97.39", port=37686,
           dest_root="/VLA-Data/scripts/lianqing/data/bipiper_dataset"),
]


def _load_remotes() -> list[Remote]:
    raw = os.environ.get("KAI0_SYNC_REMOTES", "")
    if not raw:
        return list(DEFAULT_REMOTES)
    try:
        items = json.loads(raw)
        return [Remote(**it) for it in items]
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logging.getLogger(__name__).error(
            "invalid KAI0_SYNC_REMOTES=%r, falling back to defaults: %s", raw, e
        )
        return list(DEFAULT_REMOTES)


ENABLED: bool = os.environ.get("KAI0_SYNC_ENABLED", "1") == "1"
REMOTES: list[Remote] = _load_remotes()
RETRIES: int = int(os.environ.get("KAI0_SYNC_RETRIES", "3"))
BACKOFF_BASE_S: float = float(os.environ.get("KAI0_SYNC_BACKOFF_S", "2"))
TIMEOUT_S: int = int(os.environ.get("KAI0_SYNC_TIMEOUT_S", "600"))  # 单次 rsync 上限

# 独立文件 logger, 避免 uvicorn access log 里塞满 rsync 统计
_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "sync.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_sync_log = logging.getLogger("kai0.sync")
if not _sync_log.handlers:
    h = logging.FileHandler(_LOG_PATH)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _sync_log.addHandler(h)
    _sync_log.setLevel(logging.INFO)
    _sync_log.propagate = False


# ----------------------------- workers ------------------------------------
@dataclass
class _Job:
    src: Path
    task: str
    date: str
    subset: str
    remotes: list[Remote] = field(default_factory=list)


def _rsync_cmd(src: Path, remote: Remote, task: str, date: str) -> list[str]:
    """目录级推送: src 是 subset 目录 (不带 trailing slash), 会落到 remote 的
    dest_root/task/date/ 下作为同名子目录。

    用 --rsync-path='mkdir -p X && rsync' 而非 --mkpath, 兼容 rsync 3.1.x
    (bja2 是 3.1.3, --mkpath 需要 3.2.3+). gf0 是 3.2.7 用哪种都行。"""
    ssh = (
        f"ssh -p {remote.port} -o BatchMode=yes "
        f"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
    )
    # 目标父目录 (date 级), src 的 basename (subset) 会被 rsync 自动挂到它下面
    dst_parent_path = f"{remote.dest_root}/{task}/{date}"
    dst_parent = f"{remote.user}@{remote.host}:{dst_parent_path}/"
    remote_wrap = f"mkdir -p {shlex.quote(dst_parent_path)} && rsync"
    return [
        "rsync", "-a", "--partial",
        f"--rsync-path={remote_wrap}",
        f"--timeout={TIMEOUT_S}",
        "-e", ssh,
        str(src),  # 不带 trailing slash, rsync 会在 dst 下创建 subset/ 这一层
        dst_parent,
    ]


def _run_one(cmd: list[str]) -> tuple[int, str]:
    """跑一条 rsync, 返回 (rc, short_summary)。stderr 合进 stdout, 截 200 字。"""
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_S + 30,
        )
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    tail = (p.stdout + p.stderr).strip().splitlines()[-1:]
    return p.returncode, (tail[0] if tail else "")


def _push_one_remote(src: Path, task: str, date: str, subset: str, remote: Remote) -> None:
    cmd = _rsync_cmd(src, remote, task, date)
    for attempt in range(1, RETRIES + 1):
        t0 = time.time()
        rc, summary = _run_one(cmd)
        dt_ms = int((time.time() - t0) * 1000)
        tag = f"[{remote.name}] {task}/{date}/{subset}"
        if rc == 0:
            _sync_log.info("%s ok in %d ms (attempt %d)", tag, dt_ms, attempt)
            return
        _sync_log.warning(
            "%s rc=%d attempt=%d/%d dt=%d ms: %s | cmd=%s",
            tag, rc, attempt, RETRIES, dt_ms, summary, shlex.join(cmd),
        )
        if attempt < RETRIES:
            time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)))
    _sync_log.error("%s FAILED after %d attempts", tag, RETRIES)


def _worker(job: _Job) -> None:
    """给每个 remote 并行发一条 rsync, 彼此独立。"""
    threads = []
    for r in job.remotes:
        t = threading.Thread(
            target=_push_one_remote,
            args=(job.src, job.task, job.date, job.subset, r),
            name=f"sync-{r.name}-{job.task}-{job.date}",
            daemon=True,
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


# ----------------------------- public API ---------------------------------
def sync_episode_subset(task: str, date: str, subset: str) -> None:
    """recorder.save() 调这个: 把刚写完一条 episode 的整个 subset 目录推到远端。
    整个 subset 推 (而不是单个 episode) 是为了捎带 meta/ 下的 episodes.jsonl /
    tasks.jsonl / info.json 更新 —— rsync -a 会自动跳过未变化的 mp4/parquet。
    """
    if not ENABLED:
        return
    if not REMOTES:
        _sync_log.warning("no remotes configured; skipping sync of %s/%s/%s",
                          task, date, subset)
        return
    src = DATA_ROOT / task / date / subset
    if not src.is_dir():
        # 迁移前的老扁平布局: <DATA_ROOT>/<task>_<date>/<subset>
        flat = DATA_ROOT / f"{task}_{date}" / subset
        if flat.is_dir():
            _sync_log.info("old flat layout detected; syncing %s → %s/%s", flat, task, date)
            src = flat
        else:
            _sync_log.error("sync source missing: %s (and no flat fallback)", src)
            return
    job = _Job(src=src, task=task, date=date, subset=subset, remotes=REMOTES)
    threading.Thread(target=_worker, args=(job,), name=f"sync-main-{task}-{date}-{subset}",
                     daemon=True).start()


def sync_all(only_task: str | None = None) -> int:
    """一次性把 DATA_ROOT 下所有 task/date/subset 推到远端 (同步性, 阻塞)。
    用于首次搭建 / 迁移完成后全量对齐。返回任务数。"""
    if not REMOTES:
        return 0
    from .layout import path_to_compound, split_compound, glob_all_episodes
    seen: set[tuple[str, str, str]] = set()  # (task, date, subset)
    for pq in glob_all_episodes():
        parsed = path_to_compound(pq)
        if parsed is None:
            continue
        compound, subset = parsed
        sp = split_compound(compound)
        if sp is None:
            continue
        task, date = sp
        if only_task and task != only_task:
            continue
        seen.add((task, date, subset))
    for task, date, subset in sorted(seen):
        sync_episode_subset(task, date, subset)
    return len(seen)


def recent_log_tail(n: int = 50) -> list[str]:
    """返回最近 n 行 sync.log, 供 UI / 调试用。"""
    if not _LOG_PATH.exists():
        return []
    try:
        with _LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def status() -> dict:
    """给 /api/sync/status 用的摘要。"""
    return {
        "enabled": ENABLED,
        "remotes": [
            {"name": r.name, "host": r.host, "port": r.port, "dest_root": r.dest_root}
            for r in REMOTES
        ],
        "log_path": str(_LOG_PATH),
        "log_tail": [ln.rstrip() for ln in recent_log_tail(20)],
        "ts": datetime.utcnow().isoformat() + "Z",
    }
