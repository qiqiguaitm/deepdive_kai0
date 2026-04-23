"""On-disk layout helpers.

历史: 采集数据原来在 `<DATA_ROOT>/<Task_X_YYYY-MM-DD>/<subset>/...` 扁平布局;
现在改成 `<DATA_ROOT>/<Task_X>/<YYYY-MM-DD>/<subset>/...` 层级布局, 同一个
task 的不同日期聚在一个父目录下, 方便按 task 做训练 / 同步.

约定:
  * 内存 / SQLite / API URL / UI 里依然用 **compound** 形式 "Task_X_YYYY-MM-DD"
    作为 task_id (最少改动面), 只是磁盘路径不同.
  * 所有构造/解析盘路径的点必须过这两个函数 (`compound_to_root`, `path_to_compound`),
    不要直接 `DATA_ROOT / task_id`.
  * 写新 episode 时一律走新布局; 读既有 episode 时先查新, 没有再回退老扁平路径,
    支持"一部分已迁移, 一部分还没"的中间态.
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path

from .config import DATA_ROOT

# "任意 task 名_YYYY-MM-DD" — 锚在末尾, 避免误吞 task 名里的 '_'
_DATE_SUFFIX_RE = re.compile(r"^(.+)_(\d{4}-\d{2}-\d{2})$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def today_compound(task: str) -> str:
    """`'Task_A'` → `'Task_A_2026-04-16'` (今日日期后缀)."""
    return f"{task}_{datetime.date.today().strftime('%Y-%m-%d')}"


def split_compound(compound: str) -> tuple[str, str] | None:
    """`'Task_A_2026-04-16'` → `('Task_A', '2026-04-16')`; 没有日期后缀 → None."""
    m = _DATE_SUFFIX_RE.match(compound)
    return (m.group(1), m.group(2)) if m else None


def compound_to_root(compound: str) -> Path:
    """Compound task_id → 盘上 **<task>/<date>** 目录.

    选择策略:
      1. 如果有日期后缀:
         - `DATA_ROOT/<task>/<date>` 存在 → 用新布局
         - 否则若 `DATA_ROOT/<compound>` 存在 → 用老扁平 (未迁移)
         - 都不存在 → 返回新布局 (caller 会在写新 episode 时 mkdir)
      2. 无日期后缀 (例如训练产出的 'Task_A'): 退到 `DATA_ROOT/<compound>`
    """
    split = split_compound(compound)
    if split is None:
        return DATA_ROOT / compound  # no date → flat, as-is
    task, date = split
    new = DATA_ROOT / task / date
    if new.exists():
        return new
    old = DATA_ROOT / compound
    if old.exists():
        return old
    return new  # default for fresh writes


def compound_to_subset_root(compound: str, subset: str) -> Path:
    """`'Task_A_2026-04-16', 'base'` → `DATA_ROOT/Task_A/2026-04-16/base` (or old)."""
    return compound_to_root(compound) / subset


def new_task_subset_root(task: str, subset: str) -> Path:
    """写新 episode 用: 一律走新布局 `<DATA_ROOT>/<task>/<today>/<subset>`."""
    return DATA_ROOT / task / datetime.date.today().strftime("%Y-%m-%d") / subset


def path_to_compound(p: Path) -> tuple[str, str] | None:
    """盘上一个 parquet 文件路径 → `(compound_task_id, subset)`.

    支持:
      新布局  `.../TASK/DATE/SUBSET/data/chunk-000/episode_NNNNNN.parquet`
      老布局  `.../TASK_DATE/SUBSET/data/chunk-000/episode_NNNNNN.parquet`
    None 表示非法路径 (不在 DATA_ROOT 下 / 层数不对 / 非 episode).
    """
    try:
        rel = p.resolve().relative_to(DATA_ROOT)
    except ValueError:
        return None
    parts = rel.parts
    # 至少要有 data/chunk-*/*.parquet 这三段
    if len(parts) < 4 or parts[-3] != "data" or not parts[-2].startswith("chunk-"):
        return None
    # 前缀: ... / TASK [/ DATE] / SUBSET / data
    prefix = parts[:-3]  # TASK[/DATE]/SUBSET
    if len(prefix) == 2:  # 老布局: (compound, subset)
        compound, subset = prefix
        return compound, subset
    if len(prefix) == 3:  # 新布局: (task, date, subset) — 要求 date 是 YYYY-MM-DD
        task, date, subset = prefix
        if _DATE_RE.match(date):
            return f"{task}_{date}", subset
    return None


def glob_all_episodes():
    """遍历 DATA_ROOT, yield 每个 episode.parquet 的 Path (新+老布局一起)."""
    # 老布局: <DATA_ROOT>/*/*/data/chunk-*/episode_*.parquet  (深度 5)
    # 新布局: <DATA_ROOT>/*/*/*/data/chunk-*/episode_*.parquet (深度 6)
    yield from DATA_ROOT.glob("*/*/data/chunk-*/episode_*.parquet")
    yield from DATA_ROOT.glob("*/*/*/data/chunk-*/episode_*.parquet")
