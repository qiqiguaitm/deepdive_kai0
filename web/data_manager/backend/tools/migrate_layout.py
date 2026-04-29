#!/usr/bin/env python3
"""Migrate older layouts → 当前 v2 (`<task>/<subset>/<date>`).

布局演变:
  v0 (扁平):   `<DATA_ROOT>/Task_X_YYYY-MM-DD/<subset>/...`
  v1 (按日期): `<DATA_ROOT>/Task_X/YYYY-MM-DD/<subset>/...`
  v2 (按子集): `<DATA_ROOT>/Task_X/<subset>/YYYY-MM-DD/...`   ← 当前

本工具同时迁移 v0 → v2 和 v1 → v2。默认 dry-run, --apply 真 mv。
非数据 (`*.tar`, `ckpt_downloads/`, `task_e_parts/`, `*.py`, 隐藏文件) 不动。

用法:
  python migrate_layout.py                     # dry-run, 默认 DATA_ROOT=/data1/DATA_IMP/KAI0
  python migrate_layout.py --apply             # 真迁移
  KAI0_DATA_ROOT=/tmp/foo python migrate_layout.py --apply
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

DATA_ROOT = Path(os.environ.get("KAI0_DATA_ROOT", "/data1/DATA_IMP/KAI0")).resolve()

# "Task_X_YYYY-MM-DD" — 锚在末尾, 完整目录名才能匹配
_DATE_SUFFIX_RE = re.compile(r"^(.+)_(\d{4}-\d{2}-\d{2})$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# 顶层 / 二层不动的名字 (非数据 task)
_IGNORE_NAMES = {"ckpt_downloads", "task_e_parts"}
_IGNORE_SUFFIXES = {".tar", ".tar.gz", ".zip", ".py"}


def _is_ignorable(p: Path) -> bool:
    if p.name.startswith("."):
        return True
    if p.name in _IGNORE_NAMES:
        return True
    if p.is_file() and p.suffix in _IGNORE_SUFFIXES:
        return True
    return False


def _looks_like_subset_root(p: Path) -> bool:
    """子集目录的形态: 内部至少有 data/ 或 meta/ 或 videos/ 之一。"""
    if not p.is_dir():
        return False
    for child in ("data", "meta", "videos"):
        if (p / child).is_dir():
            return True
    return False


def plan_moves() -> list[tuple[Path, Path, int, int, str]]:
    """返回 [(src, dst, size_bytes, episode_count, kind), ...]; kind ∈ {'v0', 'v1'}."""
    plan: list[tuple[Path, Path, int, int, str]] = []
    if not DATA_ROOT.exists():
        return plan
    for entry in sorted(DATA_ROOT.iterdir()):
        if _is_ignorable(entry) or not entry.is_dir():
            continue
        # v0: `Task_X_YYYY-MM-DD/<subset>` —— 顶层带日期后缀
        m = _DATE_SUFFIX_RE.match(entry.name)
        if m:
            task, date = m.group(1), m.group(2)
            for sub in sorted(entry.iterdir()):
                if not _looks_like_subset_root(sub):
                    continue
                dst = DATA_ROOT / task / sub.name / date
                plan.append((sub, dst, *_count(sub), "v0"))
            continue
        # v1: `Task_X/YYYY-MM-DD/<subset>` —— 顶层是 task, 二层是日期
        task = entry.name
        for date_dir in sorted(entry.iterdir()):
            if not date_dir.is_dir() or not _DATE_RE.match(date_dir.name):
                continue
            date = date_dir.name
            for sub in sorted(date_dir.iterdir()):
                if not _looks_like_subset_root(sub):
                    continue
                dst = DATA_ROOT / task / sub.name / date
                if dst.resolve() == sub.resolve():
                    continue  # already v2 by accident
                plan.append((sub, dst, *_count(sub), "v1"))
    return plan


def _count(d: Path) -> tuple[int, int]:
    """(size_bytes, episode_count) for a subset dir。"""
    size = 0
    eps = 0
    for p in d.rglob("episode_*.parquet"):
        eps += 1
    for p in d.rglob("*"):
        try:
            if p.is_file():
                size += p.stat().st_size
        except OSError:
            pass
    return size, eps


def fmt_gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def print_plan(plan: list[tuple[Path, Path, int, int, str]]) -> None:
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"{'kind':<5} {'src':<50} {'dst':<45} {'size':>10} {'eps':>5}")
    print("-" * 120)
    total_size, total_eps = 0, 0
    for src, dst, size, eps, kind in plan:
        print(
            f"{kind:<5} {src.relative_to(DATA_ROOT).as_posix():<50} "
            f"{dst.relative_to(DATA_ROOT).as_posix():<45} {fmt_gb(size):>10} {eps:>5}"
        )
        total_size += size
        total_eps += eps
    print("-" * 120)
    print(f"{'TOTAL':<5} {'':<50} {'':<45} {fmt_gb(total_size):>10} {total_eps:>5}")


def apply_moves(plan: list[tuple[Path, Path, int, int, str]]) -> None:
    for src, dst, _size, _eps, _kind in plan:
        if dst.exists():
            print(f"SKIP: dst exists, not overwriting: {dst}", file=sys.stderr)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        # 同盘 rename 零拷贝; 跨盘自动回落到 copy+remove (shutil.move 内部处理)
        print(f"mv {src} → {dst}")
        shutil.move(str(src), str(dst))


def cleanup_empty_v0_v1(plan: list[tuple[Path, Path, int, int, str]]) -> None:
    """迁完后试着删空的旧父目录 (v0 的 Task_X_DATE/, v1 的 Task_X/DATE/)."""
    seen: set[Path] = set()
    for src, _dst, _size, _eps, kind in plan:
        if kind == "v0":
            seen.add(src.parent)  # Task_X_DATE/
        elif kind == "v1":
            seen.add(src.parent)  # Task_X/DATE/
    for d in sorted(seen, key=lambda p: -len(p.parts)):
        try:
            d.rmdir()
            print(f"rmdir {d}")
        except OSError:
            pass  # 还有非数据残留, 不强删


def verify_post_move(plan: list[tuple[Path, Path, int, int, str]]) -> int:
    bad = 0
    for src, dst, _size, eps, _kind in plan:
        if not dst.exists():
            continue
        found = sum(1 for _ in dst.rglob("episode_*.parquet"))
        if found != eps:
            print(f"VERIFY FAIL: {dst} has {found} episodes, expected {eps}", file=sys.stderr)
            bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="execute the mv; without this it's a dry-run")
    args = ap.parse_args()

    plan = plan_moves()
    if not plan:
        print(f"No legacy-layout subsets found under {DATA_ROOT}.")
        return 0

    print_plan(plan)

    if not args.apply:
        print("\n[dry-run] re-run with --apply to execute.")
        return 0

    print("\napplying...")
    apply_moves(plan)
    cleanup_empty_v0_v1(plan)
    bad = verify_post_move(plan)
    if bad:
        print(f"\n{bad} subset(s) failed post-move verification; inspect above.", file=sys.stderr)
        return 1
    print(f"\ndone; {len(plan)} subset(s) migrated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
