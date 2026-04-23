#!/usr/bin/env python3
"""Migrate old flat layout → new hierarchical layout.

老布局:  <DATA_ROOT>/Task_X_YYYY-MM-DD/<subset>/...
新布局:  <DATA_ROOT>/Task_X/YYYY-MM-DD/<subset>/...

默认只打印计划 (dry-run); 加 --apply 才真 mv。
非数据文件 (`*.tar`, `ckpt_downloads/`, `task_e_parts/`, `*.py`, 隐藏文件) 原地保留。

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

# 不是数据 (目录或文件), 不动
_IGNORE_NAMES = {"ckpt_downloads", "task_e_parts"}
_IGNORE_SUFFIXES = {".tar", ".tar.gz", ".zip", ".py"}


def is_data_dir(entry: Path) -> bool:
    """判断是否是待迁移的老扁平数据目录, 即名字能匹配 Task_X_YYYY-MM-DD。"""
    if not entry.is_dir():
        return False
    if entry.name.startswith("."):
        return False
    if entry.name in _IGNORE_NAMES:
        return False
    return bool(_DATE_SUFFIX_RE.match(entry.name))


def plan_moves() -> list[tuple[Path, Path, int, int]]:
    """返回 [(src, dst, size_bytes, episode_count), ...]。"""
    plan: list[tuple[Path, Path, int, int]] = []
    if not DATA_ROOT.exists():
        return plan
    for entry in sorted(DATA_ROOT.iterdir()):
        if not is_data_dir(entry):
            continue
        m = _DATE_SUFFIX_RE.match(entry.name)
        assert m  # guaranteed by is_data_dir
        task, date = m.group(1), m.group(2)
        dst = DATA_ROOT / task / date
        # 统计体积 + episode 数量, 供人审
        size = 0
        episodes = 0
        for p in entry.rglob("episode_*.parquet"):
            episodes += 1
        for p in entry.rglob("*"):
            try:
                if p.is_file():
                    size += p.stat().st_size
            except OSError:
                pass
        plan.append((entry, dst, size, episodes))
    return plan


def fmt_gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def print_plan(plan: list[tuple[Path, Path, int, int]]) -> None:
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"{'src':<40} {'dst':<40} {'size':>10} {'episodes':>9}")
    print("-" * 105)
    total_size, total_eps = 0, 0
    for src, dst, size, eps in plan:
        print(
            f"{src.name:<40} {dst.relative_to(DATA_ROOT).as_posix():<40} "
            f"{fmt_gb(size):>10} {eps:>9}"
        )
        total_size += size
        total_eps += eps
    print("-" * 105)
    print(f"{'TOTAL':<40} {'':<40} {fmt_gb(total_size):>10} {total_eps:>9}")
    print()
    # 列出不会动的顶层项目, 让用户过一眼
    untouched = []
    for entry in sorted(DATA_ROOT.iterdir()):
        if is_data_dir(entry):
            continue
        untouched.append(entry.name + ("/" if entry.is_dir() else ""))
    if untouched:
        print("unchanged (non-data / already new layout):")
        for n in untouched:
            print(f"  · {n}")


def apply_moves(plan: list[tuple[Path, Path, int, int]]) -> None:
    for src, dst, _size, _eps in plan:
        if dst.exists():
            # 已经存在说明要么上次迁过一半, 要么同名冲突: 不合并, 保险起见跳过
            print(f"SKIP: dst exists, not overwriting: {dst}", file=sys.stderr)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        # 同盘 rename 零拷贝; 跨盘自动回落到 copy+remove (shutil.move 内部处理)
        print(f"mv {src} → {dst}")
        shutil.move(str(src), str(dst))


def verify_post_move(plan: list[tuple[Path, Path, int, int]]) -> int:
    """校验新路径下 episode 数和原计划一致。返回不一致的条目数。"""
    bad = 0
    for src, dst, _size, eps in plan:
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
        print(f"No old-layout directories found under {DATA_ROOT}.")
        return 0

    print_plan(plan)

    if not args.apply:
        print("\n[dry-run] re-run with --apply to execute.")
        return 0

    print("\napplying...")
    apply_moves(plan)
    bad = verify_post_move(plan)
    if bad:
        print(f"\n{bad} directory(s) failed post-move verification; inspect above.", file=sys.stderr)
        return 1
    print(f"\ndone; {len(plan)} directory(s) migrated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
