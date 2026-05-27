#!/usr/bin/env python3
"""Build mirror dirs with patched tasks.jsonl for exp1 (hard prompt) mixed training.

For each source dataset (kai0_base, kai0_dagger, vis_v2_merged), create a sibling
under xvla/data/mixed_hard/<name>/ that:
  - symlinks data/, videos/, norm_stats.json from source (zero copy)
  - rewrites meta/tasks.jsonl with "<domain> Flatten and fold the cloth."
  - copies meta/info.json, episodes.jsonl, episodes_stats.jsonl from source

Usage:
  python build_mixed_hard.py [--force]
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path

KAI = "kai Flatten and fold the cloth."
VIS = "vis Flatten and fold the cloth."

DATA_ROOT = Path("/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A")
DST_ROOT = Path("/vePFS/tim/workspace/deepdive_kai0/xvla/data/mixed_hard")

# (src_name, domain_prompt)
SOURCES = [
    ("kai0_base", KAI),
    ("kai0_dagger", KAI),
    ("vis_v2_merged", VIS),
]


def mirror_one(src: Path, dst: Path, prompt: str):
    if dst.exists():
        sys.exit(f"dst exists: {dst} — use --force or rm first")
    dst.mkdir(parents=True)

    # symlink data/ and videos/ recursively at chunk level (cheap, no per-file)
    for sub in ("data", "videos"):
        sp = src / sub
        if sp.is_dir():
            os.symlink(sp.resolve(), dst / sub)

    # mirror meta dir as new dir + copy + override tasks.jsonl
    (dst / "meta").mkdir()
    for f in ("info.json", "episodes.jsonl", "episodes_stats.jsonl"):
        sf = src / "meta" / f
        if sf.is_file():
            shutil.copy(sf, dst / "meta" / f)
    # tasks.jsonl with patched prompt
    (dst / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": prompt}) + "\n"
    )

    # norm_stats.json + ts marker if present
    for f in ("norm_stats.json", ".kai0_ts_validated"):
        sf = src / f
        if sf.is_file():
            shutil.copy(sf, dst / f) if not sf.is_symlink() else os.symlink(sf.resolve(), dst / f)
    # always touch ts marker on the mirror so lerobot skips timestamp check
    (dst / ".kai0_ts_validated").touch()

    print(f"  ✓ {dst}  prompt='{prompt}'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for name, prompt in SOURCES:
        src = DATA_ROOT / name
        if not src.is_dir():
            sys.exit(f"src missing: {src}")
        dst = DST_ROOT / name
        if args.force and dst.exists():
            shutil.rmtree(dst)
        mirror_one(src, dst, prompt)

    print(f"\ndone. {len(SOURCES)} mirrors at {DST_ROOT}")


if __name__ == "__main__":
    main()
