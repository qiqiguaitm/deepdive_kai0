#!/usr/bin/env python3
"""Build mixed_hard mirror dirs on uc01 NFS for exp1.

Same as build_mixed_hard.py but uses uc-local source dataset paths:
  kai0_base   = /data/shared/ubuntu/workspace/dataset/Kai0_official/Task_A/base
  kai0_dagger = /data/shared/ubuntu/workspace/dataset/Kai0_official/Task_A/dagger
  vis_v2_merged = /data/shared/ubuntu/workspace/dataset/Task_A/vis_v2_merged   (rsync'd)

Outputs go to /data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/mixed_hard/
(on uc01 NFS, visible to uc02). Same patched tasks.jsonl logic ("kai " / "vis " prefix).

Run on uc01: ssh uc01 'python3 /data/.../xvla/scripts/build_mixed_hard_uc.py'
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path

KAI = "kai Flatten and fold the cloth."
VIS = "vis Flatten and fold the cloth."

UC_KAI_ROOT = Path("/data/shared/ubuntu/workspace/dataset/Kai0_official/Task_A")
UC_VIS_ROOT = Path("/data/shared/ubuntu/workspace/dataset/Task_A")
DST_ROOT = Path("/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/mixed_hard")

# (src_path, dst_name, domain_prompt)
SOURCES = [
    (UC_KAI_ROOT / "base",    "kai0_base",     KAI),
    (UC_KAI_ROOT / "dagger",  "kai0_dagger",   KAI),
    (UC_VIS_ROOT / "vis_v2_merged", "vis_v2_merged", VIS),
]


def mirror_one(src: Path, dst: Path, prompt: str):
    if dst.exists():
        sys.exit(f"dst exists: {dst} — use --force or rm first")
    dst.mkdir(parents=True)

    for sub in ("data", "videos"):
        sp = src / sub
        if sp.is_dir():
            os.symlink(sp.resolve(), dst / sub)

    (dst / "meta").mkdir()
    for f in ("info.json", "episodes.jsonl", "episodes_stats.jsonl"):
        sf = src / "meta" / f
        if sf.is_file():
            shutil.copy(sf, dst / "meta" / f)
    (dst / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": prompt}) + "\n"
    )

    for f in ("norm_stats.json", ".kai0_ts_validated"):
        sf = src / f
        if sf.is_file():
            shutil.copy(sf, dst / f) if not sf.is_symlink() else os.symlink(sf.resolve(), dst / f)
    (dst / ".kai0_ts_validated").touch()

    print(f"  ✓ {dst}  prompt='{prompt}'  src={src}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for src, name, prompt in SOURCES:
        if not src.is_dir():
            sys.exit(f"src missing: {src}")
        dst = DST_ROOT / name
        if args.force and dst.exists():
            shutil.rmtree(dst)
        mirror_one(src, dst, prompt)

    print(f"\ndone. {len(SOURCES)} mirrors at {DST_ROOT}")


if __name__ == "__main__":
    main()
