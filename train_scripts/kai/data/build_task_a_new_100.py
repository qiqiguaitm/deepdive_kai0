#!/usr/bin/env python3
"""Build A_new_100 = first 100 episodes (originals only, no mirror) from A_new_pure_200.

Source: /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200/
Dest:   /vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_100/

ep 0..99 in source are kind="original" (confirmed via meta/episodes.jsonl).
ep 100..199 are kind="mirror" — excluded here.

Layout (lerobot v2.1):
  data/chunk-000/episode_{0..99:06d}.parquet
  videos/chunk-000/{top_head,hand_left,hand_right}/episode_{0..99:06d}.mp4
  meta/{info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl}

Usage:
  python build_task_a_new_100.py [--src ...] [--dst ...]
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path

CAMERAS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200")
    p.add_argument("--dst", default="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_100")
    p.add_argument("--n", type=int, default=100, help="keep first N originals")
    p.add_argument("--symlink-video", action="store_true", help="symlink video files instead of copy (saves disk)")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        sys.exit(f"src not found: {src}")
    if dst.exists():
        sys.exit(f"dst already exists: {dst} (delete first)")

    dst.mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir()
    for cam in CAMERAS:
        (dst / "videos" / "chunk-000" / cam).mkdir(parents=True)

    # ---- meta filter ----
    src_meta = src / "meta"
    eps = []
    for line in (src_meta / "episodes.jsonl").open():
        d = json.loads(line)
        if d.get("kind") == "original" and d["episode_index"] < args.n:
            eps.append(d)
    assert len(eps) == args.n, f"expected {args.n} originals, got {len(eps)}"
    print(f"filtered {len(eps)} original episodes (ep 0..{args.n-1})")

    ep_stats_by_idx = {}
    for line in (src_meta / "episodes_stats.jsonl").open():
        d = json.loads(line)
        ep_stats_by_idx[d["episode_index"]] = d

    # ---- copy parquet + videos ----
    total_frames = 0
    for ep in eps:
        idx = ep["episode_index"]
        # parquet
        sname = f"episode_{idx:06d}.parquet"
        s = src / "data" / "chunk-000" / sname
        d = dst / "data" / "chunk-000" / sname
        shutil.copy(s, d)
        total_frames += ep["length"]
        # videos
        for cam in CAMERAS:
            sv = src / "videos" / "chunk-000" / cam / f"episode_{idx:06d}.mp4"
            dv = dst / "videos" / "chunk-000" / cam / f"episode_{idx:06d}.mp4"
            if args.symlink_video:
                os.symlink(sv.resolve(), dv)
            else:
                shutil.copy(sv, dv)

    # ---- meta out ----
    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for ep in eps:
            f.write(json.dumps(ep) + "\n")
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        for ep in eps:
            es = ep_stats_by_idx[ep["episode_index"]]
            f.write(json.dumps(es) + "\n")
    shutil.copy(src_meta / "tasks.jsonl", dst / "meta" / "tasks.jsonl")

    # info.json: patched total_episodes / total_frames / total_videos / splits
    info = json.loads((src_meta / "info.json").read_text())
    info["total_episodes"] = args.n
    info["total_frames"] = total_frames
    info["total_videos"] = args.n * len(CAMERAS)
    info["splits"] = {"train": f"0:{args.n}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # norm_stats: copy from source (computed on full 200 incl mirror)
    # Note: ideally recompute on 100 originals only, but lerobot accepts source-side stats.
    # We'll recompute later via openpi's compute_norm_stats.
    if (src / "norm_stats.json").exists():
        shutil.copy(src / "norm_stats.json", dst / "norm_stats.json")
        print("copied norm_stats.json from src (recompute optional)")
    if (src / "manifest.json").exists():
        shutil.copy(src / "manifest.json", dst / "manifest.json")

    print(f"done → {dst}")
    print(f"  episodes={args.n}  frames={total_frames}")


if __name__ == "__main__":
    main()
