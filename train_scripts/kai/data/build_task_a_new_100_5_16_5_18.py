#!/usr/bin/env python3
"""Merge raw kai0/vis 5/16-v2 + 5/18-v2 dirs into a single lerobot v2.1 dataset.

Source: /data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/vis_base/2026-05-{16,18}-v2/  (NFS shared)
Splits 10 episodes from the LAST 10 of 5/18-v2 as held-out val. Output:
  TRAIN: .../self_built/A_new_100_5_16_5_18/      (92 ep)
  VAL:   .../self_built/A_new_100_5_16_5_18_val/  (10 ep)

Schema conversion: raw uses `episode_id`, lerobot wants `episode_index`.
Output uses symlinks for parquet + videos (zero copy).

Run ONCE from uc01 (NFS shared paths). Then:
    python gen_episodes_stats.py <train_dir>
    python gen_episodes_stats.py <val_dir>
    python compute_norm_states_fast.py --config-name <cfg>
"""
from __future__ import annotations
import json, sys, os, shutil
from pathlib import Path

CAMERAS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")
RAW_CAM = {
    "observation.images.top_head":   "top_head",
    "observation.images.hand_left":  "hand_left",
    "observation.images.hand_right": "hand_right",
}
SRC_ROOT = Path("/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/vis_base/v2")
SRC_DIRS = [SRC_ROOT / "2026-05-16-v2", SRC_ROOT / "2026-05-18-v2"]
DST_TRAIN = Path("/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_100_5_16_5_18")
DST_VAL   = Path("/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_100_5_16_5_18_val")
N_VAL = 10  # last 10 episodes (from 5/18) held out for inline-eval


def _prep(dst: Path):
    if dst.exists():
        sys.exit(f"dst exists: {dst} (delete first)")
    dst.mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir()
    for cam in CAMERAS:
        (dst / "videos" / "chunk-000" / cam).mkdir(parents=True)


def _link_ep(dst: Path, new_idx: int, sd: Path, old_idx: int, chunks_size: int):
    """Symlink parquet + 3 mp4s for one episode. Returns (success, missing_videos)."""
    old_chunk = old_idx // chunks_size
    sp = sd / f"data/chunk-{old_chunk:03d}/episode_{old_idx:06d}.parquet"
    if not sp.is_file():
        print(f"  skip missing parquet: {sp}", file=sys.stderr)
        return False
    os.symlink(sp.resolve(), dst / f"data/chunk-000/episode_{new_idx:06d}.parquet")
    for cam, raw in RAW_CAM.items():
        sv = sd / f"videos/chunk-{old_chunk:03d}/{raw}/episode_{old_idx:06d}.mp4"
        if not sv.is_file():
            print(f"  skip missing video: {sv}", file=sys.stderr)
            continue
        os.symlink(sv.resolve(), dst / f"videos/chunk-000/{cam}/episode_{new_idx:06d}.mp4")
    return True


def _write_meta(dst: Path, eps: list, info_ref: dict, tasks_src: Path):
    with (dst / "meta/episodes.jsonl").open("w") as f:
        for ep in eps:
            f.write(json.dumps(ep) + "\n")
    shutil.copy(tasks_src, dst / "meta/tasks.jsonl")
    info = json.loads(json.dumps(info_ref))
    n = len(eps)
    info["total_episodes"] = n
    info["total_frames"] = sum(e["length"] for e in eps)
    info["total_videos"] = n * len(CAMERAS)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{n}"}
    if "depth_path" in info:
        del info["depth_path"]
    if "features" in info:
        info["features"] = {k: v for k, v in info["features"].items() if "depth" not in k.lower()}
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2))


def main():
    _prep(DST_TRAIN)
    _prep(DST_VAL)

    # First pass: enumerate all (src_dir, old_idx) pairs in deterministic order
    all_eps = []  # list of (sd, old_idx, ep_meta, chunks_size, info)
    info_ref = None
    for sd in SRC_DIRS:
        if not sd.is_dir():
            sys.exit(f"missing source: {sd}")
        info = json.load((sd / "meta/info.json").open())
        info_ref = info_ref or info
        chunks_size = info.get("chunks_size", 1000)
        ep_lines = [json.loads(l) for l in (sd / "meta/episodes.jsonl").open()]
        ep_by_idx = {e["episode_id"]: e for e in ep_lines}
        for old_idx in sorted(ep_by_idx.keys()):
            all_eps.append((sd, old_idx, ep_by_idx[old_idx], chunks_size))

    n_total = len(all_eps)
    # Take the LAST N_VAL episodes as val; rest as train. With 5/16 (2) + 5/18 (100),
    # this puts the last 10 of 5/18 into val (and 5/16 fully in train).
    train_eps_src = all_eps[:-N_VAL]
    val_eps_src   = all_eps[-N_VAL:]
    print(f"split: total {n_total} → train {len(train_eps_src)} + val {len(val_eps_src)}")

    def _build(dst, eps_src):
        out_eps = []
        new_idx = 0
        for sd, old_idx, ep, chunks_size in eps_src:
            if not _link_ep(dst, new_idx, sd, old_idx, chunks_size):
                continue
            ep2 = ep.copy()
            ep2.pop("episode_id", None)
            ep2["episode_index"] = new_idx
            ep2["_src_dir"] = sd.name
            ep2["_src_idx"] = old_idx
            out_eps.append(ep2)
            new_idx += 1
        _write_meta(dst, out_eps, info_ref, SRC_DIRS[0] / "meta/tasks.jsonl")
        print(f"  {dst}: {new_idx} eps, {sum(e['length'] for e in out_eps)} frames")

    _build(DST_TRAIN, train_eps_src)
    _build(DST_VAL,   val_eps_src)
    print(f"\nNEXT:")
    print(f"  python gen_episodes_stats.py {DST_TRAIN}")
    print(f"  python gen_episodes_stats.py {DST_VAL}")


if __name__ == "__main__":
    main()
