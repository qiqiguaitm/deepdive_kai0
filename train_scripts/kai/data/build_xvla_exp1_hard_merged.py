#!/usr/bin/env python3
"""Pre-merge 3 mixed_hard sources into a single lerobot dataset for exp1.

Why: the multi-dataset code path (`_create_concat_torch_dataset` with 3 LeRobotDataset
instances + 3x metadata init + per-source tolerance/index checks) makes uc02's
DataLoader startup too slow, so uc01's NCCL clique init times out before uc02
reaches the JIT. Successful pi05init cluster training (24 GPU) used the single-
dataset code path. By pre-merging here we use that same path.

Critical: lerobot's __getitem__ reads `episode_index` from the PARQUET (not from
meta/episodes.jsonl), so the parquet column MUST match the merged sequential
episode index. Same for `index` (global frame index) and `task_index`.

Sources (uc01 NFS shared):
  /data/.../mixed_hard/kai0_base       (3055 ep, prompt "kai Flatten and fold the cloth.")
  /data/.../mixed_hard/kai0_dagger     (3457 ep, "kai ...")
  /data/.../mixed_hard/vis_v2_merged   (895 ep, "vis Flatten and fold the cloth.")

Destination single lerobot dataset:
  /data/.../self_built/xvla_exp1_hard_merged/
    data/chunk-000/episode_NNNNNN.parquet  rewritten with corrected
                                            (episode_index, index, task_index)
    videos/chunk-000/<cam>/episode_NNNNNN.mp4  all symlinks
    meta/tasks.jsonl   2 entries: 0=kai prompt, 1=vis prompt
    meta/episodes.jsonl, episodes_stats.jsonl, info.json
"""
from __future__ import annotations
import json, sys, os, shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")
SRC_ROOT = Path("/data/shared/ubuntu/workspace/deepdive_kai0/xvla/data/mixed_hard")

# (src_name, target_task_index)
# kai → task_index 0, vis → task_index 1 (matches merged tasks.jsonl below)
SOURCES = [
    ("kai0_base",     0),
    ("kai0_dagger",   0),
    ("vis_v2_merged", 1),
]
DST = Path("/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/self_built/xvla_exp1_hard_merged")

KAI_PROMPT = "kai Flatten and fold the cloth."
VIS_PROMPT = "vis Flatten and fold the cloth."


def _rewrite_parquet(src_path: Path, dst_path: Path, new_ep_idx: int, global_idx_offset: int, target_task_idx: int):
    """Write a new parquet with corrected episode_index, index, task_index columns."""
    t = pq.read_table(src_path)
    n = t.num_rows
    new_cols = {}
    for col in t.column_names:
        if col == "episode_index":
            tt = t.schema.field("episode_index").type
            new_cols[col] = pa.array([new_ep_idx] * n, type=tt)
        elif col == "index":
            tt = t.schema.field("index").type
            new_cols[col] = pa.array([global_idx_offset + i for i in range(n)], type=tt)
        elif col == "task_index":
            tt = t.schema.field("task_index").type
            new_cols[col] = pa.array([target_task_idx] * n, type=tt)
        else:
            new_cols[col] = t[col]
    t2 = pa.table(new_cols)
    pq.write_table(t2, dst_path)
    return n  # frame count


def main():
    if DST.exists():
        sys.exit(f"dst exists: {DST} (delete first)")
    DST.mkdir(parents=True)
    (DST / "data" / "chunk-000").mkdir(parents=True)
    (DST / "meta").mkdir()
    for cam in CAMERAS:
        (DST / "videos" / "chunk-000" / cam).mkdir(parents=True)

    out_eps = []
    out_stats = []
    new_idx = 0
    global_frame_offset = 0
    info_ref = None
    for src_name, target_task_index in SOURCES:
        sd = SRC_ROOT / src_name
        if not sd.is_dir():
            sys.exit(f"missing source: {sd}")
        info = json.load((sd / "meta/info.json").open())
        info_ref = info_ref or info
        chunks_size = info.get("chunks_size", 1000)
        ep_lines = [json.loads(l) for l in (sd / "meta/episodes.jsonl").open()]
        ep_by_idx = {e["episode_index"]: e for e in ep_lines}
        stats_lines = [json.loads(l) for l in (sd / "meta/episodes_stats.jsonl").open()] if (sd / "meta/episodes_stats.jsonl").is_file() else []
        stats_by_idx = {s["episode_index"]: s for s in stats_lines}

        print(f"  {src_name}: {len(ep_by_idx)} episodes (target task_index={target_task_index})")
        for old_idx in sorted(ep_by_idx.keys()):
            old_chunk = old_idx // chunks_size
            sp = sd / f"data/chunk-{old_chunk:03d}/episode_{old_idx:06d}.parquet"
            sp_resolved = sp.resolve()
            if not sp_resolved.is_file():
                print(f"  skip missing parquet: {sp}", file=sys.stderr)
                continue
            dp = DST / f"data/chunk-000/episode_{new_idx:06d}.parquet"
            n_frames = _rewrite_parquet(sp_resolved, dp, new_idx, global_frame_offset, target_task_index)
            # Videos: symlink all 3
            for cam in CAMERAS:
                sv = sd / f"videos/chunk-{old_chunk:03d}/{cam}/episode_{old_idx:06d}.mp4"
                sv_resolved = sv.resolve()
                if not sv_resolved.is_file():
                    print(f"  skip missing video: {sv}", file=sys.stderr)
                    continue
                dv = DST / f"videos/chunk-000/{cam}/episode_{new_idx:06d}.mp4"
                os.symlink(sv_resolved, dv)
            # Episode meta
            ep = ep_by_idx[old_idx].copy()
            ep["episode_index"] = new_idx
            ep["_src_name"] = src_name
            ep["_src_idx"] = old_idx
            ep["_target_task_index"] = target_task_index
            out_eps.append(ep)
            # Stats
            if old_idx in stats_by_idx:
                st = stats_by_idx[old_idx].copy()
                st["episode_index"] = new_idx
                out_stats.append(st)
            global_frame_offset += n_frames
            new_idx += 1
            if new_idx % 500 == 0:
                print(f"    ... {new_idx} ep / {global_frame_offset} frames")
    print(f"\nmerged: {new_idx} episodes, {global_frame_offset} frames total (all parquets rewritten)")

    # Write meta
    with (DST / "meta/episodes.jsonl").open("w") as f:
        for ep in out_eps:
            f.write(json.dumps(ep) + "\n")
    if out_stats:
        with (DST / "meta/episodes_stats.jsonl").open("w") as f:
            for st in out_stats:
                f.write(json.dumps(st) + "\n")
    # tasks.jsonl with 2 entries (kai → 0, vis → 1)
    with (DST / "meta/tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": KAI_PROMPT}) + "\n")
        f.write(json.dumps({"task_index": 1, "task": VIS_PROMPT}) + "\n")
    # info.json
    info = json.loads(json.dumps(info_ref))
    n = len(out_eps)
    info["total_episodes"] = n
    info["total_frames"] = global_frame_offset
    info["total_videos"] = n * len(CAMERAS)
    info["total_chunks"] = 1
    info["chunks_size"] = max(8000, n + 100)  # ensure all eps fall in chunk-000
    info["total_tasks"] = 2
    info["splits"] = {"train": f"0:{n}"}
    if "depth_path" in info:
        del info["depth_path"]
    if "features" in info:
        info["features"] = {k: v for k, v in info["features"].items() if "depth" not in k.lower()}
    (DST / "meta/info.json").write_text(json.dumps(info, indent=2))
    print(f"done → {DST}")


if __name__ == "__main__":
    main()
