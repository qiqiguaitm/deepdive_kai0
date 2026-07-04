#!/usr/bin/env python3
"""Prune episodes with broken (unrecoverable) video symlinks from A_v4_base_dagger, renumber contiguous.

2026-07-04: cnbj 磁盘清理删了 11 个 episode 的源视频 (v4→源 symlink 断链, TOS 也无 404 → 彻底丢失).
lerobot LeRobotDataset.__init__ 断言要求所有 video file 存在 → warm14k --resume 崩. 排除这 11ep 重建为 2006ep.
Runs on gf3 (mounts cnbj /vePFS-North-E) with North-E kai0 venv. Writes A_v4_base_dagger_pruned then caller swaps.
"""
import json, os, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

ROOT = Path("/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/data/Task_A/self_built")
SRC = ROOT / "A_v4_base_dagger"
DST = ROOT / "A_v4_base_dagger_pruned"
BROKEN = {560, 601, 630, 632, 759, 765, 822, 1253, 1295, 1569, 1903}
CAMS = ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]
CHUNK = 0

eps = [json.loads(l) for l in (SRC / "meta" / "episodes.jsonl").read_text().splitlines()]
stats = {}
sp = SRC / "meta" / "episodes_stats.jsonl"
if sp.exists():
    for l in sp.read_text().splitlines():
        d = json.loads(l); stats[d["episode_index"]] = d
kept = [e for e in eps if e["episode_index"] not in BROKEN]
print(f"episodes {len(eps)} -> kept {len(kept)} (removed {len(eps)-len(kept)})", flush=True)
assert len(eps) - len(kept) == len(BROKEN), "removed count != broken set"

if DST.exists():
    shutil.rmtree(DST)
(DST / "data" / f"chunk-{CHUNK:03d}").mkdir(parents=True)
(DST / "meta").mkdir()

new_eps, new_stats = [], []
gidx = 0; total_frames = 0
for new_ep, e in enumerate(kept):
    old = e["episode_index"]
    df = pq.read_table(SRC / "data" / f"chunk-{CHUNK:03d}" / f"episode_{old:06d}.parquet").to_pandas()
    n = len(df)
    df["episode_index"] = np.int64(new_ep)
    if "index" in df.columns:
        df["index"] = np.arange(gidx, gidx + n, dtype=np.int64)
    gidx += n; total_frames += n
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   DST / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet")
    for cam in CAMS:
        sv = (SRC / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{old:06d}.mp4").resolve()
        if not sv.is_file():
            raise SystemExit(f"FATAL: kept ep {old} cam {cam} video不存在 {sv} (不该发生, 断链集应只11ep)")
        dv = DST / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{new_ep:06d}.mp4"
        dv.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(sv), dv)
    ne = dict(e); ne["episode_index"] = new_ep; new_eps.append(ne)
    if old in stats:
        st = dict(stats[old]); st["episode_index"] = new_ep; new_stats.append(st)

info = json.loads((SRC / "meta" / "info.json").read_text())
info["total_episodes"] = len(kept)
info["total_frames"] = total_frames
info["total_chunks"] = 1
info["chunks_size"] = len(kept)   # ==total_ep → 全落 chunk-000, 无缺 chunk
(DST / "meta" / "info.json").write_text(json.dumps(info, indent=2))
with (DST / "meta" / "episodes.jsonl").open("w") as f:
    for e in new_eps: f.write(json.dumps(e) + "\n")
if new_stats:
    with (DST / "meta" / "episodes_stats.jsonl").open("w") as f:
        for s in new_stats: f.write(json.dumps(s) + "\n")
shutil.copy(SRC / "meta" / "tasks.jsonl", DST / "meta" / "tasks.jsonl")
print(f"wrote {len(kept)} ep / {total_frames} frames -> {DST}", flush=True)

# norm_stats (reuse repo norm helper)
sys.path.insert(0, "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/train_scripts/kai/data")
from norm_stats_from_dataset import compute_norm_stats
print("computing norm_stats (action_dim=32)...", flush=True)
compute_norm_stats(str(DST), action_dim=32)
print("DONE", flush=True)
