"""按 held-out manifest 把 visrobot01 物理切成两个自洽数据集(硬链接,零拷贝)。

为什么物理切而非 episodes= 子集:
  lerobot 的 _get_query_indices 用**原始** episode_index 去索引 episode_data_index,而 episodes=
  子集会把该表缩到子集长度 → IndexError(index 2067 vs size 1898)。delta_frames 必走该路径,
  故 episodes= 与 delta_frames 不兼容。物理切分后每个集从 0 连续重编号,episode_data_index 自洽,
  delta_frames 正常,且训练/评估都不再需要 episodes=。

产出(默认在 wam_fold_v1/ 下):
  visrobot01_train (1898, 重编号 0..1897)   visrobot01_val (200, 重编号 0..199)
各含 data/ videos/(硬链接)、t5_embedding/(硬链接)、meta/{info,episodes,episodes_stats,tasks}.jsonl,
以及 split_map.json(new_idx->原始 idx + manifest sha,供溯源/复现)。

用法:
  python -m scripts.wam_pipeline.split_heldout \
    --src ../kai0/data/wam_fold_v1/visrobot01 \
    --manifest assets_visrobot01/heldout_visrobot01.json --out_base ../kai0/data/wam_fold_v1
"""
import argparse
import glob
import json
import os

import pyarrow.parquet as pq

from scripts.build_wam_dataset import CHUNK_SIZE, build_info, place_video, rewrite_parquet

VIDEO_KEYS = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]


def _src_paths(src, idx):
    chunk = idx // CHUNK_SIZE
    pqf = os.path.join(src, "data", f"chunk-{chunk:03d}", f"episode_{idx:06d}.parquet")
    vids = {k: os.path.join(src, "videos", f"chunk-{chunk:03d}", k, f"episode_{idx:06d}.mp4") for k in VIDEO_KEYS}
    t5 = os.path.join(src, "t5_embedding", f"episode_{idx:06d}.pt")
    return pqf, vids, t5


def build_split(src, out_root, ids, vid_mode="link"):
    os.makedirs(os.path.join(out_root, "meta"), exist_ok=True)
    # 读源 episodes_stats(按原始 idx),供重编号后写回
    src_stats = {}
    es_src = os.path.join(src, "meta", "episodes_stats.jsonl")
    if os.path.exists(es_src):
        for l in open(es_src):
            if l.strip():
                r = json.loads(l); src_stats[int(r["episode_index"])] = r
    eps_meta, stats_meta, split_map = [], [], {}
    gframe = 0
    for new_idx, src_idx in enumerate(sorted(ids)):
        pqf, vids, t5 = _src_paths(src, src_idx)
        new_chunk = new_idx // CHUNK_SIZE
        t, n = rewrite_parquet(pqf, new_idx, gframe)
        od = os.path.join(out_root, "data", f"chunk-{new_chunk:03d}"); os.makedirs(od, exist_ok=True)
        pq.write_table(t, os.path.join(od, f"episode_{new_idx:06d}.parquet"))
        for k in VIDEO_KEYS:
            dst = os.path.join(out_root, "videos", f"chunk-{new_chunk:03d}", k, f"episode_{new_idx:06d}.mp4")
            place_video(vids[k], dst, vid_mode)
        if os.path.isfile(t5):
            dt = os.path.join(out_root, "t5_embedding"); os.makedirs(dt, exist_ok=True)
            ddst = os.path.join(dt, f"episode_{new_idx:06d}.pt")
            if os.path.exists(ddst):
                os.remove(ddst)
            os.link(t5, ddst)
        eps_meta.append({"episode_index": new_idx, "tasks": ["Flatten and fold the cloth."], "length": n,
                         "t5_embedding_path": f"t5_embedding/episode_{new_idx:06d}.pt"})
        if src_idx in src_stats:
            r = dict(src_stats[src_idx]); r["episode_index"] = new_idx; stats_meta.append(r)
        split_map[new_idx] = src_idx
        gframe += n
    with open(os.path.join(out_root, "meta", "info.json"), "w") as f:
        json.dump(build_info(len(ids), gframe), f, indent=4)
    with open(os.path.join(out_root, "meta", "episodes.jsonl"), "w") as f:
        for e in eps_meta:
            f.write(json.dumps(e) + "\n")
    if stats_meta:
        with open(os.path.join(out_root, "meta", "episodes_stats.jsonl"), "w") as f:
            for r in stats_meta:
                f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_root, "meta", "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "Flatten and fold the cloth."}) + "\n")
    with open(os.path.join(out_root, "split_map.json"), "w") as f:
        json.dump({"new_to_src": split_map}, f)
    print(f"  -> {out_root}: {len(ids)} episodes, {gframe} frames")
    return len(ids), gframe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--copy", action="store_true", help="拷贝视频(默认硬链接)")
    args = ap.parse_args()
    man = json.load(open(args.manifest))
    tr, ho = man["train_episode_indices"], man["heldout_episode_indices"]
    assert set(tr) & set(ho) == set(), "train/heldout overlap!"
    mode = "copy" if args.copy else "link"
    print(f"split {args.src}: train={len(tr)} val={len(ho)} (sha {man['episodes_jsonl_sha256'][:12]})")
    build_split(args.src, os.path.join(args.out_base, "visrobot01_train"), tr, mode)
    build_split(args.src, os.path.join(args.out_base, "visrobot01_val"), ho, mode)
    print("SPLIT_DONE")


if __name__ == "__main__":
    main()
