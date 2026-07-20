#!/usr/bin/env python
"""把 robotwin2.0(LeRobot v2.1)的 LMWM 子集(pairs 覆盖的 1315 个积木族 ep)转成 LeRobot **v3.0** 数据集。

为什么必须转: starVLA 的 gr00t dataloader 是 v3.0 专用(datasets.py 明确 raise
"LeRobot v3.0 requires _load_episodes_hf()"), 需要 meta/episodes/*/*.parquet + meta/tasks.parquet
+ meta/modality.json; 而 robotwin2.0 是 v2.1(episodes.jsonl/tasks.jsonl, 无 modality.json)。

三个关键设计(都为消除风险/成本):
  ① **保留原始 episode_index**(672~4949, 不重新编号)
     → LMWM provider 用 (episode_index, frame_index) 查 pairs, 保留原号则 **pairs.npz 无需重映射**,
       彻底避免"重编号后 provider 全部 valid=False、训练照跑但 LMWM 实际没生效"的静默失效。
  ② **每 episode 独占一个 data/video 文件**(chunk/file 逐 episode 分配, from_timestamp=0)
     → v3.0 允许逐 episode 指定 chunk/file, 于是**不需要拼接视频**(省 1315×3 次重编码)。
  ③ **data parquet 与 video 全部用软链**(两边 parquet 列完全一致, 无需转换)→ 省 ~80GB 拷贝。

stats: meta/stats_gr00t.json 缺失时 dataloader rank0 会自动计算并缓存, 故不在此生成。

用法: python build_robotwin_v30_subset.py [--out <dir>] [--limit N]
"""
import os, json, argparse, glob
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "/vePFS/tim/workspace/deepdive_kai0"
SRC = f"{REPO}/lmvla/lawam/dataset/robotwin2.0"
PAIRS = f"{REPO}/lmvla/lmwm/data/robotwin_milestone/pairs.npz"
OUT_DEFAULT = f"{REPO}/lmvla/lawam/dataset/robotwin2_lmwm_v30"
CAMS = ["observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"]
CHUNK_SIZE = 1000


def src_data_parquet(ep):
    hits = glob.glob(f"{SRC}/data/chunk-*/episode_{ep:06d}.parquet")
    return hits[0] if hits else None


def src_video(ep, cam):
    hits = glob.glob(f"{SRC}/videos/chunk-*/{cam}/episode_{ep:06d}.mp4")
    return hits[0] if hits else None


def relsym(target, linkpath):
    """建相对软链(数据集整体可搬)。"""
    os.makedirs(os.path.dirname(linkpath), exist_ok=True)
    if os.path.islink(linkpath) or os.path.exists(linkpath):
        os.remove(linkpath)
    os.symlink(os.path.relpath(target, os.path.dirname(linkpath)), linkpath)


def main():
    global SRC, PAIRS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--src", default=SRC, help="robotwin2.0(v2.1) 源目录; 跨集群时路径不同")
    ap.add_argument("--pairs", default=PAIRS, help="milestone pairs.npz(只用来选 episode 子集)")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只取前 N 个 ep(smoke)")
    a = ap.parse_args()
    OUT = a.out
    SRC, PAIRS = a.src, a.pairs   # src_data_parquet/src_video 按全局名取用

    P = np.load(PAIRS)
    eps = sorted(set(int(x) for x in np.unique(P["cur_ep"])))
    if a.limit:
        eps = eps[:a.limit]
    print(f"[eps] {len(eps)} 个 episode (原始 index {eps[0]}~{eps[-1]}, 保留不重编号)", flush=True)

    src_info = json.load(open(f"{SRC}/meta/info.json"))
    fps = int(src_info["fps"])

    # 原始 tasks.jsonl: task_index -> task string
    task_of = {}
    with open(f"{SRC}/meta/tasks.jsonl") as f:
        for line in f:
            d = json.loads(line); task_of[int(d["task_index"])] = d["task"]

    rows, cum, used_tasks = [], 0, set()
    for i, ep in enumerate(eps):
        sp = src_data_parquet(ep)
        if sp is None:
            print(f"  ! ep{ep} 无 data parquet, 跳过", flush=True); continue
        t = pq.read_table(sp)
        n = t.num_rows
        ti = int(t["task_index"][0].as_py()); used_tasks.add(ti)
        c, fi = i // CHUNK_SIZE, i % CHUNK_SIZE

        relsym(sp, f"{OUT}/data/chunk-{c:03d}/file-{fi:03d}.parquet")
        row = {
            "episode_index": ep,                      # ★ 原始号
            "data/chunk_index": c, "data/file_index": fi,
            "dataset_from_index": cum, "dataset_to_index": cum + n,
            "tasks": [task_of.get(ti, "")],
            "length": n,
            "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
        }
        for cam in CAMS:
            sv = src_video(ep, cam)
            if sv is None:
                print(f"  ! ep{ep} 缺视频 {cam}", flush=True)
            else:
                relsym(sv, f"{OUT}/videos/{cam}/chunk-{c:03d}/file-{fi:03d}.mp4")
            row[f"videos/{cam}/chunk_index"] = c
            row[f"videos/{cam}/file_index"] = fi
            row[f"videos/{cam}/from_timestamp"] = 0.0     # 每 ep 独占文件 → 偏移恒为 0
            row[f"videos/{cam}/to_timestamp"] = n / fps
        rows.append(row); cum += n
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(eps)} eps, 累计 {cum} 帧", flush=True)

    # ---- meta/episodes ----
    os.makedirs(f"{OUT}/meta/episodes/chunk-000", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), f"{OUT}/meta/episodes/chunk-000/file-000.parquet")

    # ---- meta/tasks.parquet ----
    tl = sorted(used_tasks)
    pq.write_table(
        pa.Table.from_pydict({"task_index": pa.array(tl, pa.int64()),
                              "task": pa.array([task_of.get(t, "") for t in tl], pa.string())}),
        f"{OUT}/meta/tasks.parquet")

    # ---- meta/info.json (v3.0) ----
    info = dict(src_info)
    info.update({
        "codebase_version": "v3.0",
        "total_episodes": len(rows),
        "total_frames": cum,
        "total_tasks": len(tl),
        "chunks_size": CHUNK_SIZE,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "splits": {"train": f"0:{len(rows)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    })
    for k in ("total_videos", "total_chunks"):
        info.pop(k, None)
    json.dump(info, open(f"{OUT}/meta/info.json", "w"), indent=4)

    # ---- meta/modality.json ----
    # 键名必须与 AgilexDataConfig(注册名 "robotwin_joint")的 state_keys/action_keys 逐字一致:
    #   left_joints(6) + left_gripper(1) + right_joints(6) + right_gripper(1) = aloha 14 维
    modality = {
        "state": {
            "left_joints":   {"start": 0,  "end": 6,  "absolute": True},
            "left_gripper":  {"start": 6,  "end": 7,  "absolute": True},
            "right_joints":  {"start": 7,  "end": 13, "absolute": True},
            "right_gripper": {"start": 13, "end": 14, "absolute": True},
        },
        "action": {
            "left_joints":   {"start": 0,  "end": 6},
            "left_gripper":  {"start": 6,  "end": 7},
            "right_joints":  {"start": 7,  "end": 13},
            "right_gripper": {"start": 13, "end": 14},
        },
        "video": {
            "cam_high":        {"original_key": "observation.images.cam_high"},
            "cam_left_wrist":  {"original_key": "observation.images.cam_left_wrist"},
            "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    json.dump(modality, open(f"{OUT}/meta/modality.json", "w"), indent=2)

    print(f"[done] {OUT}: {len(rows)} eps, {cum} 帧, {len(tl)} tasks", flush=True)
    print("BUILD_V30_DONE", flush=True)


if __name__ == "__main__":
    main()
