"""构建 WAM 专用 LeRobot 数据集。

把 kai0 / vis 的多个来源合并、规范化为本项目(GigaWorld-Policy WAM)所需的干净 LeRobot 格式:
  - 剔除 depth(vis_dagger 仅在 info.json 声明、无实际数据)与 dagger 专用列(intervention)
  - 统一相机命名为 WAM 标准:top_head→cam_high, hand_left→cam_left_wrist, hand_right→cam_right_wrist
  - 统一视频目录命名为 feature key 全名(observation.images.cam_*)
  - 合并各来源为单一连续编号的 LeRobot 集,统一 task 文本与 meta schema
  - 按 embodiment 分别输出:visrobot01 / kairobot01(本体标识在训练 config 里用 embodiment= 指定)

视频默认硬链接(同文件系统、同用户,瞬时零拷贝);--copy 改为真实拷贝;--transcode-av1 把 av1 源转码为 h264(需 ffmpeg,仅在 WAM 无法解 av1 时用)。

用法:
  python -m scripts.build_wam_dataset \
      --data_root /home/tim/workspace/deepdive_kai0/kai0/data/Task_A \
      --out_base  /home/tim/workspace/deepdive_kai0/kai0/data/wam_fold_v1 \
      [--limit N] [--copy] [--transcode-av1] [--only visrobot01]
"""
import argparse
import glob
import json
import os
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.utils import serialize_dict

from scripts.wam_pipeline.repair_action_spikes import repair_matrix

SPIKE_ABS_THRESHOLD = 10.0  # |state/action 值|>此阈值判为传感器 glitch(关节~3.2/夹爪~0.1)

CAM_SRC_TO_DST = {
    "top_head": "observation.images.cam_high",
    "hand_left": "observation.images.cam_left_wrist",
    "hand_right": "observation.images.cam_right_wrist",
}
KEEP_COLS = [
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]
TASK_TEXT = "Flatten and fold the cloth."
CHUNK_SIZE = 1000
HEIGHT, WIDTH = 480, 640
FPS = 30


def embodiment_sources(data_root):
    """embodiment -> 有序的源数据集根目录列表。"""
    vis = sorted(glob.glob(os.path.join(data_root, "vis_base", "v2", "*"))) + sorted(
        glob.glob(os.path.join(data_root, "vis_dagger", "*"))
    )
    kai = [os.path.join(data_root, "kai0_base"), os.path.join(data_root, "kai0_dagger")]
    return {
        "visrobot01": [p for p in vis if os.path.isdir(os.path.join(p, "meta"))],
        "kairobot01": [p for p in kai if os.path.isdir(os.path.join(p, "meta"))],
    }


def list_episodes(src_root):
    """返回 [(src_idx, parquet_path)],按 src_idx 升序。"""
    files = glob.glob(os.path.join(src_root, "data", "chunk-*", "episode_*.parquet"))
    out = []
    for f in files:
        base = os.path.basename(f)
        idx = int(base[len("episode_") : -len(".parquet")])
        out.append((idx, f))
    return sorted(out)


def src_video_path(src_root, cam_short, src_idx):
    """兼容 kai(observation.images.<short>)与 vis(<short>)两种视频目录命名。"""
    src_chunk = src_idx // CHUNK_SIZE
    for name in (f"observation.images.{cam_short}", cam_short):
        p = os.path.join(src_root, "videos", f"chunk-{src_chunk:03d}", name, f"episode_{src_idx:06d}.mp4")
        if os.path.isfile(p):
            return p
    return None


def place_video(src_mp4, dst_mp4, mode):
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
    if os.path.exists(dst_mp4):
        os.remove(dst_mp4)
    if mode == "transcode":
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src_mp4, "-c:v", "libx264", "-pix_fmt", "yuv420p", dst_mp4],
            check=True,
        )
    elif mode == "copy":
        import shutil

        shutil.copy2(src_mp4, dst_mp4)
    else:  # link
        os.link(src_mp4, dst_mp4)


def is_av1(src_mp4):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name", "-of", "default=nw=1:nk=1", src_mp4],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() == "av1"
    except Exception:
        return False


def rewrite_parquet(pq_path, new_idx, global_frame):
    t = pq.read_table(pq_path)
    cols = [c for c in KEEP_COLS if c in t.column_names]
    t = t.select(cols)
    n = t.num_rows
    t = t.set_column(t.column_names.index("episode_index"), "episode_index", pa.array([new_idx] * n, pa.int64()))
    t = t.set_column(t.column_names.index("index"), "index",
                     pa.array(list(range(global_frame, global_frame + n)), pa.int64()))
    if "task_index" in t.column_names:
        t = t.set_column(t.column_names.index("task_index"), "task_index", pa.array([0] * n, pa.int64()))
    # 规整 frame_index=0..n-1、timestamp=frame_index/FPS:源(尤其 vis)保留真实采集时间戳
    # (~30fps 抖动 + 裁段留下的大停顿),会让 lerobot check_timestamps_sync 集内违规、用
    # pprint 格式化百万项卡死,且 48 帧 action-chunk 按 1/fps 网格取帧失败。统一为固定帧率
    # 网格(与 kairobot01、info.json 声明的 fps 一致),丢弃的只是采集抖动。
    if "frame_index" in t.column_names:
        t = t.set_column(t.column_names.index("frame_index"), "frame_index",
                         pa.array(list(range(n)), pa.int64()))
    if "timestamp" in t.column_names:
        t = t.set_column(t.column_names.index("timestamp"), "timestamp",
                         pa.array([i / FPS for i in range(n)], pa.float32()))
    # 净化单帧传感器尖刺(kai 源有复发性编码器 glitch,如 1895.83/-292.66,孤立单帧),
    # 用同维时间近邻插值替换,否则会把某维 norm_stats 的 std 拉到 10+。
    for col in ("observation.state", "action"):
        if col in t.column_names:
            field = t.schema.field(col)
            M = np.stack(t.column(col).to_pylist()).astype(np.float32)
            if repair_matrix(M, SPIKE_ABS_THRESHOLD):
                t = t.set_column(t.schema.get_field_index(col), field, pa.array(list(M), type=field.type))
    return t, n


def episode_stats_from_table(t):
    """从合并后的 parquet 表计算 per-episode 统计(仅 state/action,跳过视频)。

    v2.1 必需 meta/episodes_stats.jsonl;只为数值列算 stats —— 视频列不在 parquet 内,
    逐集解码代价大,且本项目图像走 Wan VAE/255 归一化、动作走独立 norm_stats_delta.json,
    都不消费数据集自带的视频 stats。
    """
    feats = {
        "observation.state": {"dtype": "float32", "shape": [14]},
        "action": {"dtype": "float32", "shape": [14]},
    }
    ep_data = {}
    for k in feats:
        arr = np.asarray([np.asarray(v) for v in t.column(k).to_pylist()], dtype=np.float32)
        ep_data[k] = arr.reshape(len(arr), *feats[k]["shape"])
    return compute_episode_stats(ep_data, feats)


def build_info(total_episodes, total_frames):
    def vid(key):
        return {
            "dtype": "video",
            "shape": [HEIGHT, WIDTH, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": HEIGHT, "video.width": WIDTH, "video.codec": "h264",
                "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                "video.fps": FPS, "video.channels": 3, "has_audio": False,
            },
        }
    features = {k: vid(k) for k in CAM_SRC_TO_DST.values()}
    features["observation.state"] = {"dtype": "float32", "shape": [14], "names": None}
    features["action"] = {"dtype": "float32", "shape": [14], "names": None}
    for c in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        features[c] = {"dtype": "int64" if c != "timestamp" else "float32", "shape": [1], "names": None}
    return {
        "codebase_version": "v2.1",
        "robot_type": "agilex",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * 3,
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def build_embodiment(emb, roots, out_base, limit, vid_mode, transcode_av1):
    out_root = os.path.join(out_base, emb)
    os.makedirs(os.path.join(out_root, "meta"), exist_ok=True)
    new_idx, global_frame = 0, 0
    episodes_meta = []
    episodes_stats_meta = []
    print(f"\n=== {emb}: {len(roots)} source datasets ===")
    for root in roots:
        eps = list_episodes(root)
        if limit:
            eps = eps[:limit]
        for src_idx, pq_path in eps:
            new_chunk = new_idx // CHUNK_SIZE
            # parquet
            t, n = rewrite_parquet(pq_path, new_idx, global_frame)
            out_pq_dir = os.path.join(out_root, "data", f"chunk-{new_chunk:03d}")
            os.makedirs(out_pq_dir, exist_ok=True)
            pq.write_table(t, os.path.join(out_pq_dir, f"episode_{new_idx:06d}.parquet"))
            episodes_stats_meta.append((new_idx, episode_stats_from_table(t)))
            # videos
            for cam_short, dst_key in CAM_SRC_TO_DST.items():
                src_mp4 = src_video_path(root, cam_short, src_idx)
                if src_mp4 is None:
                    raise FileNotFoundError(f"missing video {cam_short} for {root} ep {src_idx}")
                mode = vid_mode
                if vid_mode == "link" and transcode_av1 and is_av1(src_mp4):
                    mode = "transcode"
                dst_mp4 = os.path.join(out_root, "videos", f"chunk-{new_chunk:03d}", dst_key,
                                       f"episode_{new_idx:06d}.mp4")
                place_video(src_mp4, dst_mp4, mode)
            episodes_meta.append({"episode_index": new_idx, "tasks": [TASK_TEXT], "length": n})
            global_frame += n
            new_idx += 1
        print(f"  [{os.path.basename(root)}] -> running total {new_idx} eps, {global_frame} frames")
    # meta
    with open(os.path.join(out_root, "meta", "info.json"), "w") as f:
        json.dump(build_info(new_idx, global_frame), f, indent=4)
    with open(os.path.join(out_root, "meta", "episodes.jsonl"), "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(out_root, "meta", "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_TEXT}) + "\n")
    # v2.1 必需:per-episode 统计(格式同 lerobot write_episode_stats:{episode_index, stats})
    with open(os.path.join(out_root, "meta", "episodes_stats.jsonl"), "w") as f:
        for idx, st in episodes_stats_meta:
            f.write(json.dumps({"episode_index": idx, "stats": serialize_dict(st)}) + "\n")
    print(f"  >> {emb} DONE: {new_idx} episodes, {global_frame} frames -> {out_root}")
    return new_idx, global_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--limit", type=int, default=0, help="每个源子集只取前 N 条(0=全部),用于测试")
    ap.add_argument("--copy", action="store_true", help="真实拷贝视频(默认硬链接)")
    ap.add_argument("--transcode-av1", action="store_true", help="把 av1 源视频转码为 h264(需 ffmpeg)")
    ap.add_argument("--only", default=None, help="只构建指定 embodiment(visrobot01/kairobot01)")
    args = ap.parse_args()

    vid_mode = "copy" if args.copy else "link"
    sources = embodiment_sources(args.data_root)
    if args.only:
        sources = {args.only: sources[args.only]}
    summary = {}
    for emb, roots in sources.items():
        summary[emb] = build_embodiment(emb, roots, args.out_base, args.limit, vid_mode, args.transcode_av1)
    print("\n==== SUMMARY ====")
    for emb, (e, fr) in summary.items():
        print(f"  {emb}: {e} episodes, {fr} frames")


if __name__ == "__main__":
    main()
