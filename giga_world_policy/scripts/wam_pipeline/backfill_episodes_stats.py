"""为缺失 meta/episodes_stats.jsonl 的 LeRobot v2.1 数据集补建 per-episode 统计。

背景: scripts/build_wam_dataset.py 合并数据时只写了 info.json / episodes.jsonl /
tasks.jsonl,漏写了 v2.1 必需的 meta/episodes_stats.jsonl。导致官方 lerobot 的
LeRobotDatasetMetadata.load_metadata() 在 v2.1 分支找不到该文件 → 回退 HF Hub →
HF_HUB_OFFLINE 报 OfflineModeIsEnabled,整条 norm_stats / train 管线在 len(dataset)
处即崩。

本脚本只为数值列(默认 observation.state + action)计算 stats,**跳过 3 路视频**:
  - 视频列不在 parquet 里(在 videos/),逐集解码 2098+6512 集代价巨大;
  - 本项目图像走 Wan VAE / 255 归一化,训练/推理都不消费数据集自带的视频 stats;
  - giga 的动作归一化走独立的 norm_stats_delta.json(compute_norm_stats 重算)。
若日后某环节确需视频 stats,可加 --include-video 另行扩展。

用法:
  python -m scripts.wam_pipeline.backfill_episodes_stats <dataset_root>
  python -m scripts.wam_pipeline.backfill_episodes_stats <dataset_root> --keys observation.state action
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.utils import EPISODES_STATS_PATH, write_episode_stats
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="数据集根目录(含 meta/ data/)")
    ap.add_argument(
        "--keys",
        nargs="+",
        default=["observation.state", "action"],
        help="要计算 stats 的数值列(默认 state+action,跳过视频)",
    )
    args = ap.parse_args()

    root = Path(args.root)
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    features = info["features"]
    data_path_tmpl = info["data_path"]
    chunks_size = int(info.get("chunks_size", 1000))

    out = root / EPISODES_STATS_PATH  # meta/episodes_stats.jsonl
    if out.exists():
        # write_episode_stats 是追加写,重跑前必须清掉旧文件,否则 episode 重复
        print(f"[backfill] 删除已存在的 {out}(避免追加重复)")
        out.unlink()

    # 读 episodes.jsonl 拿到全部 episode_index(权威顺序)
    episodes = []
    with open(meta / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line)["episode_index"])
    episodes.sort()
    print(f"[backfill] {root.name}: {len(episodes)} episodes, keys={args.keys}")

    missing_keys = [k for k in args.keys if k not in features]
    if missing_keys:
        raise KeyError(f"info.features 缺少列 {missing_keys}(可用: {list(features)})")

    for ep in tqdm(episodes, desc=f"backfill {root.name}"):
        chunk = ep // chunks_size
        pqf = root / data_path_tmpl.format(episode_chunk=chunk, episode_index=ep)
        if not pqf.is_file():
            raise FileNotFoundError(f"episode {ep} parquet 缺失: {pqf}")
        table = pq.read_table(pqf, columns=args.keys)
        cols = table.to_pydict()

        episode_data = {}
        for k in args.keys:
            shape = tuple(features[k].get("shape", []))
            arr = np.asarray([np.asarray(v) for v in cols[k]])
            # (N, *feature_shape);标量列 shape=() -> (N,) 交给 compute_episode_stats 处理
            if shape:
                arr = arr.reshape(len(cols[k]), *shape)
            episode_data[k] = arr

        ep_stats = compute_episode_stats(episode_data, {k: features[k] for k in args.keys})
        write_episode_stats(ep, ep_stats, root)

    print(f"[backfill] 完成 -> {out}")


if __name__ == "__main__":
    main()
