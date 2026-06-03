"""把 LeRobot 数据集每集的 timestamp 规整到理想 fps 网格: timestamp = frame_index / fps。

为什么需要:
  visrobot01 的源数据保留了真实采集时间戳(~30fps 但有抖动 + 偶发大停顿,且 build_wam
  裁掉静止段后留下时间跳变),而 info.json 声明 fps=30。官方 lerobot 的
  check_timestamps_sync 要求集内相邻 Δt ≈ 1/fps(默认 tol=1e-4),vis 几乎每帧违规 →
  在构造 LeRobotDataset 时用 pprint 格式化百万级违规项而卡死;且 48 帧 action chunk
  按 [i/fps] 网格取帧也会因 timestamp 不在网格而失败。
  kairobot01 的 timestamp 本就是规整的 30fps 网格(0,1/30,2/30,...),无需处理。

把 timestamp 重写为 frame_index/fps 后:集内 Δt 恒为 1/fps → 检查通过、chunk 精确取帧。
这等价于"按固定 30fps 序列对待"(与 kai、与 info.json 声明一致),丢弃的只是采集抖动,
对固定帧率的 World-Action 模型本就该丢弃。frame_index 单调连续是前提(已校验)。

用法:
  python -m scripts.wam_pipeline.regularize_timestamps <dataset_root> [--dry-run]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="数据集根目录(含 meta/ data/)")
    ap.add_argument("--dry-run", action="store_true", help="只检查不写回")
    args = ap.parse_args()

    root = Path(args.root)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    data_path_tmpl = info["data_path"]
    chunks_size = int(info.get("chunks_size", 1000))

    episodes = []
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line)["episode_index"])
    episodes.sort()
    print(f"[regularize] {root.name}: {len(episodes)} episodes, fps={fps}, dry_run={args.dry_run}")

    n_changed = 0
    max_shift = 0.0
    for ep in tqdm(episodes, desc=f"regularize {root.name}"):
        chunk = ep // chunks_size
        pqf = root / data_path_tmpl.format(episode_chunk=chunk, episode_index=ep)
        table = pq.read_table(pqf)

        fi = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
        # frame_index 必须是 0..N-1 连续(否则规整会错位)
        if not np.array_equal(fi, np.arange(len(fi))):
            raise ValueError(f"episode {ep}: frame_index 非 0..N-1 连续,拒绝规整: head={fi[:5]}")

        ts_field = table.schema.field("timestamp")
        new_ts = (fi / fps).astype(np.float32 if ts_field.type == pa.float32() else np.float64)

        old_ts = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
        max_shift = max(max_shift, float(np.abs(old_ts - new_ts).max()))

        if not args.dry_run:
            idx = table.schema.get_field_index("timestamp")
            new_col = pa.array(new_ts, type=ts_field.type)
            table = table.set_column(idx, ts_field, new_col)
            pq.write_table(table, pqf)
            n_changed += 1

    print(f"[regularize] {'(dry-run) ' if args.dry_run else ''}rewrote {n_changed} files; "
          f"max |old_ts - new_ts| = {max_shift:.4f}s")


if __name__ == "__main__":
    main()
