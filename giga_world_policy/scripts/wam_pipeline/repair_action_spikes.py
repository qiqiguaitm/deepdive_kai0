"""修复 LeRobot 数据集 observation.state / action 中的单帧传感器尖刺。

kai 源数据存在复发性编码器 glitch(如 1895.83 / -292.66 / 33070.78 等离谱值,
出现在夹爪/关节维的孤立单帧),会严重污染 norm_stats(某维 std 被单点拉到 10+)。
piper 关节角合理范围 ~[-3.2, 3.2] rad、夹爪 ~[0, 0.1],故 |value|>阈值(默认 10)
即判为 garbage,用同维时间最近的有效帧线性插值替换(边界用最近有效值)。

state 与 action 同帧同维通常镜像损坏,两列都修。

用法:
  python -m scripts.wam_pipeline.repair_action_spikes <dataset_root> [--abs-threshold 10] [--dry-run]
也可作为库函数 repair_matrix(M, thr) 被 build_wam_dataset 调用做产出即净化。
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def repair_matrix(M: np.ndarray, thr: float):
    """就地修复 (N, D) 矩阵中 |.|>thr 的点(同维线性插值)。返回修复点数。"""
    n_fixed = 0
    N = len(M)
    for d in range(M.shape[1]):
        col = M[:, d]
        bad = np.abs(col) > thr
        if not bad.any():
            continue
        good_idx = np.where(~bad)[0]
        if len(good_idx) == 0:
            continue  # 整列都坏,跳过(异常,留给上游处理)
        for fr in np.where(bad)[0]:
            # 最近的左右有效邻居
            left = good_idx[good_idx < fr]
            right = good_idx[good_idx > fr]
            if len(left) and len(right):
                l, r = left[-1], right[0]
                col[fr] = col[l] + (col[r] - col[l]) * (fr - l) / (r - l)
            elif len(left):
                col[fr] = col[left[-1]]
            else:
                col[fr] = col[right[0]]
            n_fixed += 1
    return n_fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--abs-threshold", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    info = json.loads((root / "meta" / "info.json").read_text())
    cs = int(info.get("chunks_size", 1000))
    tmpl = info["data_path"]
    eps = [json.loads(l)["episode_index"] for l in open(root / "meta" / "episodes.jsonl") if l.strip()]

    total_fixed = 0
    touched_eps = []
    for ep in eps:
        f = root / tmpl.format(episode_chunk=ep // cs, episode_index=ep)
        t = pq.read_table(f)
        changed = 0
        new_cols = {}
        for col in ("observation.state", "action"):
            if col not in t.column_names:
                continue
            field = t.schema.field(col)
            M = np.stack(t.column(col).to_pylist()).astype(np.float32)
            c = repair_matrix(M, args.abs_threshold)
            if c:
                changed += c
                new_cols[col] = pa.array(list(M), type=field.type)
        if changed:
            touched_eps.append((ep, changed))
            total_fixed += changed
            if not args.dry_run:
                for col, arr in new_cols.items():
                    t = t.set_column(t.schema.get_field_index(col), t.schema.field(col), arr)
                pq.write_table(t, f)

    print(f"[repair] {root.name}: {'(dry-run) ' if args.dry_run else ''}修复 {total_fixed} 个点 / "
          f"{len(touched_eps)} 个 episode(含 state+action 镜像)")
    for ep, c in touched_eps:
        print(f"   ep {ep}: {c} 点")


if __name__ == "__main__":
    main()
