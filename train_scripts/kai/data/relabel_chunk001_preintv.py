#!/usr/bin/env python3
"""relabel_chunk001_preintv.py — 在【已存在的旧 chunk-001】上 relabel-only 补 dagger_frame_class
{0=robot, 1=intv_core, 2=preintv}, 不动 video/meta (只加/改 parquet 一列)。

为什么 relabel-only 够 (不用重生成):
  inference 在第一个 freedrive(冻结)键关闭 (_do_takeover: execute=false → sleep(0.5) → close),
  尾 ~15 帧是 execute=false 后的滑行伪影。旧 stitch 脚本 find_keep_indices 已 drop 掉 inf 末 15 帧
  (= 那段滑行)。故旧 chunk-001 里每个 0→1(inf→dag) 边界之前留下的, 正是【模型接管前真实行为尾巴】。
  直接在其上标 preintv 即可, 无需重拼、无需重编码视频。

relabel 规则 (与 stitch_dagger_episodes.classify_segment 同口径):
  intervention==1                          → 1 (intv_core)
  每个 0→1 边界之前 PREINTV_MARGIN 帧的 intervention==0 帧 → 2 (preintv, 动/静不限, 含卡死冻结)
  其余 intervention==0                      → 0 (robot)

用法:
  python relabel_chunk001_preintv.py --date 2026-06-16 --dry-run   # 预览 class 分布
  python relabel_chunk001_preintv.py --date 2026-06-16             # 落盘
  python relabel_chunk001_preintv.py --all                        # 全部 v4 日期
  python relabel_chunk001_preintv.py --all --dry-run
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ROOT = "/data1/DATA_IMP/KAI0/Task_A/dagger/v4"
PREINTV_MARGIN = 22   # =round(0.75*30) Sirius 反应窗; 与 stitch_dagger_episodes.PREINTV_MARGIN 一致


def relabel_one(pq_path: str) -> tuple[np.ndarray, dict]:
    """→ (new_class_array, count_dict). 不写文件。"""
    df = pd.read_parquet(pq_path, columns=["intervention"])
    iv = df["intervention"].values.astype(int)
    n = len(iv)
    cls = np.zeros(n, dtype=np.int8)
    cls[iv == 1] = 1  # intv_core
    # 0→1 边界: 其前 PREINTV_MARGIN 帧的 intervention==0 帧 → preintv
    for i in range(1, n):
        if iv[i - 1] == 0 and iv[i] == 1:
            lo = max(0, i - PREINTV_MARGIN)
            seg = slice(lo, i)
            mask = iv[seg] == 0            # 只标真正的 inf 帧 (防上一段 dag 太近)
            idx = np.arange(lo, i)[mask]
            cls[idx] = 2
    cnt = {int(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True))}
    return cls, cnt


def process_date(date: str, dry_run: bool) -> dict:
    d = date if date.endswith("-v4") else f"{date}-v4"
    files = sorted(glob.glob(f"{ROOT}/{d}/data/chunk-001/episode_*.parquet"))
    if not files:
        print(f"[{d}] 无 chunk-001, 跳过")
        return {}
    agg = {0: 0, 1: 0, 2: 0}
    n_boundaries = 0
    for pq in files:
        cls, cnt = relabel_one(pq)
        for k, v in cnt.items():
            agg[k] = agg.get(k, 0) + v
        n_boundaries += int(np.sum((cls == 2)) > 0)
        if not dry_run:
            df = pd.read_parquet(pq)
            df["dagger_frame_class"] = cls
            df.to_parquet(pq, index=False)
    tot = sum(agg.values())
    tag = "DRY" if dry_run else "WROTE"
    print(f"[{d}] {tag}: {len(files)} eps, {tot} 帧 | "
          f"robot={agg.get(0,0)} ({100*agg.get(0,0)/max(1,tot):.0f}%) "
          f"intv={agg.get(1,0)} ({100*agg.get(1,0)/max(1,tot):.0f}%) "
          f"preintv={agg.get(2,0)} ({100*agg.get(2,0)/max(1,tot):.1f}%)")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.date and not args.all:
        ap.error("需要 --date 或 --all")
    dates = ([os.path.basename(x) for x in sorted(glob.glob(f"{ROOT}/2026-*-v4"))]
             if args.all else [args.date])
    grand = {0: 0, 1: 0, 2: 0}
    for date in dates:
        agg = process_date(date, args.dry_run)
        for k, v in agg.items():
            grand[k] = grand.get(k, 0) + v
    tot = sum(grand.values())
    if tot:
        print(f"\n{'='*60}\n合计: {tot} 帧 | robot {100*grand[0]/tot:.0f}% | "
              f"intv {100*grand[1]/tot:.0f}% | preintv {100*grand[2]/tot:.1f}%")


if __name__ == "__main__":
    main()
