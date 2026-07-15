#!/usr/bin/env python
"""把 CRAVE 折线(polyline 去阶梯)逐帧 value 标签写成 KAI0 pi0-AE 训练数据集.

标签来自 dump_polyline_labels_kai_full.py → temp/crave_ae_labels/polyline/ep*.npy
(native 30Hz, 0→1, 双锚 Viterbi 去阶梯). 底座 = kai0_base(3055ep, 干净),
symlink meta/+videos/, data/ parquet 逐 ep 加 stage_progress_gt 列(逐帧对齐).

输出:
  crave_stage_poly       (raw polyline, 含真实回落)   ← 主实验
  crave_stage_poly_mono  (cummax 单调版, --mono)      ← 对照

对齐 crave_stage_{A,B}(同底座/同列/同 norm_stats)以控变量.
Run: /home/tim/miniconda3/envs/srpo/bin/python lmvla/crave/experiments/write_crave_stage_poly.py [--mono]
"""
from __future__ import annotations
import os, sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(os.environ.get("REPO", "/vePFS/tim/workspace/deepdive_kai0"))
SRC = REPO / "kai0/data/Task_A/kai0_base"
OUT = REPO / "kai0/data/Task_A/self_built"
MONO = "--mono" in sys.argv
LAB = REPO / ("lmvla/crave/temp/crave_ae_labels/polyline_mono" if MONO else "lmvla/crave/temp/crave_ae_labels/polyline")
DSNAME = "crave_stage_poly_mono" if MONO else "crave_stage_poly"
CSQ = 1000
_STD = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]


def link_shared(dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("meta", "videos"):
        dd = dst / sub
        if dd.is_symlink() or dd.exists():
            continue
        os.symlink((SRC / sub).resolve(), dd)


def main():
    dst = OUT / DSNAME
    link_shared(dst)
    (dst / "data").mkdir(parents=True, exist_ok=True)
    parts = sorted((SRC / "data").glob("chunk-*/episode_*.parquet"))
    n_ok = n_interp = n_fb = 0
    for p in parts:
        e = int(p.stem.split("_")[1])
        df = pd.read_parquet(p)
        df = df[[c for c in _STD if c in df.columns]]   # 只留标准列(防 HF CastError)
        lab_f = LAB / f"ep{e}.npy"
        if lab_f.exists():
            v = np.clip(np.load(lab_f).astype(np.float32), 0.0, 1.0)   # 双锚已 0→1, 只安全裁剪, 不 norm01
            if len(v) == len(df):
                df["stage_progress_gt"] = v; n_ok += 1
            else:
                xi = np.linspace(0, 1, len(df)); xa = np.linspace(0, 1, len(v))
                df["stage_progress_gt"] = np.interp(xi, xa, v).astype(np.float32); n_interp += 1
        else:
            df["stage_progress_gt"] = np.linspace(0, 1, len(df)).astype(np.float32); n_fb += 1
        outp = dst / "data" / p.parent.name / p.name
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(outp)
    # norm_stats: copy 自同底座的 crave_stage_A(state/action 分布一致); 无则从 kai0_base 取
    for cand in (OUT / "crave_stage_A" / "norm_stats.json", SRC / "norm_stats.json"):
        if cand.exists():
            shutil.copy(cand, dst / "norm_stats.json"); break
    print(f"[{DSNAME}] wrote {len(parts)} eps | poly={n_ok} interp={n_interp} fallback={n_fb} -> {dst}", flush=True)
    print(f"  norm_stats: {'copied' if (dst/'norm_stats.json').exists() else 'MISSING (需手动 compute_norm_states_fast)'}", flush=True)


if __name__ == "__main__":
    main()
