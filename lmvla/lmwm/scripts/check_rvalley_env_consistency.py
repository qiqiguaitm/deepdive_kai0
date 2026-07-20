#!/usr/bin/env python
"""核查 rvalley 分段/建对是否跨环境一致(§6 待办:scipy find_peaks/gaussian_filter1d 版本差异)。

r 场 = cdist(numpy) + gaussian_filter1d/find_peaks(scipy)。两环境 scipy 版本不同则分段可能漂移。
在同一批 DINOv3 特征上跑 r_and_segments, 落 (每 ep 段数, 段边界, 段脊) 指纹, 两环境比对。

用法:
  <python> check_rvalley_env_consistency.py --feat <dir> --ntask 4 --out /tmp/rv_env_<tag>.json
  python check_rvalley_env_consistency.py --compare /tmp/rv_env_a.json /tmp/rv_env_b.json
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_dinov3base")
    ap.add_argument("--ntask", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.03)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()

    if a.compare:
        x, y = (json.load(open(p)) for p in a.compare)
        import scipy
        ndiff = sum(1 for k in x if k != "_env" and x[k] != y.get(k))
        print(f"envs: {x['_env']}  vs  {y['_env']}")
        print(f"共 {len(x)-1} ep, 指纹不一致的 ep 数 = {ndiff}")
        if ndiff:
            for k in list(x)[:200]:
                if k != "_env" and x[k] != y.get(k):
                    print(f"  ep{k}: {x[k]}  vs  {y.get(k)}")
        print("判定:", "✅ rvalley 分段跨环境一致" if ndiff == 0 else "⚠️ 分段漂移 —— 两环境产物不可混用")
        return

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import p1_libero_rvalley_pairs as R
    R.THR = a.thr
    import scipy
    import transformers

    files = sorted(glob.glob(f"{a.feat}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4]))
    import pandas as pd
    dpar = sorted(glob.glob(f"{R.ROOT}/data/**/*.parquet", recursive=True))
    ep2task = pd.read_parquet(dpar[0], columns=["episode_index", "task_index"]) \
        .groupby("episode_index")["task_index"].first().to_dict()
    from collections import defaultdict
    task_eps = defaultdict(list)
    for f in files:
        e = int(os.path.basename(f)[2:-4]); task_eps[ep2task.get(e, -1)].append(e)
    tasks = sorted(t for t in task_eps if len(task_eps[t]) >= 5)[:a.ntask]

    fp = {"_env": f"tf{transformers.__version__}/scipy{scipy.__version__}"}
    for tk in tasks:
        teps = task_eps[tk]
        ep_gist = {e: np.load(f"{a.feat}/ep{e}.npz")["grid"].astype(np.float32).mean(1) for e in teps}
        res = R.r_and_segments(ep_gist)
        for e in teps:
            seg, ridge, n = res[e]
            fp[str(e)] = [int(n), [int(x) for x in seg], [int(x) for x in ridge]]
    print(f"[{fp['_env']}] {len(fp)-1} ep 指纹", flush=True)
    if a.out:
        json.dump(fp, open(a.out, "w"))
        print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
