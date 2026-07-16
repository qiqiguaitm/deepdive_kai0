#!/usr/bin/env python
"""V6-T3 自适应视界目标: 取代 r-脊固定目标 + 取代旧「r-门控」。
病根: r-脊是全局语义目标, 在冻结 pooled-DINOv3 空间做全局定位, 别名任务(task8 双moka壶)塌缩→坏hint。
优雅解: 单一目标, 按「检索时间一致性 c(o)」自适应视界——离线烘进 target, 架构零改, 推理不变。
  c(o) = clip(1 - time_spread(o)/T_ref, 0, 1);  time_spread = k=8 跨-ep 近邻归一化时间的 std(=别名度)
  idx_target = round(idx_local + c·(idx_ridge - idx_local)),  idx_local=min(t+7,L-1), idx_ridge=下一段r脊
  c=1(清晰)→够到 r-脊(task6红利); c=0(别名)→收缩回 t+7(baseline抗别名安全区)。target 始终是真帧 latent(on-manifold)。
输出与 p1_libero_rvalley_pairs 同格式 → 下游(train_lmwm / build_target_compact)仅换 PAIRS 路径。
用法: srpo python p1_libero_adaptive_pairs.py   → lmwm/data/libero_adaptive/pairs.npz
"""
import os, glob
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
FEAT = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_dinov3base"
OUT = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_adaptive"
THR = 0.03
LAG = 7            # 局部视界(t+7, 同 baseline)
KNN = 8           # 检索一致性近邻数
TREF_PCTL = 90    # T_ref = 每任务 time_spread 的分位(尺度无关, 同 r median-heuristic 哲学)
def l2(x): return x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)

def r_seg_coh(gd):
    """返回 每 ep: (段边界 seg, 段脊 ridge(局部idx), n, c[n] 检索时间一致性)。"""
    eps = list(gd); F = l2(np.concatenate([gd[e] for e in eps]).astype(np.float32))
    ep = np.concatenate([np.full(len(gd[e]), i) for i, e in enumerate(eps)]); ne = len(eps)
    lens = [len(gd[e]) for e in eps]; offs = np.cumsum([0]+lens)
    tnorm = np.concatenate([np.linspace(0, 1, L) if L > 1 else np.zeros(1) for L in lens])  # 归一化时间
    D = cdist(F, F); dmin = np.full((len(F), ne), 1e9, np.float32)
    for j in range(ne): dmin[:, j] = D[:, ep == j].min(1)
    other = ep[:, None] != np.arange(ne)[None]; sig = np.median(dmin[other])
    r = (np.exp(-dmin**2/(2*sig*sig))*other).sum(1)/(ne-1)
    # 检索时间一致性: 跨-ep k 近邻的归一化时间 std(小=同相位可信, 大=别名不可信)
    Dm = D.copy(); Dm[ep[:, None] == ep[None]] = 1e9; np.fill_diagonal(Dm, 1e9)
    k = min(KNN, len(F)-1)
    nn = np.argpartition(Dm, k, axis=1)[:, :k]
    ts = tnorm[nn].std(1)                                    # time_spread per frame
    Tref = np.percentile(ts, TREF_PCTL) + 1e-9
    cvec = np.clip(1.0 - ts/Tref, 0.0, 1.0)                  # coherence per frame ∈[0,1]
    res = {}
    for i, e in enumerate(eps):
        s, en = offs[i], offs[i+1]; n = en-s; rr = r[s:en]
        v, _ = find_peaks(-gaussian_filter1d(rr, 1.4), prominence=THR, distance=max(2, n//12))
        seg = [0]+list(v)+[n]; ridge = [a+int(np.argmax(rr[a:b])) for a, b in zip(seg[:-1], seg[1:])]
        res[e] = (seg, ridge, n, cvec[s:en])
    return res

def main():
    files = sorted(glob.glob(f"{FEAT}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4]))
    dpar = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    ep2task = pd.read_parquet(dpar[0], columns=["episode_index", "task_index"]).groupby("episode_index")["task_index"].first().to_dict()
    ep_gist = {}
    for f in files:
        e = int(os.path.basename(f)[2:-4]); ep_gist[e] = np.load(f)["grid"].astype(np.float32).mean(1)
    from collections import defaultdict
    task_eps = defaultdict(list)
    for e in ep_gist: task_eps[ep2task.get(e, -1)].append(e)
    print(f"[tasks] {len(task_eps)} 任务", flush=True)
    cur_ep, cur_fi, tgt_fi, cur_ms, pair_task = [], [], [], [], []
    nseg = []; c_all = []; reach_frac = []
    for tk, teps in sorted(task_eps.items()):
        if len(teps) < 5: continue
        res = r_seg_coh({e: ep_gist[e] for e in teps})
        for e in teps:
            seg, ridge, n, cc = res[e]; nseg.append(len(ridge))
            for p in range(n):
                si = np.searchsorted(seg, p, "right")-1
                idx_ridge = ridge[si+1] if si+1 < len(ridge) else n-1     # 下一段 canonical 脊(末段→末帧)
                idx_local = min(p+LAG, n-1)                                # 局部视界 t+7
                c = float(cc[p])
                idx_t = int(round(idx_local + c*(idx_ridge - idx_local)))  # 自适应视界
                idx_t = int(np.clip(idx_t, min(p+1, n-1), n-1))            # 必须未来, clip
                cur_ep.append(e); cur_fi.append(p); tgt_fi.append(idx_t); cur_ms.append(si); pair_task.append(tk)
                c_all.append(c)
                if idx_ridge != idx_local: reach_frac.append((idx_t-idx_local)/(idx_ridge-idx_local))
    cur_ep = np.array(cur_ep); cur_fi = np.array(cur_fi); tgt_fi = np.array(tgt_fi); cur_ms = np.array(cur_ms); pair_task = np.array(pair_task)
    tot_frames = sum(len(v) for v in ep_gist.values())
    print(f"[seg] 每ep段数 中位={int(np.median(nseg))} 范围[{min(nseg)},{max(nseg)}]", flush=True)
    print(f"[coh] c 中位={np.median(c_all):.2f} | c<0.3(别名)占={np.mean(np.array(c_all)<0.3)*100:.0f}% | c>0.7(清晰)占={np.mean(np.array(c_all)>0.7)*100:.0f}%", flush=True)
    print(f"[reach] 目标视界向脊靠拢比例 中位={np.median(reach_frac):.2f} (1=纯r脊/M'', 0=纯t+7/baseline)", flush=True)
    print(f"[pairs] {len(cur_ep)} 对 / {tot_frames} 帧 = 覆盖 {len(cur_ep)/tot_frames*100:.0f}%", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/pairs.npz", cur_ep=cur_ep, cur_fi=cur_fi, tgt_fi=tgt_fi, cur_ms=cur_ms, pair_task=pair_task)
    print(f"[save] {OUT}/pairs.npz\nDONE", flush=True)

if __name__ == "__main__":
    main()
