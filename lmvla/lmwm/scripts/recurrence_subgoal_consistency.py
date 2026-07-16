#!/usr/bin/env python
"""V3b-proxy · 用途B(子目标条件)机制验证 —— 无需训练。
假设: r-低谷定义的【自适应子目标】(前方最近 canonical checkpoint)比【固定时延 t+τ】子目标
      在跨-demo 上更一致 → 更可学的 world-model 目标(正对 -4.2pt 里 milestone+1 vs t+7)。
做法: 每帧取 k 跨-ep 最近邻(同状态他demo); 比其【自适应子目标特征】vs【固定子目标特征】的离散度。
公平: 固定 τ = 每任务自适应前瞻的中位, 两者平均看得一样远, 只比"自适应 vs 固定"。
跨 kai0 / LIBERO / robotwin 验证。Out: assets/recurrence_subgoal_consistency.png
"""
import os, glob, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = "/vePFS/tim/workspace/deepdive_kai0"
LROOT = f"{REPO}/lmvla/lawam/dataset/libero_merged_no_noops_20hz"; LFEAT = f"{REPO}/lmvla/lmwm/data/libero_dinov3base"
KAI = f"{REPO}/lmvla/crave/data/kai_dinov3base"; RFEAT = f"{REPO}/lmvla/lmwm/data/robotwin_dinov3base"; RROOT = f"{REPO}/lmvla/lawam/dataset/robotwin2.0"
THR = 0.03
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)

def analyze(gd, k=8):
    eps = list(gd); F = l2(np.concatenate([gd[e] for e in eps]).astype(np.float32))
    ep = np.concatenate([np.full(len(gd[e]), i) for i, e in enumerate(eps)]); ne = len(eps)
    lens = [len(gd[e]) for e in eps]; offs = np.cumsum([0]+lens); M = len(F)
    D = cdist(F, F)
    # r 场
    dmin = np.full((M, ne), 1e9, np.float32)
    for j in range(ne): dmin[:, j] = D[:, ep == j].min(1)
    other = ep[:, None] != np.arange(ne)[None]; sig = np.median(dmin[other])
    r = (np.exp(-dmin**2/(2*sig*sig))*other).sum(1)/(ne-1)
    # 三种子目标: 下一段 r-脊(canonical 收敛点) / 下一 r-谷边界 / 固定 t+τ
    ridge = np.zeros(M, int); valley = np.zeros(M, int); look = np.zeros(M, int)
    for i in range(ne):
        s, e = offs[i], offs[i+1]; n = e-s; rr = r[s:e]
        v, _ = find_peaks(-gaussian_filter1d(rr, 1.4), prominence=THR, distance=max(2, n//12))
        seg = [0] + list(v) + [n]                          # 段边界
        rdg = [a + int(np.argmax(rr[a:b])) for a, b in zip(seg[:-1], seg[1:])]  # 每段脊
        bnds = list(v) + [n-1]
        for p in range(n):
            si = np.searchsorted(seg, p, "right") - 1
            ridge[s+p] = s + (rdg[si+1] if si+1 < len(rdg) else n-1); look[s+p] = ridge[s+p]-(s+p)
            valley[s+p] = s + next((b for b in bnds if b > p), n-1)
    tau = max(3, int(np.median(look[look > 0])) if (look > 0).any() else 5)
    fixed = np.array([min((oi := offs[np.searchsorted(offs, i, "right")-1]) + (i-oi) + tau,
                          offs[np.searchsorted(offs, i, "right")]-1) for i in range(M)])
    Dm = D.copy(); Dm[ep[:, None] == ep[None]] = 1e9; np.fill_diagonal(Dm, 1e9)
    nn = np.argsort(Dm, 1)[:, :k]
    def disp(SG): return float(np.mean([1 - (SG[i]*SG[nn[i]]).sum(1).mean() for i in range(M)]))
    return disp(F[ridge]), disp(F[valley]), disp(F[fixed]), tau

# ---- loaders (feature only) ----
def load_libero(frag):
    tdf = pd.read_parquet(f"{LROOT}/meta/tasks.parquet"); n2i = {n: int(r["task_index"]) for n, r in tdf.iterrows()}
    tid = [v for n, v in n2i.items() if frag.lower() in n.lower()][0]
    p = sorted(glob.glob(f"{LROOT}/data/**/*.parquet", recursive=True))[0]
    meta = pd.read_parquet(p, columns=["episode_index", "task_index"])
    eps = [e for e in sorted(meta[meta.task_index == tid].episode_index.unique()) if os.path.exists(f"{LFEAT}/ep{e}.npz")][:40]
    return {e: np.load(f"{LFEAT}/ep{e}.npz")["grid"].astype(np.float32).mean(1) for e in eps}
def load_kai0(cap=20, stride=4):
    idx = np.load(f"{KAI}/index.npz"); E = idx["E"]; FR = idx["FR"]
    feat = np.zeros((len(E), 768), np.float16)
    for sh in sorted(glob.glob(f"{KAI}/shard_*.npz")):
        s = np.load(sh); g = s["gidx"]; v = s["valid"] if "valid" in s.files else np.ones(len(g), bool); feat[g[v]] = s["feat"][v]
    eps = sorted(np.unique(E).tolist())[:cap]; gd = {}
    for e in eps:
        m = np.where(E == e)[0]; o = m[np.argsort(FR[m])][::stride]; gd[e] = feat[o].astype(np.float32)
    return gd
def robotwin_big(nwant=3, minep=15):
    import pyarrow.parquet as pq
    from collections import defaultdict
    cached = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(f"{RFEAT}/ep*.npz"))
    t2e = defaultdict(list)
    for e in cached:
        fs = glob.glob(f"{RROOT}/data/chunk-*/episode_{e:06d}.parquet")
        if fs: t2e[int(pq.read_table(fs[0], columns=["task_index"]).column("task_index")[0].as_py())].append(e)
    return sorted([(t, es) for t, es in t2e.items() if len(es) >= minep], key=lambda x: -len(x[1]))[:nwant]
def load_robotwin(eps):
    return {e: np.load(f"{RFEAT}/ep{e}.npz")["pooled"].astype(np.float32) for e in eps}

def main():
    t0 = time.time()
    tasks = [("kai0", load_kai0()),
             ("LIBERO-task0", load_libero("alphabet soup and the tomato sauce")),
             ("LIBERO-task6", load_libero("white mug on the plate and put the chocolate"))]
    for tid, eps in robotwin_big(3): tasks.append((f"robotwin-t{tid}", load_robotwin(eps)))
    print(f"{len(tasks)} tasks ({time.time()-t0:.0f}s)", flush=True)
    rows = []
    for name, gd in tasks:
        dr, dv, df, tau = analyze(gd)
        rows.append((name, dr, dv, df, tau)); print(f"  [{name}] tau={tau} | ridge={dr:.4f} valley={dv:.4f} fixed={df:.4f} | ridge vs fixed {'WINS' if dr<df else 'loses'} {(df-dr)/df*100:+.0f}%", flush=True)
    names = [r[0] for r in rows]; winr = np.mean([r[1] < r[3] for r in rows])*100
    print(f"\n[SUMMARY] ridge subgoal 更一致 than fixed on {winr:.0f}% tasks | mean {np.mean([(r[3]-r[1])/r[3] for r in rows])*100:+.0f}% | valley on {np.mean([r[2]<r[3] for r in rows])*100:.0f}%", flush=True)
    fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(len(names))
    ax.bar(x-0.25, [r[1] for r in rows], 0.25, label="adaptive RIDGE (next canonical)", color="#22c55e")
    ax.bar(x, [r[2] for r in rows], 0.25, label="adaptive VALLEY (next boundary)", color="#ef4444")
    ax.bar(x+0.25, [r[3] for r in rows], 0.25, label="fixed lag (t+tau)", color="#d98b00")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, fontsize=8, ha="right"); ax.set_ylabel("cross-demo subgoal dispersion (lower=better target)")
    ax.set_title(f"V3b · which is a more consistent world-model target?\nRIDGE beats fixed on {winr:.0f}% tasks (lower=better)", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); out = f"{REPO}/lmvla/lmwm/docs/assets/recurrence_subgoal_consistency.png"; fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"SAVED {out} ({time.time()-t0:.0f}s)\nSUBGOAL_DONE", flush=True)

if __name__ == "__main__":
    main()
