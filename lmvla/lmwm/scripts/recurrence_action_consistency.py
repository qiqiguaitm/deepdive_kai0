#!/usr/bin/env python
"""V3a-proxy · 用途A(训练: r 加权 / 无标注 advantage)机制验证 —— 无需训练。
假设: 高 r(canonical, 跨demo共识)帧的跨-demo 动作应一致(BC well-posed);
      低 r(特异)帧动作发散(ill-posed)。→ 若 corr(r, 动作一致性)>0, 则"按 r 加权 BC"有据。
做法: 每帧取 k 个跨-ep 最近邻, 量其动作离散度 disp; 测 corr(r, -disp) + r分位 vs disp。
跨 kai0 / LIBERO(task0,6) / robotwin(72任务中取几个≥15ep)验证普适。
Run: OMP_NUM_THREADS=8 srpo python recurrence_action_consistency.py
Out: lmwm/docs/assets/recurrence_action_consistency.png
"""
import os, glob, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = "/vePFS/tim/workspace/deepdive_kai0"
LROOT = f"{REPO}/lmvla/lawam/dataset/libero_merged_no_noops_20hz"; LFEAT = f"{REPO}/lmvla/lmwm/data/libero_dinov3base"
KAI = f"{REPO}/lmvla/crave/data/kai_dinov3base"; KPAR = f"{REPO}/kai0/data/Task_A/kai0_base"
RROOT = f"{REPO}/lmvla/lawam/dataset/robotwin2.0"; RFEAT = f"{REPO}/lmvla/lmwm/data/robotwin_dinov3base"
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)

def recurrence_and_neighbors(F, ep, k=8):
    ne = len(set(ep.tolist())); uep = np.array(sorted(set(ep.tolist()))); M = len(F)
    D = cdist(F, F)
    # r: 软核跨-ep 密度
    dmin = np.full((M, ne), 1e9, np.float32)
    for j, e in enumerate(uep): dmin[:, j] = D[:, ep == e].min(1)
    other = ep[:, None] != uep[None]; sig = np.median(dmin[other])
    K = np.exp(-dmin**2/(2*sig*sig)); K[~other] = 0; r = K.sum(1)/(ne-1)
    # k 个跨-ep 最近邻(排除本 ep)
    Dm = D.copy()
    same = ep[:, None] == ep[None]; Dm[same] = 1e9; np.fill_diagonal(Dm, 1e9)
    nn = np.argsort(Dm, 1)[:, :k]
    return r, nn

def task_metric(F, A, ep, k=8):
    F = l2(F.astype(np.float32))
    A = (A - A.mean(0)) / (A.std(0) + 1e-6)                # 动作 z-score
    r, nn = recurrence_and_neighbors(F, ep, k)
    disp = np.array([np.linalg.norm(A[i] - A[nn[i]], axis=1).mean() for i in range(len(F))])
    rho = spearmanr(r, -disp).correlation                 # >0 = 高r低离散(一致)
    # r 分位 vs 平均 disp
    q = np.clip((r.argsort().argsort() / len(r) * 10).astype(int), 0, 9)
    prof = np.array([disp[q == b].mean() if (q == b).any() else np.nan for b in range(10)])
    return rho, prof, disp.mean()

# ---------- loaders: 返回 F[M,768], A[M,da], ep[M] ----------
_LIB = {}
def _lib_table():
    if not _LIB:
        p = sorted(glob.glob(f"{LROOT}/data/**/*.parquet", recursive=True))[0]
        _LIB["df"] = pd.read_parquet(p, columns=["episode_index", "frame_index", "task_index", "action"])
    return _LIB["df"]

def load_libero(frag):
    tdf = pd.read_parquet(f"{LROOT}/meta/tasks.parquet"); n2i = {n: int(r["task_index"]) for n, r in tdf.iterrows()}
    tid = [v for n, v in n2i.items() if frag.lower() in n.lower()][0]
    df = _lib_table(); g = df[df.task_index == tid]
    eps = [e for e in sorted(g.episode_index.unique()) if os.path.exists(f"{LFEAT}/ep{e}.npz")][:40]
    Fs, As, es = [], [], []
    for e in eps:
        f = np.load(f"{LFEAT}/ep{e}.npz")["grid"].astype(np.float32).mean(1)
        a = np.stack(g[g.episode_index == e].sort_values("frame_index")["action"].to_numpy())
        n = min(len(f), len(a)); Fs.append(f[:n]); As.append(a[:n]); es.append(np.full(n, e))
    return np.concatenate(Fs), np.concatenate(As), np.concatenate(es)

def load_kai0(cap=20, stride=4):
    idx = np.load(f"{KAI}/index.npz"); E = idx["E"]; FR = idx["FR"]
    feat = np.zeros((len(E), 768), np.float16)
    for sh in sorted(glob.glob(f"{KAI}/shard_*.npz")):
        s = np.load(sh); g = s["gidx"]; v = s["valid"] if "valid" in s.files else np.ones(len(g), bool); feat[g[v]] = s["feat"][v]
    eps = sorted(np.unique(E).tolist())[:cap]; Fs, As, es = [], [], []
    for e in eps:
        m = np.where(E == e)[0]; o = m[np.argsort(FR[m])][::stride]; fr = FR[m][np.argsort(FR[m])][::stride]
        a = np.stack(pd.read_parquet(f"{KPAR}/data/chunk-{e//1000:03d}/episode_{e:06d}.parquet", columns=["action"])["action"].to_numpy())
        fr = np.minimum(fr, len(a)-1); Fs.append(feat[o].astype(np.float32)); As.append(a[fr]); es.append(np.full(len(o), e))
    return np.concatenate(Fs), np.concatenate(As), np.concatenate(es)

def robotwin_tasks(nwant=3, minep=15):
    cached = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(f"{RFEAT}/ep*.npz"))
    from collections import defaultdict
    t2e = defaultdict(list)
    for e in cached:
        fs = glob.glob(f"{RROOT}/data/chunk-*/episode_{e:06d}.parquet")
        if fs:
            import pyarrow.parquet as pq; t2e[int(pq.read_table(fs[0], columns=["task_index"]).column("task_index")[0].as_py())].append(e)
    big = sorted([(t, es) for t, es in t2e.items() if len(es) >= minep], key=lambda x: -len(x[1]))[:nwant]
    return big

def load_robotwin(eps):
    Fs, As, es = [], [], []
    for e in eps:
        f = np.load(f"{RFEAT}/ep{e}.npz")["pooled"].astype(np.float32)
        pp = glob.glob(f"{RROOT}/data/chunk-*/episode_{e:06d}.parquet")[0]
        a = np.stack(pd.read_parquet(pp, columns=["action"])["action"].to_numpy())
        n = min(len(f), len(a)); Fs.append(f[:n]); As.append(a[:n]); es.append(np.full(n, e))
    return np.concatenate(Fs), np.concatenate(As), np.concatenate(es)

def main():
    t0 = time.time()
    tasks = [("kai0", *load_kai0()),
             ("LIBERO-task0", *load_libero("alphabet soup and the tomato sauce")),
             ("LIBERO-task6", *load_libero("white mug on the plate and put the chocolate"))]
    for i, (tid, eps) in enumerate(robotwin_tasks(3)):
        tasks.append((f"robotwin-t{tid}({len(eps)}ep)", *load_robotwin(eps)))
    print(f"{len(tasks)} tasks loaded ({time.time()-t0:.0f}s)", flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5)); rows = []
    for name, F, A, ep in tasks:
        rho, prof, md = task_metric(F, A, ep)
        rows.append((name, rho)); print(f"  [{name}] N={len(F)} da={A.shape[1]} | corr(r,-action_disp)={rho:+.3f} | disp {prof[0]:.2f}(low r)->{prof[-1]:.2f}(high r)", flush=True)
        ax[1].plot(np.arange(10), prof, marker="o", ms=3, label=f"{name} (ρ={rho:+.2f})")
    names = [r[0] for r in rows]; rhos = [r[1] for r in rows]
    ax[0].barh(range(len(names)), rhos, color=["#22c55e" if x > 0 else "#ef4444" for x in rhos])
    ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(names, fontsize=8); ax[0].axvline(0, color="k", lw=.8)
    ax[0].set_xlabel("Spearman corr(r, -action_dispersion)"); ax[0].set_title(f"Use-A premise: r vs action consistency\nmedian ρ={np.median(rhos):+.3f}, >0 on {np.mean(np.array(rhos)>0)*100:.0f}% tasks", fontsize=10); ax[0].grid(alpha=.2)
    ax[1].set_xlabel("recurrence r decile (low->high)"); ax[1].set_ylabel("mean cross-demo action dispersion")
    ax[1].set_title("high r -> lower action dispersion (BC target more consistent)", fontsize=10); ax[1].legend(fontsize=6.5); ax[1].grid(alpha=.2)
    fig.suptitle("V3a-proxy · recurrence as label-free BC-reliability / advantage weight (no training)", fontsize=12)
    fig.tight_layout()
    out = f"{REPO}/lmvla/lmwm/docs/assets/recurrence_action_consistency.png"; fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"\n[SUMMARY] median ρ={np.median(rhos):+.3f} | >0 on {np.mean(np.array(rhos)>0)*100:.0f}% tasks\nSAVED {out} ({time.time()-t0:.0f}s)\nACTCONS_DONE", flush=True)

if __name__ == "__main__":
    main()
