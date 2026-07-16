#!/usr/bin/env python
"""诊断视觉别名对 recurrence 场的影响 + 加 proprio 值不值(纯离线只读)。
别名 = 图像空间近、真实任务态远。逐帧取图像 kNN(跨-ep), 量近邻:
  time_spread = 近邻归一化时间 std(图像说"同"、时间说"不同" → 别名)
主判据: image-only vs image⊕proprio(能量1:1) 的 time_spread —— 加 state 能否分开别名帧。
Run: OMP_NUM_THREADS=8 srpo python recurrence_aliasing_diag.py
Out: lmwm/docs/assets/recurrence_aliasing_diag.png
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
RFEAT = f"{REPO}/lmvla/lmwm/data/robotwin_dinov3base"; RROOT = f"{REPO}/lmvla/lawam/dataset/robotwin2.0"
def l2(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-9)
def zs(x): return (x-x.mean(0))/(x.std(0)+1e-6)

def recurrence(F, ep):
    ne = len(set(ep.tolist())); uep = np.array(sorted(set(ep.tolist()))); M = len(F)
    D = cdist(F, F); dmin = np.full((M, ne), 1e9, np.float32)
    for j, e in enumerate(uep): dmin[:, j] = D[:, ep == e].min(1)
    other = ep[:, None] != uep[None]; sig = np.median(dmin[other])
    return (np.exp(-dmin**2/(2*sig*sig))*other).sum(1)/(ne-1), D

def knn_tspread(Fq, t, ep, D=None, k=8):
    Dm = (cdist(Fq, Fq) if D is None else D.copy())
    Dm[ep[:, None] == ep[None]] = 1e9; np.fill_diagonal(Dm, 1e9)
    nn = np.argsort(Dm, 1)[:, :k]
    return t[nn].std(1)                                # 每帧近邻时间 std

def analyze(F, S, t, ep):
    F = l2(F.astype(np.float32)); Sz = l2(zs(S.astype(np.float32)))
    r, D = recurrence(F, ep)
    ts_img = knn_tspread(F, t, ep, D)                  # image-only
    ts_joint = knn_tspread(np.concatenate([F, Sz], 1), t, ep)   # image⊕proprio 能量1:1
    # r 局部粗糙度(逐 ep 沿时间的 |Δr|)vs 别名
    rough = np.zeros(len(F))
    for e in sorted(set(ep.tolist())):
        idx = np.where(ep == e)[0]; o = idx[np.argsort(t[idx])]
        d = np.abs(np.diff(r[o], prepend=r[o][0])); rough[o] = d
    return dict(ts_img=ts_img, ts_joint=ts_joint, t=t, r=r, rough=rough)

# ---------- loaders: 返回 F[M,768], S[M,ds], t[M], ep[M] ----------
_LIB = {}
def _lib():
    if not _LIB: _LIB["df"] = pd.read_parquet(sorted(glob.glob(f"{LROOT}/data/**/*.parquet", recursive=True))[0], columns=["episode_index","frame_index","task_index","observation.state"])
    return _LIB["df"]
def load_libero(frag):
    tdf = pd.read_parquet(f"{LROOT}/meta/tasks.parquet"); n2i = {n: int(r["task_index"]) for n, r in tdf.iterrows()}
    tid = [v for n, v in n2i.items() if frag.lower() in n.lower()][0]; g = _lib(); g = g[g.task_index == tid]
    eps = [e for e in sorted(g.episode_index.unique()) if os.path.exists(f"{LFEAT}/ep{e}.npz")][:40]
    Fs,Ss,ts,es = [],[],[],[]
    for e in eps:
        f = np.load(f"{LFEAT}/ep{e}.npz")["grid"].astype(np.float32).mean(1)
        s = np.stack(g[g.episode_index==e].sort_values("frame_index")["observation.state"].to_numpy())
        n = min(len(f), len(s)); Fs.append(f[:n]); Ss.append(s[:n]); ts.append(np.linspace(0,1,n)); es.append(np.full(n,e))
    return np.concatenate(Fs), np.concatenate(Ss), np.concatenate(ts), np.concatenate(es)
def load_kai0(cap=20, stride=4):
    idx = np.load(f"{KAI}/index.npz"); E=idx["E"]; FR=idx["FR"]; feat=np.zeros((len(E),768),np.float16)
    for sh in sorted(glob.glob(f"{KAI}/shard_*.npz")):
        s=np.load(sh); gg=s["gidx"]; v=s["valid"] if "valid" in s.files else np.ones(len(gg),bool); feat[gg[v]]=s["feat"][v]
    eps=sorted(np.unique(E).tolist())[:cap]; Fs,Ss,ts,es=[],[],[],[]
    for e in eps:
        m=np.where(E==e)[0]; o=m[np.argsort(FR[m])][::stride]; fr=FR[m][np.argsort(FR[m])][::stride]
        st=np.stack(pd.read_parquet(f"{KPAR}/data/chunk-{e//1000:03d}/episode_{e:06d}.parquet",columns=["observation.state"])["observation.state"].to_numpy())
        fr=np.minimum(fr,len(st)-1); Fs.append(feat[o].astype(np.float32)); Ss.append(st[fr]); ts.append(np.linspace(0,1,len(o))); es.append(np.full(len(o),e))
    return np.concatenate(Fs),np.concatenate(Ss),np.concatenate(ts),np.concatenate(es)
def robotwin_big(n=2, minep=15):
    import pyarrow.parquet as pq; from collections import defaultdict
    cached=sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(f"{RFEAT}/ep*.npz")); t2e=defaultdict(list)
    for e in cached:
        fs=glob.glob(f"{RROOT}/data/chunk-*/episode_{e:06d}.parquet")
        if fs: t2e[int(pq.read_table(fs[0],columns=["task_index"]).column("task_index")[0].as_py())].append(e)
    return sorted([(t,es) for t,es in t2e.items() if len(es)>=minep], key=lambda x:-len(x[1]))[:n]
def load_robotwin(eps):
    Fs,Ss,ts,es=[],[],[],[]
    for e in eps:
        f=np.load(f"{RFEAT}/ep{e}.npz")["pooled"].astype(np.float32)
        s=np.stack(pd.read_parquet(glob.glob(f"{RROOT}/data/chunk-*/episode_{e:06d}.parquet")[0],columns=["observation.state"])["observation.state"].to_numpy())
        n=min(len(f),len(s)); Fs.append(f[:n]); Ss.append(s[:n]); ts.append(np.linspace(0,1,n)); es.append(np.full(n,e))
    return np.concatenate(Fs),np.concatenate(Ss),np.concatenate(ts),np.concatenate(es)

def main():
    t0=time.time()
    tasks=[("kai0",*load_kai0()),
           ("LIBERO-task0(clear)",*load_libero("alphabet soup and the tomato sauce")),
           ("LIBERO-task6(diffuse)",*load_libero("white mug on the plate and put the chocolate"))]
    for tid,eps in robotwin_big(2): tasks.append((f"robotwin-t{tid}",*load_robotwin(eps)))
    print(f"{len(tasks)} tasks loaded ({time.time()-t0:.0f}s)", flush=True)
    fig,ax=plt.subplots(1,3,figsize=(16,5)); rows=[]; HI=0.15
    for i,(name,F,S,t,ep) in enumerate(tasks):
        d=analyze(F,S,t,ep)
        mi,mj=d["ts_img"].mean(),d["ts_joint"].mean(); prev=np.mean(d["ts_img"]>HI)
        rho=spearmanr(d["ts_img"],d["rough"]).correlation
        rows.append((name,mi,mj,prev,rho)); print(f"  [{name}] N={len(F)} ds={S.shape[1]} | time_spread img={mi:.3f} joint={mj:.3f} ({(mi-mj)/mi*100:+.0f}%) | 别名率(>{HI})={prev:.0%} | corr(alias,r_rough)={rho:+.2f}", flush=True)
        # severity vs time (30 bins, image-only)
        tb=np.clip((d["t"]*30).astype(int),0,29); prof=np.array([d["ts_img"][tb==b].mean() if (tb==b).any() else np.nan for b in range(30)])
        ax[1].plot(np.linspace(0,1,30),prof,marker=".",ms=3,label=name)
    names=[r[0] for r in rows]; x=np.arange(len(names))
    ax[0].bar(x-0.2,[r[1] for r in rows],0.4,label="image-only",color="#7c3aed")
    ax[0].bar(x+0.2,[r[2] for r in rows],0.4,label="image⊕proprio",color="#22c55e")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names,rotation=25,fontsize=7,ha="right"); ax[0].set_ylabel("mean neighbor time_spread (lower=less aliasing)")
    ax[0].set_title("KEY: does +proprio reduce aliasing?\n(green<purple = proprio separates aliased frames)",fontsize=10); ax[0].legend(); ax[0].grid(alpha=.2)
    ax[1].axhline(HI,color="r",ls="--",lw=.8,alpha=.5); ax[1].set_xlabel("normalized time"); ax[1].set_ylabel("time_spread (image-only)")
    ax[1].set_title("Where aliasing concentrates (expect task6 tail high)",fontsize=10); ax[1].legend(fontsize=6.5); ax[1].grid(alpha=.2)
    ax[2].bar(x,[r[3] for r in rows],color="#d98b00"); ax[2].set_xticks(x); ax[2].set_xticklabels(names,rotation=25,fontsize=7,ha="right")
    ax[2].set_ylabel(f"aliasing prevalence (%frames time_spread>{HI})"); ax[2].set_title("Aliasing prevalence per task",fontsize=10); ax[2].grid(alpha=.2)
    fig.suptitle("Visual-aliasing diagnosis for recurrence field r(o) — image-only vs image⊕proprio",fontsize=12); fig.tight_layout()
    out=f"{REPO}/lmvla/lmwm/docs/assets/recurrence_aliasing_diag.png"; fig.savefig(out,dpi=115,bbox_inches="tight")
    mi=np.array([r[1] for r in rows]); mj=np.array([r[2] for r in rows])
    print(f"\n[SUMMARY] +proprio 降 time_spread 中位 {np.median((mi-mj)/mi)*100:+.0f}% | 别名率中位 {np.median([r[3] for r in rows]):.0%}\nSAVED {out} ({time.time()-t0:.0f}s)\nALIAS_DONE",flush=True)

if __name__=="__main__":
    main()
