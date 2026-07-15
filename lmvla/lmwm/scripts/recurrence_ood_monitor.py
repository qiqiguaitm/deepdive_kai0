#!/usr/bin/env python
"""V3c-proxy · 用途C(部署OOD监控): r 的流形判别力 —— 跨任务注入, 不需失败rollout。
对每个参考任务: 80%ep 建 demo 流形(带宽σ由其内部跨-ep dmin中位标定);
  in-task 查询 = 留出20%ep的帧(应高r);  off-task 查询 = 别的任务的帧(应低r)。
判据: AUROC(r_in 为正 / r_off 为负)。高AUROC = "低r=脱稿"的部署监控成立且普适。
Run: OMP_NUM_THREADS=8 srpo python recurrence_ood_monitor.py
Out: lmwm/docs/assets/recurrence_ood.png
"""
import os, glob, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = "/vePFS/tim/workspace/deepdive_kai0"
ROOT = f"{REPO}/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
FEAT = f"{REPO}/lmvla/lmwm/data/libero_dinov3base"
rng = np.random.RandomState(0)
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def bank_sigma(refeps):
    """参考流形内部标定带宽 σ = 跨-ep 最近邻距离中位。"""
    F = l2(np.concatenate(refeps).astype(np.float32))
    ep = np.concatenate([np.full(len(e), i) for i, e in enumerate(refeps)]); ne = len(refeps)
    D = cdist(F, F); dmin = np.full((len(F), ne), 1e9, np.float32)
    for j in range(ne): dmin[:, j] = D[:, np.where(ep == j)[0]].min(1)
    other = ep[:, None] != np.arange(ne)[None]
    return F, ep, ne, float(np.median(dmin[other]))

def r_query(F, ep, ne, sig, Q):
    """查询帧 Q 对参考流形的复现密度(mean over ref episodes)。"""
    D = cdist(l2(Q.astype(np.float32)), F); m = len(Q)
    dmin = np.full((m, ne), 1e9, np.float32)
    for j in range(ne): dmin[:, j] = D[:, np.where(ep == j)[0]].min(1)
    return np.exp(-dmin**2 / (2*sig*sig)).mean(1)

def load_tasks(nmax=12):
    dpar = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    meta = pd.concat([pd.read_parquet(p, columns=["episode_index", "task_index"]) for p in dpar])
    ep2task = meta.groupby("episode_index")["task_index"].first().to_dict()
    from collections import defaultdict
    te = defaultdict(list)
    for e, t in ep2task.items():
        if os.path.exists(f"{FEAT}/ep{e}.npz"): te[t].append(e)
    tasks = {}
    for t, eps in sorted(te.items()):
        if len(eps) < 10: continue
        tasks[t] = {e: np.load(f"{FEAT}/ep{e}.npz")["grid"].astype(np.float32).mean(1) for e in eps}
        if len(tasks) >= nmax: break
    return tasks

def main():
    t0 = time.time(); tasks = load_tasks(12); tids = list(tasks)
    print(f"{len(tids)} tasks ({time.time()-t0:.0f}s)", flush=True)
    rows = []; dists = []
    for ti in tids:
        eps = list(tasks[ti]); rng.shuffle(eps)
        k = max(2, int(len(eps)*0.2)); held = eps[:k]; refe = eps[k:]
        refeps = [tasks[ti][e] for e in refe]
        F, ep, ne, sig = bank_sigma(refeps)
        in_q = np.concatenate([tasks[ti][e] for e in held])
        # off-task 查询: 从别的任务各抽一些帧, 数量≈in_q
        others = [x for x in tids if x != ti]; off_list = []
        per = max(20, len(in_q)//len(others))
        for oj in others:
            oe = list(tasks[oj]); fr = tasks[oj][oe[rng.randint(len(oe))]]
            off_list.append(fr[rng.choice(len(fr), min(per, len(fr)), replace=False)])
        off_q = np.concatenate(off_list)[:len(in_q)]
        r_in = r_query(F, ep, ne, sig, in_q); r_off = r_query(F, ep, ne, sig, off_q)
        y = np.r_[np.ones(len(r_in)), np.zeros(len(r_off))]; s = np.r_[r_in, r_off]
        auc = roc_auc_score(y, s)
        rows.append((ti, auc, r_in.mean(), r_off.mean())); dists.append((ti, r_in, r_off))
        print(f"  task{ti}: AUROC={auc:.3f} | r_in={r_in.mean():.3f} r_off={r_off.mean():.3f} (gap {r_in.mean()-r_off.mean():+.3f})", flush=True)
    aucs = np.array([r[1] for r in rows])
    print(f"\n[SUMMARY] tasks={len(rows)} | AUROC median={np.median(aucs):.3f} min={aucs.min():.3f} | "
          f">0.9 on {np.mean(aucs>0.9)*100:.0f}% | >0.8 on {np.mean(aucs>0.8)*100:.0f}%", flush=True)
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].bar(range(len(aucs)), sorted(aucs, reverse=True), color="#7c3aed", alpha=.85)
    ax[0].axhline(0.5, color="#888", ls="--", label="chance 0.5"); ax[0].axhline(np.median(aucs), color="#22c55e", ls=":", label=f"median {np.median(aucs):.3f}")
    ax[0].set_xlabel("task (sorted)"); ax[0].set_ylabel("AUROC (in-task vs off-task)"); ax[0].set_ylim(0.4, 1.02)
    ax[0].set_title(f"r as on-manifold detector: AUROC median {np.median(aucs):.3f}, >0.9 on {np.mean(aucs>0.9)*100:.0f}% tasks", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=.2)
    for ti, r_in, r_off in dists[:6]:
        ax[1].hist(r_in, bins=25, alpha=.35, color="#22c55e", density=True)
        ax[1].hist(r_off, bins=25, alpha=.35, color="#ef4444", density=True)
    ax[1].set_xlabel("recurrence r vs demo manifold"); ax[1].set_ylabel("density")
    ax[1].set_title("green=in-task frames (high r) vs red=off-task frames (low r)\n(6 example tasks overlaid) -> low r flags off-manifold", fontsize=10); ax[1].grid(alpha=.2)
    fig.suptitle("V3c-proxy · recurrence as deploy-time on-manifold / OOD monitor (cross-task injection, no failure rollouts needed)", fontsize=12)
    fig.tight_layout()
    out = f"{REPO}/lmvla/lmwm/docs/assets/recurrence_ood.png"; fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"SAVED {out} ({time.time()-t0:.0f}s)\nOOD_DONE", flush=True)

if __name__ == "__main__":
    main()
