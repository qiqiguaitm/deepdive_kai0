#!/usr/bin/env python
"""全量导出 kai0_base 3055 ep 的【去阶梯 polyline 双锚 Viterbi】逐帧 value 标签 → per-ep npy.

复用已跑通的 render_kai_online_gru.py 的标签链(shard→PCA128 ⊕ proprio14 → BGMM milestone →
daw() polyline),去掉其 CAP=1000 子采样 + GRU 训练,对【全 3055 ep】逐帧打标.
kai base bank(data/kai_dinov3base)shard 已核验为 native 30Hz(shard_frames==parquet_rows,
FR=0..N-1)→ polyline 长度天然对齐 parquet 行,无需插值.

用于喂 KAI0 pi0-AE 的 crave_stage_poly 数据集. 见 plan:
  docs/training/future_plans/plans/crave_polyline_kai_ae_retrain_plan.md

输出:
  temp/crave_ae_labels/polyline/ep*.npy       (raw 折线, native, 0→1)
  temp/crave_ae_labels/polyline_mono/ep*.npy  (cummax 单调版)
  temp/crave_ae_labels/polyline_sanity.png    (抽 6 held-out ep · polyline vs norm-time)

env: /home/tim/miniconda3/envs/srpo/bin/python
run(全量): PYTHONPATH=lmvla/crave/src python lmvla/crave/experiments/dump_polyline_labels_kai_full.py
run(干跑): ... dump_polyline_labels_kai_full.py --dry 500
"""
import sys, numpy as np, time, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

REPO = Path('/vePFS/tim/workspace/deepdive_kai0')
rng = np.random.RandomState(0)
FPS = 30.; CSQ = 1000
KAI = REPO / 'kai0/data/Task_A/kai0_base'
OUTROOT = REPO / 'lmvla/crave/temp/crave_ae_labels'
LAB = OUTROOT / 'polyline'; LABM = OUTROOT / 'polyline_mono'
LAB.mkdir(parents=True, exist_ok=True); LABM.mkdir(parents=True, exist_ok=True)

DRY = int(sys.argv[sys.argv.index('--dry') + 1]) if '--dry' in sys.argv else None


def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
def cc(a, b): return np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else np.nan


def daw(F, C, P, lam):  # 双锚 Viterbi → polyline(去阶梯) — 逐字复用 render_kai_online_gru.daw
    sC = l2(F[:3].mean(0)[None])[0]; eC = l2(F[-3:].mean(0)[None])[0]
    C2 = np.vstack([C, sC, eC]); Pp = np.concatenate([P, [0.], [1.]])
    bins = np.unique(np.concatenate([[0.], Pp, [1.]])); nb = len(bins)
    cb = [int(np.searchsorted(bins, v)) for v in Pp]; pen = lam * np.abs(bins[:, None] - bins[None])
    de = np.linalg.norm(F[:, None] - C2[None], axis=2); em = np.full((len(F), nb), 1e3)
    for ti in range(len(Pp)): em[:, cb[ti]] = np.minimum(em[:, cb[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(F), nb), int)
    for j in range(1, len(F)):
        tr = cost[None, :] + pen; kk = tr.argmin(1); cost = em[j] + tr[np.arange(nb), kk]; BP[j] = kk
    si = nb - 1; path = np.zeros(len(F), int); path[-1] = si
    for j in range(len(F) - 2, -1, -1): si = BP[j + 1][si]; path[j] = si
    step = bins[path]; segs = []; a = 0
    for t in range(1, len(step)):
        if step[t] != step[t - 1]: segs.append((a, t - 1, step[t - 1])); a = t
    segs.append((a, len(step) - 1, step[-1])); reps = []
    for i0, i1, val in segs:
        cand = [ti for ti in range(len(Pp)) if abs(Pp[ti] - val) < 1e-9]; fr = np.arange(i0, i1 + 1); bd = 1e18; bf = i0
        for ti in cand:
            dd = np.linalg.norm(F[fr] - C2[ti], axis=1); k = int(dd.argmin())
            if dd[k] < bd: bd = dd[k]; bf = fr[k]
        reps.append((bf, float(val)))
    if reps[0][0] != 0: reps = [(0, float(step[0]))] + reps
    if reps[-1][0] != len(step) - 1: reps = reps + [(len(step) - 1, float(step[-1]))]
    rf = np.array([r[0] for r in reps]); rv = np.array([r[1] for r in reps]); keep = np.concatenate([[True], np.diff(rf) > 0])
    return np.interp(np.arange(len(step)), rf[keep], rv[keep]).astype(np.float32)


print('加载 kai base bank...', flush=True); t0 = time.time()
d = REPO / 'lmvla/crave/data/kai_dinov3base'; idx = np.load(d / 'index.npz'); E = idx['E']; FR = idx['FR']
feat = np.zeros((len(E), 768), np.float16)
for sh in sorted(d.glob('shard_*.npz')):
    s = np.load(sh); g = s['gidx']; v = s['valid'] if 'valid' in s else np.ones(len(g), bool); feat[g[v]] = s['feat'][v]
eps = sorted(np.unique(E).tolist())
if DRY:
    eps = [eps[i] for i in sorted(rng.choice(len(eps), min(DRY, len(eps)), replace=False))]
    print(f'  [DRY] 子采样 {len(eps)} eps 冒烟(milestone+labels 都只在这些 ep 上)', flush=True)
keep = np.isin(E, eps); E = E[keep]; FR = FR[keep]; feat = feat[keep]
print(f'  {len(eps)} eps {len(E)} frames; PCA 768→128...', flush=True)
pca = PCA(128, random_state=0).fit(l2(feat[rng.choice(len(feat), min(20000, len(feat)), replace=False)].astype(np.float32)))
IMG = l2((l2(feat.astype(np.float32)) - pca.mean_.astype(np.float32)) @ pca.components_.astype(np.float32).T)

print(f'  [{time.time()-t0:.0f}s] 读 proprio parquet(+native 长度)...', flush=True)
POS = np.zeros((len(E), 14), np.float32); NLEN = {}
for e in eps:
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; fr = FR[m][np.argsort(FR[m])]
    st = np.stack(pd.read_parquet(KAI / f'data/chunk-{e//CSQ:03d}/episode_{e:06d}.parquet',
                  columns=['observation.state'])['observation.state'].to_numpy()).astype(np.float32)
    POS[o] = st[np.minimum(fr, len(st) - 1)]; NLEN[e] = len(st)
SMU = POS.mean(0); SSD = POS.std(0) + 1e-6
JOINT = np.concatenate([IMG, l2((POS - SMU) / SSD)], 1).astype(np.float32)  # 142D, img:pos 能量 1:1
D = JOINT.shape[1]; NC = len(eps)
T = np.zeros(len(E), np.float32)
for e in eps:
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; T[o] = np.linspace(0, 1, len(o))

print(f'  [{time.time()-t0:.0f}s] BayesianGMM on {D}D (img⊕proprio)...', flush=True)
bg = BayesianGaussianMixture(n_components=40, covariance_type='diag', weight_concentration_prior=1e-2,
                             max_iter=120, random_state=0).fit(JOINT[rng.choice(len(JOINT), min(80000, len(JOINT)), replace=False)])
labs = bg.predict(JOINT); C = []; P = []
for k in range(40):
    m = labs == k
    if m.sum() < 20: continue
    if len(set(E[m].tolist())) / NC >= 0.5: C.append(JOINT[m].mean(0)); P.append(float(np.median(T[m])))
C = l2(np.array(C, np.float32)); P = np.array(P); lam = 16. * FPS / 3.
order = np.argsort(P); C = C[order]; P = P[order]
print(f'  [{time.time()-t0:.0f}s] M={len(C)} milestones · P=[{P.min():.3f},{P.max():.3f}] · median 间隔看单调', flush=True)

# ---- 逐 ep polyline teacher → 存 native-len npy ----
print(f'  [{time.time()-t0:.0f}s] 逐 ep daw() polyline → dump...', flush=True)
nd = 0; corrs = []; monos = []; ends = []; SAMPLE = []
for e in eps:
    m = np.where(E == e)[0]; o = m[np.argsort(FR[m])]; f = JOINT[o]; t = T[o]
    poly = daw(f, C, P, lam)
    if len(poly) != NLEN[e]:  # 兜底(shard 已核验 native 对齐, 正常不触发)
        poly = np.interp(np.linspace(0, 1, NLEN[e]), np.linspace(0, 1, len(poly)), poly).astype(np.float32)
    np.save(LAB / f'ep{e}.npy', poly)
    np.save(LABM / f'ep{e}.npy', np.maximum.accumulate(poly).astype(np.float32))
    corrs.append(cc(poly, np.linspace(0, 1, len(poly)))); monos.append(float((np.diff(poly) >= -1e-6).mean()))
    ends.append(float(poly[-1])); nd += 1
    if len(SAMPLE) < 6 and NLEN[e] > 300: SAMPLE.append((e, poly))
    if nd % 500 == 0: print(f'    [{nd}/{len(eps)}] {(time.time()-t0)/60:.1f}min', flush=True)

print(f'\nDONE {nd} eps ({(time.time()-t0)/60:.1f}min)', flush=True)
print(f'  M(milestones)      = {len(C)}   (proven img⊕proprio ≈12; img-only 会塌到 3)', flush=True)
print(f'  polyline vs T corr = {np.nanmean(corrs):.3f}  (proven teacher-vs-T ≈0.95+)', flush=True)
print(f'  单调率(raw)        = {np.mean(monos):.3f}   末值 median = {np.median(ends):.3f} (期望≈1.0)', flush=True)
print(f'  -> {LAB}  +  {LABM}', flush=True)

# sanity 图
fig, axes = plt.subplots(2, 3, figsize=(15, 7)); axes = axes.ravel()
for ax, (e, poly) in zip(axes, SAMPLE):
    ax.plot(np.linspace(0, 1, len(poly)), poly, color='#2ca02c', lw=1.8, label='polyline teacher')
    ax.plot(np.linspace(0, 1, len(poly)), np.linspace(0, 1, len(poly)), color='#e8830c', lw=1.0, alpha=.6, label='norm time')
    ax.set_title(f'ep{e} n={len(poly)} corr={cc(poly, np.linspace(0,1,len(poly))):.3f}', fontsize=9)
    ax.set_ylim(-.03, 1.03); ax.grid(alpha=.25)
axes[0].legend(fontsize=7, loc='lower right')
tag = f'DRY {len(eps)}ep' if DRY else f'FULL {len(eps)}ep'
fig.suptitle(f'polyline 双锚 Viterbi teacher (M={len(C)}, {tag}) · 6 sample eps', fontsize=11)
fig.tight_layout()
sp = OUTROOT / 'polyline_sanity.png'; fig.savefig(sp, dpi=115, bbox_inches='tight'); print('SAVED', sp, flush=True)
