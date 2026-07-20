#!/usr/bin/env python
"""A线 gate: CRAVE v1 value 管线(gen_final_v3 配方)换特征空间的标签质量对照。
唯一变量 = 视觉特征(DINOv3 vs So400m-mean, 同一批 110 对齐 ep); 其余照抄 v1:
PCA128 ⊕ proprio14(各自L2 1:1) → BGMM(40,diag,wcp1e-2) → mode_split + per-mode coverage≥0.5
→ median 值 → 多发射 Viterbi(λ=16) → per-ep 插值 → corr/mono/end vs stage_progress_gt。
判据: So400m-mean 的 corr 不显著低于 DINOv3 同小样 → A线全量绿灯。
用法: python kai0_v1gate_space.py dino|so400m-mean"""
import glob, os, sys, time
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from scipy.ndimage import gaussian_filter1d

ENC = sys.argv[1]
BASE = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/kai0_aligned_urvc"
DS = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_base/data/chunk-000"
GTD = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_advantage/data/chunk-000"
MIN_COV, LAM = 0.50, 16.0
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

imgL, stL, gtL, nlist = [], [], [], []
for f in sorted(glob.glob(f"{BASE}/{ENC}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4])):
    e = int(os.path.basename(f)[2:-4]); d = np.load(f)
    p, fidx = d["pooled"].astype(np.float32), d["fidx"]
    st = np.stack(pd.read_parquet(f"{DS}/episode_{e:06d}.parquet", columns=["observation.state"])["observation.state"].to_numpy())
    gt = pd.read_parquet(f"{GTD}/episode_{e:06d}.parquet", columns=["stage_progress_gt"])["stage_progress_gt"].values
    idx = np.minimum(fidx, len(st) - 1)
    imgL.append(p); stL.append(st[idx].astype(np.float32)); gtL.append(gt[np.minimum(fidx, len(gt)-1)].astype(np.float32))
    nlist.append(len(p))
NC = len(imgL)
img = np.concatenate(imgL); ST = np.concatenate(stL)
T = np.concatenate([np.linspace(0, 1, n) for n in nlist]); Ev = np.concatenate([np.full(n, i) for i, n in enumerate(nlist)])

pca = PCA(n_components=128, random_state=0).fit(l2(img))
fq = l2(pca.transform(l2(img)).astype(np.float32))
SMU, SSD = ST.mean(0), ST.std(0) + 1e-8
jointF = np.concatenate([fq, l2((ST - SMU) / SSD)], 1); Jn = l2(jointF)

t0 = time.time()
bgmm = BayesianGaussianMixture(n_components=40, covariance_type="diag", weight_concentration_prior=1e-2,
                               n_init=1, max_iter=150, random_state=0).fit(Jn)
labs = bgmm.predict(Jn)
print(f"[{ENC}] eff_components={(bgmm.weights_>0.01).sum()} ({time.time()-t0:.0f}s)", flush=True)

def mode_split(Tc, nbins=30):
    h, ed = np.histogram(Tc, bins=nbins, range=(0, 1)); h = h.astype(float) / (h.sum() + 1e-9)
    hs = gaussian_filter1d(h, 1.2); c = (ed[:-1] + ed[1:]) / 2
    peaks = [i for i in range(nbins) if hs[i] >= hs[max(0,i-1)] and hs[i] >= hs[min(nbins-1,i+1)] and hs[i] >= 0.10*hs.max()]
    merged = []
    for p in peaks:
        if merged and abs(c[p]-c[merged[-1]]) < 0.10:
            if hs[p] > hs[merged[-1]]: merged[-1] = p
        else: merged.append(p)
    final = [merged[0]] if merged else [int(np.argmax(hs))]
    for p in merged[1:]:
        valley = hs[final[-1]:p+1].min()
        if valley < 0.6*min(hs[final[-1]], hs[p]): final.append(p)
        elif hs[p] > hs[final[-1]]: final[-1] = p
    if len(final) <= 1: return [(float(np.median(Tc)), np.ones(len(Tc), bool))]
    cuts = [c[a+int(np.argmin(hs[a:b+1]))] for a, b in zip(final[:-1], final[1:])]
    edges = [0.0]+cuts+[1.0]; out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (Tc >= lo) & (Tc < hi)
        if msk.sum() >= 5: out.append((float(np.median(Tc[msk])), msk))
    return out if out else [(float(np.median(Tc)), np.ones(len(Tc), bool))]

cand = []
for k in range(40):
    mk = labs == k
    if mk.sum() >= 20: cand.append(Jn[mk].mean(0))
cand = np.array(cand, np.float32)
assign = np.empty(len(Jn), int)
for i in range(0, len(Jn), 20000):
    assign[i:i+20000] = np.linalg.norm(Jn[i:i+20000, None]-cand[None], axis=2).argmin(1)

targets = []
for ki in range(len(cand)):
    mk = assign == ki
    if mk.sum() < 20: continue
    for mv, sub in mode_split(T[mk]):
        cov = len(set(Ev[mk][sub].tolist())) / NC
        if cov >= MIN_COV: targets.append((float(np.median(T[mk][sub])), cand[ki]))
targets.sort(key=lambda t: t[0])
vals = np.array([t[0] for t in targets]); Ctgt = np.array([t[1] for t in targets], np.float32)
print(f"[{ENC}] M={len(vals)} milestones: {[round(v,2) for v in vals]}", flush=True)

bins = np.unique(np.concatenate([[0.0], vals, [1.0]])); nb = len(bins)
cbn = [int(np.searchsorted(bins, v)) for v in vals]; pen = LAM*np.abs(bins[:, None]-bins[None])
def vit(Fq):
    de = np.linalg.norm(Fq[:, None]-Ctgt[None], axis=2); em = np.full((len(Fq), nb), 1e3)
    for ti in range(len(vals)): em[:, cbn[ti]] = np.minimum(em[:, cbn[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(Fq), nb), int)
    for j in range(1, len(Fq)):
        tr = cost[None, :]+pen; kk = tr.argmin(1); cost = em[j]+tr[np.arange(nb), kk]; BP[j] = kk
    cost[nb-1] -= 2; s = int(cost.argmin()); path = np.zeros(len(Fq), int); path[-1] = s
    for j in range(len(Fq)-2, -1, -1): s = BP[j+1][s]; path[j] = s
    return bins[path]

ptr = 0; cs, monos, ends = [], [], []
for i, n in enumerate(nlist):
    v = vit(Jn[ptr:ptr+n]); ptr += n
    lo, hi = v.min(), v.max()
    if hi > lo+1e-6: v = (v-lo)/(hi-lo)
    if np.std(v) > 1e-6 and np.std(gtL[i]) > 1e-6: cs.append(np.corrcoef(v, gtL[i])[0, 1])
    monos.append((np.diff(v) >= -1e-6).mean()); ends.append(v[-1])
print(f"[{ENC}] v1-gate: corr(med)={np.median(cs):.3f}  mono={np.mean(monos):.3f}  end={np.mean(ends):.3f}  (M={len(vals)})")
print("参照: v1 全量收口(3055ep,DINOv3) corr 0.943/mono 0.981/end 0.999; 本 gate 为 110ep 小样, 只看两空间相对差")
