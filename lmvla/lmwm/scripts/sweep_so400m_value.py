#!/usr/bin/env python
"""A线特化扫: v1 配方在 So400m-mean 空间的有界调参(全量 3055ep, 只评 corr 不落标签)。
基线(pca128/nc40/wcp1e-2/权重1:1)=0.578, 生产 DINOv3=0.948。
变体: pca256 | nc60 | img权重x2 | pca128+nc60+wcp1e-3(组合)。
用法: python sweep_so400m_value.py <variant>   variant∈ base,pca256,nc60,imgw2,combo
"""
import glob, os, sys, time
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from scipy.ndimage import gaussian_filter1d

V = sys.argv[1]
CFG = {"base":   dict(pca=128, nc=40, wcp=1e-2, imgw=1.0),
       "pca256": dict(pca=256, nc=40, wcp=1e-2, imgw=1.0),
       "nc60":   dict(pca=128, nc=60, wcp=1e-2, imgw=1.0),
       "imgw2":  dict(pca=128, nc=40, wcp=1e-2, imgw=2.0),
       "combo":  dict(pca=128, nc=60, wcp=1e-3, imgw=1.0)}[V]
REPO = "/vePFS/tim/workspace/deepdive_kai0"
FEAT = f"{REPO}/lmvla/lmwm/data/kai0_aligned_urvc/so400m-mean_s10"
CSQ, MIN_COV, LAM = 1000, 0.50, 16.0
DS = lambda e: f"{REPO}/kai0/data/Task_A/kai0_base/data/chunk-{e//CSQ:03d}/episode_{e:06d}.parquet"
GTP = lambda e: f"{REPO}/kai0/data/Task_A/kai0_advantage/data/chunk-{e//CSQ:03d}/episode_{e:06d}.parquet"
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

files = sorted(glob.glob(f"{FEAT}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4]))
imgL, stL, nlist = [], [], []
for f in files:
    e = int(os.path.basename(f)[2:-4]); d = np.load(f)
    p, fidx = d["pooled"].astype(np.float32), d["fidx"]
    st = np.stack(pd.read_parquet(DS(e), columns=["observation.state"])["observation.state"].to_numpy())
    imgL.append(p); stL.append(st[np.minimum(fidx, len(st)-1)].astype(np.float32)); nlist.append((e, len(p)))
img = np.concatenate(imgL); ST = np.concatenate(stL); del imgL, stL
T = np.concatenate([np.linspace(0, 1, n) for _, n in nlist]); Ev = np.concatenate([np.full(n, e) for e, n in nlist])
NC_ = len(nlist)

pca = PCA(n_components=CFG["pca"], random_state=0).fit(l2(img))
fq = l2(pca.transform(l2(img)).astype(np.float32)) * CFG["imgw"]
SMU, SSD = ST.mean(0), ST.std(0) + 1e-8
Jn = l2(np.concatenate([fq, l2((ST - SMU) / SSD)], 1)); del img, fq
t0 = time.time()
bgmm = BayesianGaussianMixture(n_components=CFG["nc"], covariance_type="diag", weight_concentration_prior=CFG["wcp"],
                               n_init=1, max_iter=150, random_state=0).fit(Jn)
labs = bgmm.predict(Jn)
print(f"[{V}] eff={(bgmm.weights_>0.01).sum()}/{CFG['nc']} ({time.time()-t0:.0f}s)", flush=True)

def mode_split(Tc, nbins=30):
    h, ed = np.histogram(Tc, bins=nbins, range=(0, 1)); h = h.astype(float)/(h.sum()+1e-9)
    hs = gaussian_filter1d(h, 1.2); c = (ed[:-1]+ed[1:])/2
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

cand = np.array([Jn[labs == k].mean(0) for k in range(CFG["nc"]) if (labs == k).sum() >= 20], np.float32)
assign = np.empty(len(Jn), int)
for i in range(0, len(Jn), 20000):
    assign[i:i+20000] = np.linalg.norm(Jn[i:i+20000, None]-cand[None], axis=2).argmin(1)
targets = []
for ki in range(len(cand)):
    mk = assign == ki
    if mk.sum() < 20: continue
    for mv, sub in mode_split(T[mk]):
        if len(set(Ev[mk][sub].tolist())) / NC_ >= MIN_COV: targets.append((float(np.median(T[mk][sub])), cand[ki]))
targets.sort(key=lambda t: t[0])
vals = np.array([t[0] for t in targets]); Ctgt = np.array([t[1] for t in targets], np.float32)
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

ptr = 0; cs = []
for e, n in nlist:
    v = vit(Jn[ptr:ptr+n]); ptr += n
    gt = pd.read_parquet(GTP(e), columns=["stage_progress_gt"])["stage_progress_gt"].values.astype(np.float32)
    v30 = np.interp(np.linspace(0, 1, len(gt)), np.linspace(0, 1, n), v)
    lo, hi = v30.min(), v30.max()
    if hi > lo+1e-6: v30 = (v30-lo)/(hi-lo)
    if np.std(v30) > 1e-6 and np.std(gt) > 1e-6: cs.append(np.corrcoef(v30, gt)[0, 1])
print(f"[{V}] M={len(vals)} corr(med)={np.median(cs):.3f}  (基线 base=0.578, 生产 DINOv3=0.948)")
