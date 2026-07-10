#!/usr/bin/env python
"""代表帧折线读出(polyline)—— 在双锚 Viterbi 分段基础上的平滑替代.

思路(在 viterbi 基础上):
  ① 双锚 Viterbi → 逐帧 milestone value(阶梯 step);
  ② 每个连续 stage 段, 取【离该簇质心最近的一帧】为代表帧, 钉在簇 value;
  ③ 相邻代表帧之间线性连成折线, 每帧 value = 折线对应值.
硬阶梯 → 锚在"最典型帧"上的分段线性; 更贴合平滑的监督 progress.

实测(200ep vs 监督 stage_gt):
  阶梯 step:      corr 0.944  单调 0.981
  折线 polyline:  corr 0.957  单调 0.790   ← corr 反超(GT 本身是平滑 ramp)
  折线+cummax:    corr ≈0.957 单调 1.000   ← 单调版(与 crave_stage_B 一致)

输出:
  temp/crave_ae_labels/polyline/ep*.npy       (raw 折线, 30Hz, 0→1)
  temp/crave_ae_labels/polyline_mono/ep*.npy  (cummax 单调版)
Run: PYTHONPATH="<repo>/lmwam(lmvla)/crave/src" python .../gen_polyline_labels.py
"""
import time, numpy as np, pandas as pd
from pathlib import Path
from crave.config import resolve_dataset

REPO = Path("/home/tim/workspace/deepdive_kai0"); LAM = 16.0; CSQ = 1000
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

s = np.load(REPO / "temp/crave_final_v3.npz")
vals = s["vals"]; Ctgt = s["Ctgt"]; pca_m = s["pca_mean"]; pca_c = s["pca_components"]; SMU = s["SMU"]; SSD = s["SSD"]
FEAT = REPO / "temp/crave_d3b_pca128/feats"; eps = sorted(int(p.stem[2:]) for p in FEAT.glob("ep*.npy"))
cfg = resolve_dataset("kai0_base"); DS = Path(cfg.root)
zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E_idx, FR_idx = zf["E"], zf["FR"]

def feat3(e):
    f = np.load(FEAT / f"ep{e}.npy").astype(np.float32); fq = l2((l2(f) - pca_m) @ pca_c.T); n = len(fq)
    loc = np.where(E_idx == e)[0]; o = np.argsort(FR_idx[loc]); fr = FR_idx[loc][o][:n]
    st = np.stack(pd.read_parquet(DS / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                                   columns=["observation.state"])["observation.state"].to_numpy()).astype(np.float32)
    return np.concatenate([fq, l2((st[np.minimum(fr, len(st) - 1)] - SMU) / SSD)], 1), len(st)

print(f"computing anchors over {len(eps)} eps...", flush=True)
ss = []; ee = []
for e in eps: F, _ = feat3(e); ss.append(F[:3].mean(0)); ee.append(F[-3:].mean(0))
sC = l2(np.mean(ss, 0)[None])[0]; eC = l2(np.mean(ee, 0)[None])[0]
Ct2 = np.vstack([Ctgt, sC, eC]).astype(np.float32); Pord = np.concatenate([vals, [0.], [1.]])
bins = np.unique(np.concatenate([[0.], Pord, [1.]])); nb = len(bins)
cb = [int(np.searchsorted(bins, v)) for v in Pord]; pen = LAM * np.abs(bins[:, None] - bins[None])

def viterbi(Fq):   # 逐帧 value(阶梯)
    de = np.linalg.norm(Fq[:, None] - Ct2[None], axis=2); em = np.full((len(Fq), nb), 1e3)
    for ti in range(len(Pord)): em[:, cb[ti]] = np.minimum(em[:, cb[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(Fq), nb), int)
    for j in range(1, len(Fq)):
        tr = cost[None, :] + pen; kk = tr.argmin(1); cost = em[j] + tr[np.arange(nb), kk]; BP[j] = kk
    si = nb - 1; path = np.zeros(len(Fq), int); path[-1] = si
    for j in range(len(Fq) - 2, -1, -1): si = BP[j + 1][si]; path[j] = si
    return bins[path]

def polyline(Fq):  # 阶梯 → 代表帧折线
    step = viterbi(Fq); N = len(Fq); segs = []; a = 0
    for t in range(1, N):
        if step[t] != step[t - 1]: segs.append((a, t - 1, step[t - 1])); a = t
    segs.append((a, N - 1, step[-1]))
    reps = []
    for (i0, i1, val) in segs:
        cand = [ti for ti in range(len(Pord)) if abs(Pord[ti] - val) < 1e-9]   # 同 value 候选簇
        fr = np.arange(i0, i1 + 1); bestd = 1e18; bestf = i0
        for ti in cand:
            d = np.linalg.norm(Fq[fr] - Ct2[ti], axis=1); k = int(d.argmin())
            if d[k] < bestd: bestd = d[k]; bestf = fr[k]
        reps.append((bestf, float(val)))
    if reps[0][0] != 0: reps = [(0, float(step[0]))] + reps
    if reps[-1][0] != N - 1: reps = reps + [(N - 1, float(step[-1]))]
    rf = np.array([r[0] for r in reps]); rv = np.array([r[1] for r in reps])
    keep = np.concatenate([[True], np.diff(rf) > 0]); rf = rf[keep]; rv = rv[keep]   # rf 严格递增
    return np.interp(np.arange(N), rf, rv)

LAB = REPO / "temp/crave_ae_labels/polyline"; LABM = REPO / "temp/crave_ae_labels/polyline_mono"
LAB.mkdir(parents=True, exist_ok=True); LABM.mkdir(parents=True, exist_ok=True)
t0 = time.time(); nd = 0; monos = []; ends = []
for e in eps:
    F, N = feat3(e); poly = polyline(F)
    poly30 = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(poly)), poly).astype(np.float32)   # 3Hz→30Hz
    np.save(LAB / f"ep{e}.npy", poly30)
    np.save(LABM / f"ep{e}.npy", np.maximum.accumulate(poly30).astype(np.float32))   # cummax 单调版
    monos.append(float((np.diff(poly30) >= -1e-6).mean())); ends.append(float(poly30[-1])); nd += 1
    if nd % 1000 == 0: print(f"  [{nd}/{len(eps)}] {(time.time()-t0)/60:.1f}min", flush=True)
print(f"DONE {nd} eps · 末值median={np.median(ends):.3f} 单调(raw)={np.mean(monos):.3f} "
      f"({(time.time()-t0)/60:.1f}min) -> {LAB} + {LABM}", flush=True)
