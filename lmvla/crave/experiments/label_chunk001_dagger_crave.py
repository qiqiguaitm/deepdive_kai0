#!/usr/bin/env python3
"""CRAVE pipeline labeling for dagger chunk-001 + matching base data.

Steps:
  1. Load kai_dinov3base features (ALL 3055 eps) → compute PCA
  2. Load proprio from kai0_base → compute SMU/SSD
  3. Compute joint (img⊕proprio) features → BGMM → M milestones
  4. Sample 387 base eps + label ALL 387 chunk-001 dagger eps with daw()
  5. Save per-ep stage_progress_gt npy files

Uses EXACT same pipeline as dump_polyline_labels_kai_full.py (same PCA, BGMM, daw).
Output: temp/crave_ae_labels/chunk001_val/{base,dagger}/

Run: PYTHONPATH=lmvla/crave/src python lmvla/crave/experiments/label_chunk001_dagger_crave.py [--dry 50]
"""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
rng = np.random.RandomState(0)
FPS = 30.0; CSQ = 1000
KAI = REPO / "kai0/data/Task_A/kai0_base"
OUT = REPO / "lmvla/crave/temp/crave_ae_labels/chunk001_val"
DRY = int(sys.argv[sys.argv.index("--dry") + 1]) if "--dry" in sys.argv else None
N_BASE = 387  # match dagger volume
SEED = 42

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
def cc(a, b): return np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else np.nan

def daw(F, C, P, lam):
    """双锚 Viterbi → polyline (exact copy from render_kai_online_gru.py)."""
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
    rf = np.array([r[0] for r in reps]); rv = np.array([r[1] for r in reps])
    keep = np.concatenate([[True], np.diff(rf) > 0])
    return np.interp(np.arange(len(step)), rf[keep], rv[keep]).astype(np.float32)


# ==================== Step 1: Load BASE features + PCA ====================
print("加载 kai_dinov3base...", flush=True); t0 = time.time()
d = REPO / "lmvla/crave/data/kai_dinov3base"
idx = np.load(d / "index.npz"); Eb = idx["E"]; FRb = idx["FR"]
feat_b = np.zeros((len(Eb), 768), np.float16)
for sh in sorted(d.glob("shard_*.npz")):
    s = np.load(sh); g = s["gidx"]; v = s["valid"] if "valid" in s else np.ones(len(g), bool)
    feat_b[g[v]] = s["feat"][v]
base_eps = sorted(np.unique(Eb).tolist())
print(f"  {len(base_eps)} eps / {len(Eb)} frames", flush=True)

print("PCA 768→128...", flush=True)
pca = PCA(128, random_state=0).fit(l2(feat_b[rng.choice(len(feat_b), min(20000, len(feat_b)), replace=False)].astype(np.float32)))
IMG_b = l2((l2(feat_b.astype(np.float32)) - pca.mean_.astype(np.float32)) @ pca.components_.astype(np.float32).T)

# ==================== Step 2: Proprio + Joint ====================
print(f"[{time.time()-t0:.0f}s] 读 base proprio + compute SMU/SSD...", flush=True)
POS_b = np.zeros((len(Eb), 14), np.float32)
# ⚠️ np.where per-ep 对 3.36M 做 3055 次 O(N) 扫描 → 2h+.
# 改: np.unique 一次分桶 → O(N).
_, ep_inv, ep_counts = np.unique(Eb, return_inverse=True, return_counts=True)
ep_offsets = np.zeros(len(ep_counts) + 1, dtype=np.int64)
np.cumsum(ep_counts, out=ep_offsets[1:])
# 逐块读 proprio (分桶已对齐 Eb 顺序)
for idx, e in enumerate(base_eps):
    start, end = ep_offsets[idx], ep_offsets[idx + 1]
    o_sorted = np.arange(start, end)[np.argsort(FRb[start:end])]
    fr = FRb[o_sorted]
    st = np.stack(pd.read_parquet(KAI / f"data/chunk-{e//CSQ:03d}/episode_{e:06d}.parquet",
                  columns=["observation.state"])["observation.state"].to_numpy()).astype(np.float32)
    POS_b[o_sorted] = st[np.minimum(fr, len(st) - 1)]
    if e % 1000 == 0: print(f"  proprio {e}/{max(base_eps)}", flush=True)
SMU = POS_b.mean(0); SSD = POS_b.std(0) + 1e-6
JOINT_b = np.concatenate([IMG_b, l2((POS_b - SMU) / SSD)], 1).astype(np.float32)
D = JOINT_b.shape[1]; NC = len(base_eps)
T_b = np.zeros(len(Eb), np.float32)
for idx, e in enumerate(base_eps):
    start, end = ep_offsets[idx], ep_offsets[idx + 1]
    o = np.arange(start, end)[np.argsort(FRb[start:end])]
    T_b[o] = np.linspace(0, 1, end - start)

# ==================== Step 3: BGMM milestones ====================
print(f"[{time.time()-t0:.0f}s] BayesianGMM on {D}D...", flush=True)
bg = BayesianGaussianMixture(n_components=40, covariance_type="diag", weight_concentration_prior=1e-2,
                             max_iter=120, random_state=0).fit(JOINT_b[rng.choice(len(JOINT_b), min(80000, len(JOINT_b)), replace=False)])
labs = bg.predict(JOINT_b); C, P = [], []
for k in range(40):
    m = labs == k
    if m.sum() < 20: continue
    if len(set(Eb[m].tolist())) / NC >= 0.5:
        C.append(JOINT_b[m].mean(0)); P.append(float(np.median(T_b[m])))
C = l2(np.array(C, np.float32)); P = np.array(P); lam = 16.0 * FPS / 3.0
order = np.argsort(P); C = C[order]; P = P[order]
print(f"  M={len(C)} milestones  P=[{P.min():.3f}, {P.max():.3f}]", flush=True)

# ==================== Step 4: Sample base eps + label ====================
(BASE_OUT := OUT / "base").mkdir(parents=True, exist_ok=True)
np.random.seed(SEED); sampled_base = sorted(np.random.choice(base_eps, min(N_BASE, len(base_eps)), replace=False))
sampled_base_set = set(sampled_base)
print(f"\n标 {len(sampled_base)} base eps...", flush=True)
b_corrs = []
for idx, e in enumerate(base_eps):
    if e not in sampled_base_set: continue
    start, end = ep_offsets[idx], ep_offsets[idx + 1]
    o = np.arange(start, end)[np.argsort(FRb[start:end])]
    f = JOINT_b[o]
    poly = daw(f, C, P, lam); np.save(BASE_OUT / f"ep{e}.npy", poly)
    b_corrs.append(cc(poly, np.linspace(0, 1, len(poly))))
print(f"  base polyline vs T corr: mean={np.nanmean(b_corrs):.3f}", flush=True)

# ==================== Step 5: Load dagger features + label ====================
(DAGG_OUT := OUT / "dagger").mkdir(parents=True, exist_ok=True)
dd = REPO / "lmvla/crave/data/dagger_chunk001_dinov3base"
if not (dd / "index.npz").exists():
    print("\nERROR: dagger features not extracted yet. Run extract_dagger_chunk001_d3b.py first.", flush=True)
    sys.exit(1)

print(f"\n加载 dagger features...", flush=True)
idx_d = np.load(dd / "index.npz"); Ed = idx_d["E"]; FRd = idx_d["FR"]
feat_d = np.zeros((len(Ed), 768), np.float16)
for sh in sorted(dd.glob("shard_*.npz")):
    s = np.load(sh); g = s["gidx"]; v = s["valid"] if "valid" in s else np.ones(len(g), bool)
    feat_d[g[v]] = s["feat"][v]
dagger_eps = sorted(np.unique(Ed).tolist())
dagger_eps_set = set(dagger_eps)
print(f"  {len(dagger_eps)} dagger eps / {len(Ed)} frames", flush=True)

# Apply PCA from base
IMG_d = l2((l2(feat_d.astype(np.float32)) - pca.mean_.astype(np.float32)) @ pca.components_.astype(np.float32).T)

# Load proprio from dagger parquets
DAGGER_ROOT = REPO / "kai0/data/Task_A/vis_dagger/v4"
print(f"[{time.time()-t0:.0f}s] 读 dagger proprio...", flush=True)
_, ed_inv, ed_counts = np.unique(Ed, return_inverse=True, return_counts=True)
ed_offsets = np.zeros(len(ed_counts) + 1, dtype=np.int64)
np.cumsum(ed_counts, out=ed_offsets[1:])
POS_d = np.zeros((len(Ed), 14), np.float32)
# 预建 global_ep→parquet_path (extraction 用 date_hash*10000+local_ep 编码全局 id)
def decode_global_ep(ge):
    """date_hash * 10000 + local_ep → (date_dir, local_ep)"""
    dh = ge // 10000; local = ge % 10000
    month = dh // 100; day = dh % 100
    return f"2026-{month:02d}-{day:02d}-v4", local

ep2pq = {}
for pq in DAGGER_ROOT.glob("*/data/chunk-001/episode_*.parquet"):
    local_ep = int(pq.stem.split("_")[1])
    dt = pq.parent.parent.parent.name  # 2026-XX-XX-v4
    parts = dt.split("-"); dh = int(parts[1]) * 100 + int(parts[2])
    ge = dh * 10000 + local_ep
    if ge in dagger_eps_set: ep2pq[ge] = (pq, local_ep)

for idx, e in enumerate(dagger_eps):
    start, end = ed_offsets[idx], ed_offsets[idx + 1]
    o_sorted = np.arange(start, end)[np.argsort(FRd[start:end])]
    fr = FRd[o_sorted]
    entry = ep2pq.get(e)
    if entry is None: continue
    pq, local_ep = entry
    st = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy()).astype(np.float32)
    POS_d[o_sorted] = st[np.minimum(fr, len(st) - 1)]
JOINT_d = np.concatenate([IMG_d, l2((POS_d - SMU) / SSD)], 1).astype(np.float32)

# Label dagger eps
print(f"\n标 {len(dagger_eps)} dagger eps...", flush=True)
d_corrs = []; d_ends = []
if DRY:
    dagger_eps = sorted(np.random.choice(dagger_eps, min(DRY, len(dagger_eps)), replace=False))
    print(f"  [DRY] {len(dagger_eps)} eps", flush=True)

for idx, e in enumerate(dagger_eps):
    start, end = ed_offsets[idx], ed_offsets[idx + 1]
    o = np.arange(start, end)[np.argsort(FRd[start:end])]
    f = JOINT_d[o]
    poly = daw(f, C, P, lam); np.save(DAGG_OUT / f"ep{e}.npy", poly)
    d_corrs.append(cc(poly, np.linspace(0, 1, len(poly))))
    d_ends.append(float(poly[-1]))

el = (time.time() - t0) / 60
print(f"\nDONE ({el:.1f}min)", flush=True)
print(f"  M={len(C)} milestones", flush=True)
print(f"  base ({len(sampled_base)}): polyline vs T corr = {np.nanmean(b_corrs):.3f}", flush=True)
print(f"  dagger ({len(dagger_eps)}): polyline vs T corr = {np.nanmean(d_corrs):.3f}  末值 median = {np.median(d_ends):.3f}", flush=True)
print(f"  → {BASE_OUT}  +  {DAGG_OUT}", flush=True)

# ==================== Sanity plot ====================
sample_dag = sorted(dagger_eps)[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 7)); axes = axes.ravel()
for ax, e in zip(axes, sample_dag):
    poly = np.load(DAGG_OUT / f"ep{e}.npy"); t = np.linspace(0, 1, len(poly))
    ax.plot(t, poly, color="#2ca02c", lw=1.8, label="polyline")
    ax.plot(t, t, color="#e8830c", lw=1.0, alpha=0.6, label="norm time")
    ax.set_title(f"dagger ep{e} n={len(poly)} corr={cc(poly, t):.3f}", fontsize=9)
    ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25)
axes[0].legend(fontsize=7, loc="lower right")
tag = f"DRY {len(dagger_eps)}" if DRY else f"FULL {len(dagger_eps)}"
fig.suptitle(f"CRAVE polyline labels — dagger chunk-001 ({tag} eps, M={len(C)})", fontsize=11)
fig.tight_layout()
sp = OUT / "label_sanity.png"; fig.savefig(sp, dpi=115, bbox_inches="tight")
print(f"SAVED {sp}", flush=True)
