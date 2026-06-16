"""CRAVE value 统一计算库(高内聚低耦合)。
三层:
  ① 共享:FeatureSpace(三路特征 raw⊕armmask⊕proprio)、viterbi(Viterbi-DP)、med/mono/advantage 指标
  ② DiscreteValue(V2.4 离散 milestone 阶梯)—— 1:1 复刻 hdf5_v24_eval.build_model
  ③ ContinuousValue(TCC frozen-feature 头 + 相似度场 DP + 子bin软期望 连续读出)
特征缓存格式: ep*.npz 含 raw/armmask/state (3Hz)。离散纯 numpy/sklearn;连续需 torch+xirl。
"""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans

# ============================== ① 共享 ==============================
def loadep(fc, e):
    d = np.load(fc / f"ep{e}.npz"); a, r, s = d["armmask"], d["raw"], d["state"]
    n = min(len(a), len(r), len(s)); return a[:n], r[:n], s[:n], n

def mkp(s):
    return np.concatenate([s, np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])], 1)

def med(a, w):
    h = w // 2; return np.array([np.median(a[max(0, j - h):j + h + 1]) for j in range(len(a))])

def advantage(v, W=50):
    return np.array([v[min(i + W, len(v) - 1)] - v[i] for i in range(len(v))])

def mono(v):
    return float(np.mean(np.diff(v) >= -1e-6))

def adv_density(v, W=50):
    return float(np.mean(np.abs(np.clip(advantage(v, W), -1, 1)) > 1e-3))

def viterbi(emit, bins, lam, end_bonus=2.0):
    """通用 Viterbi-DP: emit(NF,NB) 代价场, bins 进度刻度, lam 转移惩罚, 末格奖励 end_bonus。返回 bins[path]。"""
    NB = len(bins); pen = lam * np.abs(bins[:, None] - bins[None]); NF = len(emit)
    cost = np.full(NB, 1e9); cost[0] = emit[0, 0]; bp = np.zeros((NF, NB), int)
    for j in range(1, NF):
        tr = cost[None, :] + pen; k = tr.argmin(1); cost = emit[j] + tr[np.arange(NB), k]; bp[j] = k
    cost[NB - 1] -= end_bonus; path = np.zeros(NF, int); path[-1] = cost.argmin()
    for j in range(NF - 2, -1, -1): path[j] = bp[j + 1, path[j + 1]]
    return bins[path], path

class FeatureSpace:
    """三路嵌入: raw-DINOv2 ⊕ armmask-DINOv2 ⊕ proprio(state+Δstate, z-score)。PMU/PSD 来自 mine_eps。"""
    def __init__(self, fc, mine_eps):
        self.fc = fc
        Pm = mkp(np.concatenate([loadep(fc, e)[2] for e in mine_eps]))
        self.PMU, self.PSD = Pm.mean(0), Pm.std(0) + 1e-8

    def emb(self, a_, r_, s_):
        an = a_ / np.linalg.norm(a_, axis=1, keepdims=True)
        rn = r_ / np.linalg.norm(r_, axis=1, keepdims=True)
        Pn = (mkp(s_) - self.PMU) / self.PSD; Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
        return np.concatenate([rn, an, Pn], 1)

    def emb_ep(self, e):
        a, r, s, _ = loadep(self.fc, e); return self.emb(a, r, s)


# ====================== ② 离散 CRAVE V2.4 ======================
class DiscreteValue:
    """V2.4 离散 milestone 阶梯。与 hdf5_v24_eval.build_model 逐字一致(KMeans96+coverage修正+进度分桶+端点锚+Viterbi-DP)。"""
    def __init__(self, fs: FeatureSpace, eps, k=96, log=print):
        self.fs = fs
        A, R, S, T, E, SP, EP = [], [], [], [], [], [], []
        for e in eps:
            a, r, s, n = loadep(fs.fc, e); g = fs.emb(a, r, s)
            A.append(a); R.append(r); S.append(s); T.append(np.arange(n) / max(1, n - 1)); E.append(np.full(n, e))
            SP.append(g[:2]); EP.append(g[-2:])
        A = np.concatenate(A); R = np.concatenate(R); S = np.concatenate(S); T = np.concatenate(T); E = np.concatenate(E)
        G = fs.emb(A, R, S)
        km = KMeans(k, n_init=2, random_state=0).fit(G); lab = km.labels_; allC = km.cluster_centers_
        N = len(set(E.tolist())); tpos = np.array([T[lab == c].mean() if (lab == c).any() else .5 for c in range(k)])
        Pstart = {}
        for e in sorted(set(E.tolist())):
            m = np.where(E == e)[0][:3]; nn = np.linalg.norm(G[m][:, None] - allC[None], axis=2).argmin(1)
            Pstart[e] = float(np.median(tpos[nn]))
        cov_n = np.array([min(1, (len(set(E[lab == c].tolist())) + sum(1 for e in Pstart if Pstart[e] > tpos[c] + 0.1)) / N) for c in range(k)])
        bk = np.linspace(0, 1, 11); sel = []
        for b in range(10):
            inb = [c for c in range(k) if bk[b] <= tpos[c] < bk[b + 1]]
            if inb: sel += sorted(inb, key=lambda c: -cov_n[c])[:2]
        sel = sorted(set(sel), key=lambda c: tpos[c])

        def gr(idx):
            o = []; s0 = None; pv = None
            for i in idx:
                if pv is None or i != pv + 1:
                    if s0 is not None: o.append((s0, pv))
                    s0 = i
                pv = i
            if s0 is not None: o.append((s0, pv))
            return [x for x in o if x[1] - x[0] >= 1]

        Pk = {}
        for c in sel:
            fe = []
            for e in sorted(set(E.tolist())):
                m = np.where(E == e)[0]; rs = gr(m[lab[m] == c].tolist())
                if rs: fe.append(T[rs[0][0]])
            Pk[c] = float(np.median(fe)) if fe else float(tpos[c])
        self.order = sorted(sel, key=lambda c: Pk[c]); self.C = allC[self.order]
        self.Pord = np.array([Pk[c] for c in self.order]); self.Pk = Pk
        log(f"[DiscreteValue] milestones: {len(self.order)}  前段(P<0.5): {sum(1 for c in self.order if Pk[c] < 0.5)}")
        self.startK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(SP)).cluster_centers_
        self.endK = KMeans(8, n_init=2, random_state=0).fit(np.concatenate(EP)).cluster_centers_
        self.NB = 21; self.bins = np.linspace(0, 1, self.NB)
        self.cb = [[int(np.argmin(abs(self.bins - Pk[c])))] for c in self.order]

    def value(self, a, r, s, ret_lab=False):
        Fq = self.fs.emb(a, r, s); nq = len(Fq); d = np.linalg.norm(Fq[:, None] - self.C[None], axis=2)
        em = np.full((nq, self.NB), 1e3)
        for ci in range(len(self.order)):
            for b in self.cb[ci]: em[:, b] = np.minimum(em[:, b], d[:, ci])
        ds = np.linalg.norm(Fq[:, None] - self.startK[None], axis=2).min(1)
        de = np.linalg.norm(Fq[:, None] - self.endK[None], axis=2).min(1)
        tn = np.arange(nq) / nq
        em[:, 0] = np.minimum(em[:, 0], np.where(tn < 0.3, ds, ds + (tn - 0.3) * 6))
        em[:, self.NB - 1] = np.minimum(em[:, self.NB - 1], np.where(tn > 0.6, de, de + (0.6 - tn) * 6))
        v = med(viterbi(em, self.bins, lam=8.0, end_bonus=2.0)[0], 9)
        if ret_lab:
            dsrt = np.sort(d, axis=1); marg = dsrt[:, 0] / np.clip(dsrt[:, 1], 1e-9, None)
            return v, d.argmin(1), marg
        return v


# ====================== ③ 连续 TCC + DP ======================
class ContinuousValue:
    """TCC frozen-feature 头 + 相似度场 Viterbi-DP + 子bin软期望 → 逐帧连续 value。"""
    def __init__(self, fs: FeatureSpace, eps, n_refs=30, seed=0):
        import torch, torch.nn as nn
        self.torch, self.nn = torch, nn
        self.fs = fs; self.eps = list(eps); self.n_refs = n_refs; self.seed = seed
        self.Gd = {e: fs.emb_ep(e).astype(np.float32) for e in self.eps}
        self.din = self.Gd[self.eps[0]].shape[1]; self.head = None; self.bank = None; self.bankt = None

    def _make_head(self):
        nn = self.nn
        return nn.Sequential(nn.Linear(self.din, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128))

    def train_head(self, steps=1200, batch_eps=8, T=32, lr=1e-3, log=print):
        import sys; sys.path.insert(0, "/vePFS/tim/workspace/recurrence_research/google-research/xirl")
        from xirl.losses import compute_tcc_loss
        torch = self.torch; np.random.seed(self.seed); torch.manual_seed(self.seed)
        self.head = self._make_head(); opt = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-5)
        for step in range(steps):
            bes = list(np.random.choice(self.eps, min(batch_eps, len(self.eps)), replace=False))
            E_, I_, L_ = [], [], []
            for e in bes:
                f = self.Gd[e]; m = len(f); ix = np.sort(np.random.choice(m, size=T, replace=m < T))
                E_.append(self.head(torch.from_numpy(f[ix]))); I_.append(torch.from_numpy(ix).long()); L_.append(m)
            loss = compute_tcc_loss(embs=torch.stack(E_), idxs=torch.stack(I_), seq_lens=torch.tensor(L_),
                stochastic_matching=False, normalize_embeddings=True, loss_type="regression_mse", similarity_type="l2",
                num_cycles=20, cycle_length=2, temperature=0.1, label_smoothing=0.1, variance_lambda=0.001,
                huber_delta=0.1, normalize_indices=True)
            opt.zero_grad(); loss.backward(); opt.step()
            if log and (step + 1) % 400 == 0: log(f"[ContinuousValue] step {step+1} loss {float(loss):.4f}")
        self.head.eval()
        REFS = [e for e in self.eps if True][:self.n_refs]
        self.bank = np.concatenate([self._embed(self.Gd[e]) for e in REFS])
        self.bankt = np.concatenate([np.arange(len(self.Gd[e])) / max(1, len(self.Gd[e]) - 1) for e in REFS])
        return self

    def _embed(self, g796):
        torch = self.torch
        with torch.no_grad():
            z = self.head(torch.from_numpy(np.ascontiguousarray(g796, dtype=np.float32))).numpy()
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

    def value(self, a, r, s, NB=201, lam=0.2, soft_w=8, soft_temp=0.03, smooth=9, exclude_ep=None):
        assert self.head is not None, "先 train_head()"
        bank, bankt = self.bank, self.bankt
        zq = self._embed(self.fs.emb(a, r, s)); n = len(zq); sim = zq @ bank.T
        bins = np.linspace(0, 1, NB); binid = np.clip((bankt * (NB - 1)).round().astype(int), 0, NB - 1)
        simb = np.full((n, NB), -9.0)
        for b in range(NB):
            c = np.where(binid == b)[0]
            if len(c): simb[:, b] = sim[:, c].max(1)
        rmax = simb.max(1, keepdims=True); rmin = np.where(simb > -8, simb, np.inf).min(1, keepdims=True)
        emit = np.where(simb > -8, (rmax - simb) / (rmax - rmin + 1e-6), 1.0)
        _, path = viterbi(emit, bins, lam=lam, end_bonus=2.0)
        cont = np.zeros(n)
        for i in range(n):
            lo = max(0, path[i] - soft_w); hi = min(NB, path[i] + soft_w + 1); ss = simb[i, lo:hi].copy()
            if (ss <= -8).all(): cont[i] = bins[path[i]]; continue
            ss[ss <= -8] = ss[ss > -8].min(); w = np.exp((ss - ss.max()) / soft_temp); w /= w.sum()
            cont[i] = (w * bins[lo:hi]).sum()
        return med(cont, smooth)
