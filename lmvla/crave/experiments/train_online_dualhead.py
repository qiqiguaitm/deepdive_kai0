#!/usr/bin/env python
"""在线因果【分布式双头】value 模型 —— CRAVE 折线蒸馏 + AWBC advantage.

融合: CRAVE 折线(零标注平滑 progress) + KAI0 相对值 + RECAP 分布式头 + 因果 GRU.
  主干  : 单向 GRU(严格因果, img128⊕pos14 → h_t)
  头①   : 分布式 progress   —— Kp-bin softmax(two-hot CE), 读出 E[bins] ∈ [0,1]
  头②   : 分布式 advantage  —— Ka-bin softmax over [-1,1], 目标 = 折线 H步前向 Δprogress
          (KAI0 relative_advantage 的因果版: 靠 GRU 学到的"预期未来"直接输出, 推理零未来)
teacher = CRAVE 折线(polyline, 见 gen_polyline_labels); 折线 Δ 平滑, 远优于阶梯 Δ(退化率 42%→26%).
AWBC 用头②输出做 per-ep 分位离散 → positive/negative prompt.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH="<repo>/lmvla/crave/src" \
       python lmvla/crave/experiments/train_online_dualhead.py
"""
import time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fnn
from pathlib import Path
from scipy.stats import spearmanr
from crave.config import resolve_dataset

REPO = Path("/home/tim/workspace/deepdive_kai0"); LAM = 16.0; CSQ = 1000; DEV = "cuda:0"
H3 = 5          # 前向 advantage 窗口 @3Hz(=50帧@30Hz)
KP, KA = 41, 41 # progress bins [0,1], advantage bins [-1,1]
rng = np.random.RandomState(0); torch.manual_seed(0)
PB = torch.linspace(0, 1, KP, device=DEV); AB = torch.linspace(-1, 1, KA, device=DEV)
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

# ---- spec + 双锚 + polyline teacher ----
s = np.load(REPO / "temp/crave_final_v3.npz")
vals = s["vals"]; Ctgt = s["Ctgt"]; pca_m = s["pca_mean"]; pca_c = s["pca_components"]; SMU = s["SMU"]; SSD = s["SSD"]
FEAT = REPO / "temp/crave_d3b_pca128/feats"; alleps = sorted(int(p.stem[2:]) for p in FEAT.glob("ep*.npy"))
cfg = resolve_dataset("kai0_base"); DS = Path(cfg.root); ADV = REPO / "kai0/data/Task_A/kai0_advantage"
zf = np.load(REPO / "temp/crave_full_dinov3h/index.npz"); E_idx, FR_idx = zf["E"], zf["FR"]

def feat3(e):
    f = np.load(FEAT / f"ep{e}.npy").astype(np.float32); fq = l2((l2(f) - pca_m) @ pca_c.T); n = len(fq)
    loc = np.where(E_idx == e)[0]; o = np.argsort(FR_idx[loc]); fr = FR_idx[loc][o][:n]
    st = np.stack(pd.read_parquet(DS / f"data/chunk-{e // CSQ:03d}/episode_{e:06d}.parquet",
                                   columns=["observation.state"])["observation.state"].to_numpy()).astype(np.float32)
    return np.concatenate([fq, l2((st[np.minimum(fr, len(st) - 1)] - SMU) / SSD)], 1), len(st)

an = sorted(rng.choice(alleps, 400, replace=False)); ss = []; ee = []
for e in an: F, _ = feat3(e); ss.append(F[:3].mean(0)); ee.append(F[-3:].mean(0))
sC = l2(np.mean(ss, 0)[None])[0]; eC = l2(np.mean(ee, 0)[None])[0]
Ct2 = np.vstack([Ctgt, sC, eC]).astype(np.float32); Pord = np.concatenate([vals, [0.], [1.]])
bins = np.unique(np.concatenate([[0.], Pord, [1.]])); nb = len(bins)
cb = [int(np.searchsorted(bins, v)) for v in Pord]; pen = LAM * np.abs(bins[:, None] - bins[None])

def viterbi(Fq):
    de = np.linalg.norm(Fq[:, None] - Ct2[None], axis=2); em = np.full((len(Fq), nb), 1e3)
    for ti in range(len(Pord)): em[:, cb[ti]] = np.minimum(em[:, cb[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(Fq), nb), int)
    for j in range(1, len(Fq)):
        tr = cost[None, :] + pen; kk = tr.argmin(1); cost = em[j] + tr[np.arange(nb), kk]; BP[j] = kk
    si = nb - 1; path = np.zeros(len(Fq), int); path[-1] = si
    for j in range(len(Fq) - 2, -1, -1): si = BP[j + 1][si]; path[j] = si
    return bins[path]

def polyline(Fq):
    step = viterbi(Fq); N = len(Fq); segs = []; a = 0
    for t in range(1, N):
        if step[t] != step[t - 1]: segs.append((a, t - 1, step[t - 1])); a = t
    segs.append((a, N - 1, step[-1])); reps = []
    for (i0, i1, val) in segs:
        cand = [ti for ti in range(len(Pord)) if abs(Pord[ti] - val) < 1e-9]; fr = np.arange(i0, i1 + 1); bd = 1e18; bf = i0
        for ti in cand:
            d = np.linalg.norm(Fq[fr] - Ct2[ti], axis=1); k = int(d.argmin())
            if d[k] < bd: bd = d[k]; bf = fr[k]
        reps.append((bf, float(val)))
    if reps[0][0] != 0: reps = [(0, float(step[0]))] + reps
    if reps[-1][0] != N - 1: reps = reps + [(N - 1, float(step[-1]))]
    rf = np.array([r[0] for r in reps]); rv = np.array([r[1] for r in reps]); keep = np.concatenate([[True], np.diff(rf) > 0])
    return np.interp(np.arange(N), rf[keep], rv[keep]).astype(np.float32)

def fwd_adv(p, H):
    q = np.empty_like(p); q[:-H] = p[H:] - p[:-H]; q[-H:] = p[-1] - p[-H:]; return np.clip(q, -1, 1)

print("加载特征 + 折线 teacher + advantage 目标...", flush=True); t0 = time.time()
use = sorted(rng.choice(alleps, 2300, replace=False)); DATA = []
for e in use:
    F, N = feat3(e); poly = polyline(F); DATA.append((e, F.astype(np.float32), poly, fwd_adv(poly, H3).astype(np.float32)))
print(f"  {len(DATA)} eps ({time.time()-t0:.0f}s)", flush=True)
idx = np.arange(len(DATA)); rng.shuffle(idx); tr_i = idx[:2000]; ev_i = idx[2000:]


def two_hot(y, B):   # y:(...,) → (...,K) two-hot over uniform bins B
    K = len(B); lo, hi = B[0], B[-1]; y = y.clamp(lo, hi)
    pos = (y - lo) / (hi - lo) * (K - 1); i = pos.floor().long().clamp(0, K - 2); w = (pos - i.float())
    t = torch.zeros(*y.shape, K, device=y.device); t.scatter_(-1, i.unsqueeze(-1), (1 - w).unsqueeze(-1))
    t.scatter_(-1, (i + 1).unsqueeze(-1), w.unsqueeze(-1)); return t


class DualHeadGRU(nn.Module):
    def __init__(self, d=142, h=256, L=2):
        super().__init__(); self.g = nn.GRU(d, h, L, batch_first=True)
        self.hp = nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Linear(128, KP))
        self.ha = nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Linear(128, KA))
    def forward(self, x, lens=None):
        if lens is not None:
            p = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
            o, _ = self.g(p); o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        else:
            o, _ = self.g(x)
        return self.hp(o), self.ha(o)


def batches(ii, bs=32):
    ii = sorted(list(ii), key=lambda i: len(DATA[i][1]))
    for k in range(0, len(ii), bs):
        gp = ii[k:k + bs]; L = max(len(DATA[i][1]) for i in gp); B = len(gp)
        X = np.zeros((B, L, 142), np.float32); P = np.zeros((B, L), np.float32); A = np.zeros((B, L), np.float32)
        M = np.zeros((B, L), np.float32); ln = np.zeros(B, int)
        for b, i in enumerate(gp):
            n = len(DATA[i][1]); X[b, :n] = DATA[i][1]; P[b, :n] = DATA[i][2]; A[b, :n] = DATA[i][3]; M[b, :n] = 1; ln[b] = n
        yield (torch.tensor(X, device=DEV), torch.tensor(P, device=DEV), torch.tensor(A, device=DEV),
               torch.tensor(M, device=DEV), torch.tensor(ln))


net = DualHeadGRU().to(DEV)
print(f"参数量 {sum(p.numel() for p in net.parameters()):,}", flush=True)
opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)
GT = {DATA[i][0]: pd.read_parquet(ADV / f"data/chunk-{DATA[i][0] // CSQ:03d}/episode_{DATA[i][0]:06d}.parquet",
                                   columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy() for i in ev_i}
def cc(a, b): return np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else np.nan


@torch.no_grad()
def evaluate():
    net.eval(); pg = []; padv_p = []; padv_s = []; top = []; diff_p = []
    for i in ev_i:
        e, F, poly, _ = DATA[i]; lp, la = net(torch.tensor(F[None], device=DEV))
        prog = (Fnn.softmax(lp[0], -1) @ PB).cpu().numpy(); adv = (Fnn.softmax(la[0], -1) @ AB).cpu().numpy()
        g = GT[e]; n = len(F)
        prog30 = np.interp(np.linspace(0, 1, len(g)), np.linspace(0, 1, n), prog)
        adv30 = np.interp(np.linspace(0, 1, len(g)), np.linspace(0, 1, n), adv)
        gadv = fwd_adv(g.astype(np.float32), 50)
        pg.append(cc(prog30, g))                       # progress vs 监督GT
        padv_p.append(cc(adv30, gadv))                 # advantage Pearson vs GT-adv
        if adv30.std() > 1e-6: padv_s.append(spearmanr(adv30, gadv).correlation)  # 排序
        # AWBC top-30% 二值一致(per-ep 70 分位阈)
        pt = adv30 >= np.quantile(adv30, 0.7); gt = gadv >= np.quantile(gadv, 0.7)
        top.append((pt == gt).mean())
        # 对照: 头②专用 advantage vs 直接差分 progress 头
        dadv = fwd_adv(prog30, 50); diff_p.append(cc(dadv, gadv))
    return (np.nanmean(pg), np.nanmean(padv_p), np.nanmean(padv_s), np.mean(top), np.nanmean(diff_p))


print("训练分布式双头 GRU (50 ep)...", flush=True)
for ep in range(50):
    net.train(); tl = 0; nb_ = 0
    for X, P, A, M, ln in batches(tr_i):
        lp, la = net(X, ln)
        tp = two_hot(P, PB); ta = two_hot(A, AB)
        lossp = -(tp * Fnn.log_softmax(lp, -1)).sum(-1); lossa = -(ta * Fnn.log_softmax(la, -1)).sum(-1)
        loss = ((lossp + lossa) * M).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); nb_ += 1
    sch.step()
    if (ep + 1) % 10 == 0 or ep == 0:
        pg, pa, ps, tp, dp = evaluate()
        print(f"  ep{ep+1:>2} loss={tl/nb_:.3f} | progress corr={pg:.3f} | adv Pearson={pa:.3f} "
              f"Spearman={ps:.3f} top30%一致={tp:.3f} | (差分progress头 adv={dp:.3f})", flush=True)
torch.save(net.state_dict(), REPO / "temp/crave_online_dualhead.pt")
print("DONE -> temp/crave_online_dualhead.pt", flush=True)
