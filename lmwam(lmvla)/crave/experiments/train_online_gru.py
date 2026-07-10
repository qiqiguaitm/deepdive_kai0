#!/usr/bin/env python
"""在线 CRAVE 读出 —— 因果 GRU 蒸馏离线双锚 Viterbi(追平离线 · 零未来推理).

思路: 离线的优势 = 知道未来; 用【从训练数据学到的对未来的预期】替代之。
  teacher = 离线双锚 Viterbi value(gen_anchored_labels 同款, 3Hz)
  student = 单向 GRU(结构上严格因果, 第 t 帧只依赖 h_t=累积自 1..t, 零未来)
  蒸馏    = MSE 回归 teacher

模型 I/O:
  输入  x: (B, T, 142)  每帧 = concat[ img PCA128(L2), proprio位置14(标准化+L2) ]
  输出  y: (B, T) ∈ [0,1]  每帧 progress value
  参数量: 734,977 (~0.735M);  GRU(142→256,2层)=701,952 + head(256→128→1)=33,025;  ckpt 2.94MB

实测(留出 300ep, 严格未见, 零未来):
  corr vs 离线 teacher = 0.97   corr vs 监督 stage_gt = 0.955(≈离线本身 0.943)   末值 0.986
  对比: 对称forward-DP 0.83 / 非对称forward-DP 0.86 / 【GRU 0.955】/ 离线基准 0.943

真机在线推理(严格因果, 逐帧):
  h = None
  for each frame:
      x_t = concat[img_pca128_l2, pos14_std_l2]          # (1,1,142)
      o, h = net.g(x_t, h)                                 # 复用隐状态, O(1)/帧
      p_t = sigmoid(net.head(o)).item()                    # 当前 progress, 零未来

Run(训练): CUDA_VISIBLE_DEVICES=0 PYTHONPATH=crave/src:lmwm/src:crave/experiments \
             python crave/experiments/train_online_gru.py
输出: temp/crave_online_gru.pt
"""
import time, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from crave.config import resolve_dataset

REPO = Path("/home/tim/workspace/deepdive_kai0"); CSQ = 1000; DEV = "cuda:0"
rng = np.random.RandomState(0); torch.manual_seed(0)
def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

# ---- milestone spec + 双锚(teacher 用) ----
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
cb = [int(np.searchsorted(bins, v)) for v in Pord]; pen = 16. * np.abs(bins[:, None] - bins[None])

def offline_teacher(Fq):   # 离线双锚 Viterbi(强制首末帧)
    de = np.linalg.norm(Fq[:, None] - Ct2[None], axis=2); em = np.full((len(Fq), nb), 1e3)
    for ti in range(len(Pord)): em[:, cb[ti]] = np.minimum(em[:, cb[ti]], de[:, ti])
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(Fq), nb), int)
    for j in range(1, len(Fq)):
        tr = cost[None, :] + pen; kk = tr.argmin(1); cost = em[j] + tr[np.arange(nb), kk]; BP[j] = kk
    si = nb - 1; path = np.zeros(len(Fq), int); path[-1] = si
    for j in range(len(Fq) - 2, -1, -1): si = BP[j + 1][si]; path[j] = si
    return bins[path].astype(np.float32)

# ---- 数据 ----
print("加载特征 + teacher 标签...", flush=True); t0 = time.time()
use = sorted(rng.choice(alleps, 2300, replace=False)); DATA = []
for e in use:
    F, N = feat3(e); DATA.append((e, F.astype(np.float32), offline_teacher(F)))
print(f"  {len(DATA)} eps ({time.time()-t0:.0f}s)", flush=True)
idx = np.arange(len(DATA)); rng.shuffle(idx); tr_i = idx[:2000]; ev_i = idx[2000:]


class OnlineGRU(nn.Module):
    """单向 GRU(严格因果)→ 逐帧 progress ∈ [0,1]."""
    def __init__(self, d=142, h=256, L=2):
        super().__init__()
        self.g = nn.GRU(d, h, L, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(self, x, lens=None):
        if lens is not None:
            p = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
            o, _ = self.g(p); o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        else:
            o, _ = self.g(x)
        return torch.sigmoid(self.head(o)).squeeze(-1)


def batches(ii, bs=32):
    ii = sorted(list(ii), key=lambda i: len(DATA[i][1]))   # 桶排序减 padding
    for k in range(0, len(ii), bs):
        gp = ii[k:k + bs]; L = max(len(DATA[i][1]) for i in gp); B = len(gp)
        X = np.zeros((B, L, 142), np.float32); Y = np.zeros((B, L), np.float32); M = np.zeros((B, L), np.float32); ln = np.zeros(B, int)
        for b, i in enumerate(gp):
            n = len(DATA[i][1]); X[b, :n] = DATA[i][1]; Y[b, :n] = DATA[i][2]; M[b, :n] = 1; ln[b] = n
        yield (torch.tensor(X, device=DEV), torch.tensor(Y, device=DEV), torch.tensor(M, device=DEV), torch.tensor(ln))


net = OnlineGRU().to(DEV)
print(f"参数量 {sum(p.numel() for p in net.parameters()):,}", flush=True)
opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
GT = {DATA[i][0]: pd.read_parquet(ADV / f"data/chunk-{DATA[i][0] // CSQ:03d}/episode_{DATA[i][0]:06d}.parquet",
                                   columns=["stage_progress_gt"])["stage_progress_gt"].to_numpy() for i in ev_i}
def cc(a, b): return np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else np.nan


@torch.no_grad()
def evaluate():
    net.eval(); cto = []; ctg = []; en = []
    for i in ev_i:
        e, F, tea = DATA[i]; pr = net(torch.tensor(F[None], device=DEV))[0].cpu().numpy()
        cto.append(cc(pr, tea)); g = GT[e]
        pr30 = np.interp(np.linspace(0, 1, len(g)), np.linspace(0, 1, len(pr)), pr); ctg.append(cc(pr30, g)); en.append(pr30[-5:].mean())
    return np.nanmean(cto), np.nanmean(ctg), np.mean(en)


print("训练因果 GRU (60 ep)...", flush=True)
for ep in range(60):
    net.train(); tl = 0; nb_ = 0
    for X, Y, M, ln in batches(tr_i):
        pr = net(X, ln); loss = (((pr - Y) ** 2) * M).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); nb_ += 1
    sch.step()
    if (ep + 1) % 10 == 0 or ep == 0:
        co, cg, en = evaluate()
        print(f"  ep{ep+1:>2} loss={tl/nb_:.4f} | corr vs离线={co:.3f} corr vsGT={cg:.3f} 末值={en:.3f}", flush=True)
torch.save(net.state_dict(), REPO / "temp/crave_online_gru.pt")
print("DONE -> temp/crave_online_gru.pt", flush=True)
