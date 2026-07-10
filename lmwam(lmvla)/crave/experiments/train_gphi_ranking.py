#!/usr/bin/env python
"""Phase 2 骨架:g_φ(s)=sigmoid(head(DINOv3-H)) + margin-ranking,监督用 CRAVE-value 序。

**冻结档**(本脚本):直接用缓存的 DINOv3-H 1280-D 特征(= frozen backbone),只训 head。
- 监督 y = CRAVE 标签(anchor / viterbi,3Hz 对齐到 cache 帧),**rank 序**(非 MSE 值,非 raw-time)。
- 损失 = margin ranking,margin ∝ |Δy|(变点/大间距对被强拉开);y 语义可比 → cross-ep 对随便采。
- 判据(D2 前 sanity):冻结档 g_φ 应 **match CRAVE 曲线**(per-ep corr↑、mono↑);match 不上=损失/采样 bug。
放开档(解冻末几 block + DINO-anchor 正则)= TODO,D2 决定是否上(见 plan)。

Run: REPO=... PYTHONPATH=crave/src CUDA_VISIBLE_DEVICES=0 \
  /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/train_gphi_ranking.py --label viterbi --epochs 40
输出: crave/checkpoints/gphi/{label}/head.pt + crave/docs/visualization/ae_distill/gphi_{label}_sanity.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from crave.render import setup_mpl

plt = setup_mpl()
REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
FEAT = REPO / "temp/crave_full_dinov3h"
LAB = REPO / "temp/crave_ae_labels"
CKPT = REPO / "crave/checkpoints/gphi"
VIZ = REPO / "crave/docs/visualization/ae_distill"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class Head(nn.Module):
    def __init__(self, din=1280, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def load_data(label):
    z = np.load(FEAT / "index.npz")
    E, FR, n = z["E"].astype(np.int64), z["FR"].astype(np.int64), int(z["n"])
    feat = np.zeros((n, 1280), np.float16); valid = np.zeros(n, bool)
    for f in sorted(glob.glob(str(FEAT / "shard_*.npz"))):
        s = np.load(f); feat[s["gidx"]] = s["feat"]; valid[s["gidx"]] = s["valid"]
    feat = l2(feat.astype(np.float32))
    F, Y, EP = [], [], []
    labdir = LAB / label
    for e in sorted(set(E[valid].tolist())):
        lf = labdir / f"ep{e}.npy"
        if not lf.exists():
            continue
        lab = np.load(lf)                                   # native 30Hz CRAVE value
        m = np.where((E == e) & valid)[0]
        o = np.argsort(FR[m]); gi = m[o]; fr = np.clip(FR[gi], 0, len(lab) - 1)
        F.append(feat[gi]); Y.append(lab[fr].astype(np.float32)); EP.append(np.full(len(gi), e))
    return np.concatenate(F), np.concatenate(Y), np.concatenate(EP)


def rank_loss(g, y, margin_c=1.0, dy_min=0.03):
    """batch 内随机配对的 margin ranking;margin ∝ |Δy|(变点加权的天然形式)。"""
    b = len(g); perm = torch.randperm(b, device=g.device)
    gi, gj, yi, yj = g, g[perm], y, y[perm]
    dy = yi - yj; msk = dy.abs() > dy_min                   # 跳过近等对(无信号)
    if msk.sum() == 0:
        return g.sum() * 0.0
    m = margin_c * dy[msk].abs()
    s = torch.sign(dy[msk])
    return torch.relu(m - (gi[msk] - gj[msk]) * s).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["anchor", "viterbi"], default="viterbi")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--margin_c", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--smooth", type=float, default=5.0, help="TV 时序平滑权重")
    a = ap.parse_args()
    print(f"[gphi] label={a.label} dev={DEV}", flush=True)

    F, Y, EP = load_data(a.label)
    eps = np.array(sorted(set(EP.tolist())))
    rng = np.random.RandomState(0); rng.shuffle(eps)
    n_val = int(len(eps) * a.val_frac); val_eps = set(eps[:n_val].tolist())
    tr = ~np.isin(EP, list(val_eps)); va = ~tr
    Ft = torch.from_numpy(F[tr]).to(DEV); Yt = torch.from_numpy(Y[tr]).to(DEV)
    Fv = torch.from_numpy(F[va]).to(DEV); Yv = torch.from_numpy(Y[va]).to(DEV)
    print(f"[gphi] train {tr.sum()} / val {va.sum()} frames; {len(eps)-n_val}/{n_val} eps", flush=True)

    # 时序平滑项:同 ep 相邻帧(ranking 只约束序 → 需显式平滑, 否则 g_φ 逐帧抖 → advantage 噪声)
    EPt = EP[tr]
    cons = torch.from_numpy(np.where(EPt[:-1] == EPt[1:])[0]).to(DEV)          # Ft 中"下一帧同 ep"的下标
    print(f"[gphi] {len(cons)} consecutive-frame pairs for smoothness (λ_s={a.smooth})", flush=True)

    net = Head(F.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    nt = len(Ft)
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(nt, device=DEV); tot = 0.0; nb = 0
        for i in range(0, nt, a.bs):
            idx = perm[i:i + a.bs]
            lr_ = rank_loss(net(Ft[idx]), Yt[idx], a.margin_c)
            cb = cons[torch.randint(len(cons), (a.bs,), device=DEV)]           # 采一批相邻对
            dg = net(Ft[cb + 1]) - net(Ft[cb])
            l_sm = (dg ** 2).mean()                                            # TV 平滑(不回归 Δy, 免重现平台 A≡0)
            loss = lr_ + a.smooth * l_sm
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                gv = net(Fv)
                # val pairwise ranking accuracy
                p = torch.randperm(len(gv), device=DEV)
                dy = Yv - Yv[p]; msk = dy.abs() > 0.05
                acc = float(((gv - gv[p])[msk] * torch.sign(dy[msk]) > 0).float().mean()) if msk.sum() else float("nan")
            print(f"  ep{ep+1} loss={tot/nb:.4f} val_rank_acc={acc:.3f}", flush=True)

    (CKPT / a.label).mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "din": F.shape[1], "label": a.label}, CKPT / a.label / "head.pt")

    # ---- sanity: g_φ vs CRAVE 曲线(held-out ep)+ per-ep corr ----
    net.eval()
    from scipy.stats import pearsonr
    corrs = []
    for e in list(val_eps):
        m = EP == e
        with torch.no_grad():
            g = net(torch.from_numpy(F[m]).to(DEV)).cpu().numpy()
        y = Y[m]
        if y.std() > 1e-3:
            corrs.append(pearsonr(g, y)[0])
    corrs = np.array(corrs)
    show = list(val_eps)[:6]
    fig, axs = plt.subplots(2, 3, figsize=(16, 8)); axs = axs.ravel()
    for k, e in enumerate(show):
        m = EP == e
        with torch.no_grad():
            g = net(torch.from_numpy(F[m]).to(DEV)).cpu().numpy()
        y = Y[m]; x = np.linspace(0, 1, len(y))
        axs[k].plot(x, y, color="#2ca02c", lw=2, label=f"CRAVE-{a.label}(监督)")
        axs[k].plot(x, g, color="#1f77b4", lw=1.5, label="g_φ(冻结档)")
        c = pearsonr(g, y)[0] if y.std() > 1e-3 else float("nan")
        mono = float((np.diff(g) >= -1e-6).mean())
        axs[k].set_title(f"ep{e} corr={c:.2f} mono={mono:.2f}", fontsize=10)
        axs[k].set_ylim(-.03, 1.03); axs[k].grid(alpha=.25)
        if k == 0:
            axs[k].legend(fontsize=9)
    fig.suptitle(f"Phase2 冻结档 sanity:g_φ 是否 match CRAVE-{a.label} 曲线 · held-out · "
                 f"per-ep corr 均值 {np.nanmean(corrs):.3f}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    VIZ.mkdir(parents=True, exist_ok=True)
    fig.savefig(VIZ / f"gphi_{a.label}_sanity.png", dpi=120)
    (CKPT / a.label / "summary.json").write_text(json.dumps(
        {"label": a.label, "val_percorr_mean": float(np.nanmean(corrs)),
         "val_percorr_median": float(np.nanmedian(corrs)), "n_val_eps": len(corrs)}, indent=2))
    print(f"SANITY per-ep corr(g_φ,CRAVE): mean={np.nanmean(corrs):.3f} median={np.nanmedian(corrs):.3f}", flush=True)
    print("SAVED", VIZ / f"gphi_{a.label}_sanity.png", flush=True)


if __name__ == "__main__":
    main()
