#!/usr/bin/env python
"""重生成网站两张"软合成质心"配图为 medoid(锐)版,直接落到 crave_report/assets/。

① recurrence_milestone.png (§2.1):高覆盖 milestone 的 **medoid(左, 大)** + 6 条不同 episode 的最近真实帧(右)
   —— 替换旧"簇中心解码图(软)"。
② milestone_repr_decode.png (§4.1 ②):同 8 个 milestone 的 **合成质心(软) vs medoid(锐)** 对比
   —— 替换旧"簇中心合成解码(软)",顺带说明为何用 medoid。
Run: REPO=/home/tim/workspace/deepdive_kai0 PYTHONPATH=crave/src:lmwm/src:lmwm/scripts \
  CUDA_VISIBLE_DEVICES=0 /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/regen_website_medoid.py
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from crave.render import setup_mpl
from lmwm.retrieval_decoder import LatentRetrievalDecoder
from train_dinov3h_decoder import PooledDecoder

plt = setup_mpl()
REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
FEAT = REPO / "temp/crave_full_dinov3h"
DATA = REPO / "kai0/data/Task_A/kai0_base"
CKPT = REPO / "lmwm/checkpoints"
ASSETS = REPO / "web/showcase/reports/crave_report/assets"


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    rdec = LatentRetrievalDecoder(feature_dir=FEAT, dataset_root=DATA, device="cuda", res=None)
    mz = np.load(FEAT / "milestones_uniform_dinov3h.npz")
    C = l2(mz["C"].astype(np.float32)); pord = mz["Pord"].astype(np.float32)
    K = len(C); F = rdec.feat; E = rdec.E; ne = len(set(E.tolist()))

    # per-milestone coverage (distinct episodes) via nearest-milestone assignment
    assign = np.empty(len(F), np.int64)
    for i in range(0, len(F), 40000):
        assign[i:i + 40000] = (F[i:i + 40000] @ C.T).argmax(1)
    cov = np.array([len(set(E[assign == m].tolist())) / ne if (assign == m).any() else 0 for m in range(K)])

    # ---- ① recurrence: high-coverage mid-progress milestone -> medoid + 6 cross-ep frames ----
    cand = [m for m in range(K) if 0.2 <= pord[m] <= 0.8]
    mstar = max(cand, key=lambda m: cov[m])
    idx, cos = rdec.retrieve(C[mstar:mstar + 1], topk=80)
    med_g = int(idx[0, 0]); med_ep = int(E[med_g])
    cross, eps = [], {med_ep}
    for j in range(idx.shape[1]):
        gi = int(idx[0, j]); ep = int(E[gi])
        if ep not in eps:
            cross.append((gi, float(cos[0, j]))); eps.add(ep)
        if len(cross) == 6:
            break
    print(f"① milestone m{mstar}: 覆盖 {cov[mstar]:.0%} episodes, 进度≈{pord[mstar]:.2f}; medoid ep{med_ep}", flush=True)

    fig = plt.figure(figsize=(15, 5.4))
    gs = fig.add_gridspec(2, 5, wspace=0.06, hspace=0.12)
    axm = fig.add_subplot(gs[:, :2]); axm.imshow(cv2.resize(rdec.frame(med_g), (256, 256)))
    axm.set_xticks([]); axm.set_yticks([])
    axm.set_title(f"milestone 代表 = medoid(检索最近真实帧)\n进度≈{pord[mstar]:.2f} · 覆盖 {cov[mstar]:.0%} episodes", fontsize=11)
    for i, (gi, cs) in enumerate(cross):
        ax = fig.add_subplot(gs[i // 3, 2 + i % 3]); ax.imshow(cv2.resize(rdec.frame(gi), (160, 160)))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(f"ep{int(E[gi])}  cos{cs:.2f}", fontsize=9)
    fig.suptitle("反复出现 = 任务必经结构:同一 milestone(左 medoid)在 6 条不同 episode 反复出现(右,均锐利真实)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ASSETS / "recurrence_milestone.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("SAVED", ASSETS / "recurrence_milestone.png", flush=True)

    # ---- ② synth(soft) vs medoid(sharp) for 8 milestones ----
    d = torch.load(CKPT / "dinov3h_decoder/dec.pt", map_location="cuda", weights_only=False)
    G = PooledDecoder(din=d["din"], res=d["res"]).cuda().eval(); G.load_state_dict(d["model"])
    with torch.no_grad():
        synth = np.clip((G(torch.from_numpy(C).cuda()).cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
    order = np.argsort(pord)
    pick = [order[i] for i in np.linspace(0, K - 1, 8).round().astype(int)]
    med_g8 = [int(rdec.retrieve(C[m:m + 1], topk=1)[0][0, 0]) for m in pick]

    fig, axs = plt.subplots(2, 8, figsize=(18, 5.0))
    for c, (m, mg) in enumerate(zip(pick, med_g8)):
        axs[0, c].imshow(synth[m]); axs[0, c].set_title(f"进度 {pord[m]:.2f}", fontsize=10)
        axs[1, c].imshow(cv2.resize(rdec.frame(mg), (128, 128)))
        for r in range(2):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    axs[0, 0].set_ylabel("旧:合成质心\n(解码平均→软)", fontsize=11, rotation=0, ha="right", va="center")
    axs[1, 0].set_ylabel("新:medoid\n(检索最近真实帧→锐)", fontsize=11, rotation=0, ha="right", va="center")
    fig.suptitle("簇中心表示升级:合成质心(上,平均 ill-posed→软)→ medoid 检索(下,锐利真实)—— 簇中心可解释、代表更清晰",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(ASSETS / "milestone_repr_decode.png", dpi=120, bbox_inches="tight")
    print("SAVED", ASSETS / "milestone_repr_decode.png", flush=True)


if __name__ == "__main__":
    main()
