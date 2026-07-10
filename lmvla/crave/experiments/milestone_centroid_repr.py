#!/usr/bin/env python
"""簇中心表示:合成质心解码(软,换编码器救不动)vs medoid 检索(锐,推荐)。

Q:换更好的编码器能让"簇中心解码"更清晰吗?A:不能——簇中心=成员 latent 的**平均**,
解码平均对可形变布料 ill-posed(必软),与编码器无关。清晰的簇中心图靠 **medoid=检索离
中心最近的真实帧**(锐、真实、编码器无关)。

产两图:
  ① milestone_synth_vs_medoid.png —— 8 个 milestone:合成质心(pooled 解码,软)vs medoid(检索,锐)
  ② milestone_medoid_gallery.png  —— 全部 milestone 的 medoid 词表(按进度排序),升级版网站/文档配图
Run: REPO=/home/tim/workspace/deepdive_kai0 PYTHONPATH=crave/src:lmwm/src:lmwm/scripts \
  /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/milestone_centroid_repr.py
输出: crave/docs/visualization/decoder_benchmark/{milestone_synth_vs_medoid,milestone_medoid_gallery}.png
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
OUT = REPO / "crave/docs/visualization/decoder_benchmark"
DEV = "cuda"


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def sharp(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def main():
    rdec = LatentRetrievalDecoder(feature_dir=FEAT, dataset_root=DATA, device=DEV, res=None)
    mz = np.load(FEAT / "milestones_uniform_dinov3h.npz")
    C = l2(mz["C"].astype(np.float32))                       # (37,1280) 簇中心
    pord = mz["Pord"].astype(np.float32)
    order = np.argsort(pord)
    K = len(C)

    # medoid = 检索离每个中心最近的真实帧
    med_g = np.array([int(rdec.retrieve(C[k:k + 1], topk=1)[0][0, 0]) for k in range(K)])
    med_img = [rdec.frame(int(g)) for g in med_g]

    # 合成质心 = pooled 解码器解码中心向量(软)
    d = torch.load(CKPT / "dinov3h_decoder/dec.pt", map_location=DEV, weights_only=False)
    G = PooledDecoder(din=d["din"], res=d["res"]).to(DEV).eval(); G.load_state_dict(d["model"])
    with torch.no_grad():
        synth = G(torch.from_numpy(C).to(DEV)).cpu().numpy().transpose(0, 2, 3, 1)
    synth = np.clip((synth + 1) * 127.5, 0, 255).astype(np.uint8)

    med_sharp = float(np.mean([sharp(cv2.resize(x, (128, 128))) for x in med_img]))  # same 128px as synth
    syn_sharp = float(np.mean([sharp(x) for x in synth]))
    print(f"medoid sharp={med_sharp:.0f} | synth-centroid sharp={syn_sharp:.0f}", flush=True)

    # ---- ① 合成 vs medoid(8 个 milestone 跨进度)----
    pick = [order[i] for i in np.linspace(0, K - 1, 8).round().astype(int)]
    fig, axs = plt.subplots(2, 8, figsize=(18, 5.0))
    for c, k in enumerate(pick):
        axs[0, c].imshow(synth[k]); axs[0, c].set_title(f"P={pord[k]:.2f}", fontsize=10)
        axs[1, c].imshow(cv2.resize(med_img[k], (128, 128)))
        for r in range(2):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    axs[0, 0].set_ylabel("合成质心\n(pooled 解码)", fontsize=11, rotation=0, ha="right", va="center")
    axs[1, 0].set_ylabel("medoid\n(检索最近真实帧)", fontsize=11, rotation=0, ha="right", va="center")
    fig.suptitle(f"簇中心解码:合成质心(软,锐度 {syn_sharp:.0f},换编码器/加规模救不动 ← 平均 ill-posed) "
                 f"vs medoid 检索(锐,锐度 {med_sharp:.0f}≈真实)—— 推荐 medoid", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "milestone_synth_vs_medoid.png", dpi=120, bbox_inches="tight")
    print("SAVED", OUT / "milestone_synth_vs_medoid.png", flush=True)
    plt.close(fig)

    # ---- ② 全 milestone medoid 词表(按进度排序)----
    ncol = 8
    nrow = int(np.ceil(K / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.1 * nrow))
    axs = axs.ravel()
    for i, k in enumerate(order):
        axs[i].imshow(cv2.resize(med_img[k], (160, 160))); axs[i].set_xticks([]); axs[i].set_yticks([])
        axs[i].set_title(f"m{k}  P={pord[k]:.2f}", fontsize=9)
    for j in range(K, len(axs)):
        axs[j].axis("off")
    fig.suptitle(f"CRAVE milestone 词表 · {K} 个 milestone 的 medoid(检索最近真实帧)· 按进度排序 · DINOv3-H · kai0_base",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "milestone_medoid_gallery.png", dpi=115, bbox_inches="tight")
    print("SAVED", OUT / "milestone_medoid_gallery.png", flush=True)


if __name__ == "__main__":
    main()
