#!/usr/bin/env python
"""Sharp DINOv3-H pooled-latent -> image decoder via adversarial training.

L1/L2 decoders predict the pixel-wise mean -> blur. A GAN samples a *sharp* mode
instead of averaging. Generator = the existing PooledDecoder (so the checkpoint
loads in every existing viz script); add a PatchGAN discriminator + hinge
adversarial loss + L1 (keeps color/structure faithful). This is the fast path to
"plausible-sharp" synthetic subgoal images.

Usage:
    python lmwm/scripts/train_dinov3h_decoder_gan.py \
        --feature_dir temp/crave_full_dinov3h --dataset_root kai0/data/Task_A/kai0_base \
        --n_pairs 24000 --epochs 120 --l1_weight 15 \
        --out lmwm/checkpoints/dinov3h_decoder/dec_gan.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dinov3h_decoder import PooledDecoder, load_features, l2  # noqa: E402


class PatchD(nn.Module):
    """PatchGAN discriminator on 128x128 RGB -> patch real/fake logits."""

    def __init__(self) -> None:
        super().__init__()

        def blk(i, o, norm=True):
            layers = [nn.Conv2d(i, o, 4, 2, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.net = nn.Sequential(
            *blk(3, 64, norm=False), *blk(64, 128), *blk(128, 256),
            nn.Conv2d(256, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def sharp(img_uint8):
    return cv2.Laplacian(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()


def build_dataset(args, E, FR, F):
    rng = np.random.default_rng(args.seed)
    n_pairs = min(args.n_pairs, len(F))
    sel = rng.choice(len(F), n_pairs, replace=False)
    by_ep: dict[int, list[int]] = {}
    for i in sel:
        by_ep.setdefault(int(E[i]), []).append(int(i))
    R = args.res
    imgs = np.zeros((n_pairs, R, R, 3), dtype=np.uint8)
    feats = l2(F[sel].astype(np.float32))
    pos = {int(i): k for k, i in enumerate(sel)}
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    done = 0
    for ep, idxs in by_ep.items():
        cap = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        for gi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(FR[gi]))
            ok, fr = cap.read()
            if ok:
                imgs[pos[gi]] = cv2.resize(fr[:, :, ::-1], (R, R))
            done += 1
        cap.release()
        if done % 6000 < len(idxs):
            print(f"  read {done}/{n_pairs}", flush=True)
    return feats, imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", required=True, type=Path)
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_pairs", type=int, default=24000)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--l1_weight", type=float, default=15.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    E, FR, F = load_features(args.feature_dir)
    feats, imgs = build_dataset(args, E, FR, F)

    Y = torch.from_numpy(imgs.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).contiguous()
    X = torch.from_numpy(feats)
    n = len(X); n_val = max(256, n // 10)
    perm = torch.randperm(n); val_i, tr_i = perm[:n_val], perm[n_val:]

    G = PooledDecoder(din=X.shape[1], res=args.res).to(dev)
    D = PatchD().to(dev)
    optG = torch.optim.AdamW(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bs = 64
    for ep in range(args.epochs):
        G.train(); D.train()
        p = tr_i[torch.randperm(len(tr_i))]
        for b in range(0, len(p), bs):
            bi = p[b:b + bs]
            x, y = X[bi].to(dev), Y[bi].to(dev)
            fake = G(x)
            # D step (hinge)
            optD.zero_grad()
            d_real = D(y); d_fake = D(fake.detach())
            lossD = torch.relu(1 - d_real).mean() + torch.relu(1 + d_fake).mean()
            lossD.backward(); optD.step()
            # G step: adversarial + L1
            optG.zero_grad()
            lossG = -D(fake).mean() + args.l1_weight * (fake - y).abs().mean()
            lossG.backward(); optG.step()
        if ep == 0 or (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            G.eval()
            with torch.no_grad():
                vp = G(X[val_i].to(dev))
                vl1 = (vp - Y[val_i].to(dev)).abs().mean().item()
                vimg = np.clip((vp[:64].cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
            sh = float(np.mean([sharp(im) for im in vimg]))
            print(f"epoch {ep + 1}/{args.epochs}  val_L1={vl1:.4f}  sharp={sh:.1f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": G.state_dict(), "res": args.res, "din": int(X.shape[1]),
                "meta": {"n_pairs": len(X), "epochs": args.epochs, "loss": "hinge-GAN + L1", "l1_weight": args.l1_weight}},
               args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
