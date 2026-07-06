#!/usr/bin/env python
"""Can a π0.5 SigLIP GRID decoder recover real frames as well as the DINOv3-H grid decoder
(make_decoder big+GDL, L1 ~0.0206 at 128)? Apples-to-apples: SAME proven make_decoder arch, SAME
128 out-res, per-channel grid norm + L1 + GDL. Isolates decoder capacity from SigLIP's info ceiling.

--enc_res 224 (16x16, matches pi0.5) or 384 (27x27, 3x tokens -> higher ceiling test).
--arch big|xl (make_decoder). Reports val_L1 + sharpness vs DINOv3-H target; dumps sample PNG.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
import cv2  # noqa: E402
from train_lawm_patch import load_index, read_imgs  # noqa: E402
from _siglip_bigvision import SiglipBigVision  # noqa: E402
from crave.decoding.decoder import make_decoder  # noqa: E402

PI05_NPZ = "/vePFS/tim/workspace/openpi_cache/paligemma_weights/pt_224.npz"
PI05_NPZ_GF3 = "/vePFS-North-E/vis_robot/openpi_cache/paligemma_weights/pt_224.npz"


def sharp(u8):
    return cv2.Laplacian(cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc_res", type=int, default=224)              # 224->16x16, 384->27x27
    ap.add_argument("--out_res", type=int, default=128)              # match DINOv3-H comparison
    ap.add_argument("--arch", default="xl", choices=["big", "xl"])
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_pairs", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--gdl_weight", type=float, default=0.5)
    ap.add_argument("--out", default="", type=str)
    ap.add_argument("--pi05_npz", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    dev = args.device
    npz = args.pi05_npz or (PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3)
    tag = f"{args.arch}_enc{args.enc_res}_out{args.out_res}"
    out = Path(args.out) if args.out else REPO / f"lmwm/checkpoints/siglip_decoder/dec_{tag}.pt"
    png = REPO / f"lmwm/outputs/siglip_decoder_{tag}.png"

    E, FR, _ = load_index(args.feature_dir)
    rng = np.random.default_rng(args.seed)
    uniq = np.array(sorted(set(rng.choice(len(E), min(args.n_pairs, len(E)), replace=False).tolist())))
    imgsE, imgsT = read_imgs(args.dataset_root, args.camera, E, FR, uniq, args.enc_res, args.out_res)
    print(f"read {len(uniq)} frames; SigLIP@{args.enc_res} encoding ...", flush=True)
    enc = SiglipBigVision(npz, device=dev)
    grids = enc.encode_grid(imgsE, bs=32); din = grids.shape[1]
    print(f"grid {grids.shape} -> decode {args.arch}@{args.out_res}", flush=True)

    mu = grids.mean((0, 2, 3), keepdims=True); sd = grids.std((0, 2, 3), keepdims=True) + 1e-6
    X = torch.from_numpy(((grids - mu) / sd).astype(np.float32))
    Y = torch.from_numpy(imgsT.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).contiguous()
    n = len(X); n_val = max(64, n // 10); perm = torch.randperm(n)
    val_i, tr_i = perm[:n_val], perm[n_val:]
    dec = make_decoder(din, args.arch).to(dev)
    print(f"decoder params {sum(p.numel() for p in dec.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    bs = args.bs
    for ep in range(args.epochs):
        dec.train(); p = tr_i[torch.randperm(len(tr_i))]
        for b in range(0, len(p), bs):
            bi = p[b:b + bs]; x, y = X[bi].to(dev), Y[bi].to(dev)
            pr = dec(x)
            loss = (pr - y).abs().mean() + 0.5 * ((pr - y) ** 2).mean()
            if args.gdl_weight > 0:
                loss = loss + args.gdl_weight * (
                    ((pr[:, :, :, 1:] - pr[:, :, :, :-1]) - (y[:, :, :, 1:] - y[:, :, :, :-1])).abs().mean()
                    + ((pr[:, :, 1:, :] - pr[:, :, :-1, :]) - (y[:, :, 1:, :] - y[:, :, :-1, :])).abs().mean())
            opt.zero_grad(); loss.backward(); opt.step()
        if ep == 0 or (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            dec.eval()
            with torch.no_grad():
                vp = dec(X[val_i].to(dev)); vl = (vp - Y[val_i].to(dev)).abs().mean().item()
                vimg = np.clip((vp[:64].cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
            print(f"epoch {ep+1}/{args.epochs} val_L1={vl:.4f} sharp={np.mean([sharp(im) for im in vimg]):.1f}", flush=True)

    dec.eval()
    with torch.no_grad():
        vi = val_i[:8]; pr = dec(X[vi].to(dev)).cpu()
    to = lambda t: ((t.permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    P, GT = to(pr), to(Y[vi])
    panel = np.concatenate([np.concatenate([P[i], GT[i]], 1) for i in range(8)], 0)[:, :, ::-1]
    cv2.imwrite(str(png), panel)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dec.state_dict(), "dec": args.arch, "res": args.out_res, "enc_res": args.enc_res,
                "din": int(din), "mu": mu, "sd": sd, "val_L1": vl,
                "input": f"pi05 SigLIP grid @{args.enc_res}"}, out)
    print(f"saved {out} val_L1={vl:.4f} (DINOv3-H grid ref ~0.0206@128); samples -> {png}", flush=True)


if __name__ == "__main__":
    main()
