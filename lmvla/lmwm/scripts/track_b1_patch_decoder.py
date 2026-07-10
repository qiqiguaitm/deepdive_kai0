#!/usr/bin/env python
"""Track B1: train and PERSIST a DINOv3-H patch-grid decoder (grid -> 128x128 RGB).

train_patch_decoder.py only returned a closure; here we save the module state +
normalization (mu/sd) so the decoder can be reloaded AND back-propped through for
decode-space grid prediction (Track B2). Reports held-out self-reconstruction L1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from train_patch_decoder import load_index, read_frames  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from crave.decoding.decoder import make_decoder, P as GRID_P  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=16000)
    ap.add_argument("--n_val", type=int, default=800)
    ap.add_argument("--enc_res", type=int, default=256)
    ap.add_argument("--tgt_res", type=int, default=128)
    ap.add_argument("--dec", default="small")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--out", default="lmwm/checkpoints/patch_decoder/patch_dec.pt", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    E, FR = load_index(args.feature_dir)
    rng = np.random.default_rng(2026)
    eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    is_val = np.array([e in val_eps for e in E])
    tr_g = rng.choice(np.where(~is_val)[0], min(args.n_train, int((~is_val).sum())), replace=False)
    va_g = rng.choice(np.where(is_val)[0], min(args.n_val, int(is_val.sum())), replace=False)

    print(f"reading {len(tr_g)}+{len(va_g)} frames ...", flush=True)
    tr_enc, tr_tgt = read_frames(args.dataset_root, args.camera, E, FR, tr_g, args.enc_res, args.tgt_res)
    va_enc, va_tgt = read_frames(args.dataset_root, args.camera, E, FR, va_g, args.enc_res, args.tgt_res)
    enc = load_encoder("dinov3-h", device=str(dev))
    print("encoding grids ...", flush=True)
    tr_grid = enc.encode_grid(tr_enc).astype(np.float32)   # (N,1280,16,16)
    va_grid = enc.encode_grid(va_enc).astype(np.float32)
    din = tr_grid.shape[1]

    mu = tr_grid.mean(axis=(0, 2, 3), dtype=np.float32)
    sd = tr_grid.std(axis=(0, 2, 3)).astype(np.float32) + 1e-4
    muT = torch.from_numpy(mu).view(1, din, 1, 1).to(dev); sdT = torch.from_numpy(sd).view(1, din, 1, 1).to(dev)
    Y = torch.from_numpy(tr_tgt.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).contiguous().to(dev)
    Gg = torch.from_numpy(tr_grid).to(dev)
    D = make_decoder(din, args.dec).to(dev)
    opt = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    n = len(tr_grid); bs = 64
    print(f"training patch decoder (dec={args.dec}, din={din}) ...", flush=True)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, bs):
            bi = perm[b:b + bs]
            pred = D((Gg[bi] - muT) / sdT)
            loss = (pred - Y[bi]).abs().mean() + 0.5 * ((pred - Y[bi]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    D.eval()

    with torch.no_grad():
        rec = []
        for b in range(0, len(va_grid), 256):
            x = (torch.from_numpy(va_grid[b:b + 256]).to(dev) - muT) / sdT
            rec.append(D(x).cpu().numpy())
        rec = np.concatenate(rec)
    rec_u8 = np.clip((rec.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
    l1 = float(np.abs(va_tgt.astype(float) - rec_u8.astype(float)).mean())
    def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": D.state_dict(), "mu": mu, "sd": sd, "din": din, "dec": args.dec, "res": args.tgt_res,
                "grid_P": GRID_P, "val_L1_frac": l1 / 255,
                "val_sharp": float(np.mean([sharp(x) for x in rec_u8])),
                "real_sharp": float(np.mean([sharp(x) for x in va_tgt]))}, args.out)
    print(f"saved {args.out} | val L1 frac={l1/255:.4f} (pooled 0.13) | "
          f"sharp {np.mean([sharp(x) for x in rec_u8]):.0f} vs real {np.mean([sharp(x) for x in va_tgt]):.0f}", flush=True)


if __name__ == "__main__":
    main()
