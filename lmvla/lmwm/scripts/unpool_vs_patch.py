#!/usr/bin/env python
"""Compare the two paths to a faithfully-decodable LMWM prediction:

  Path 1 (治本): predict the patch-grid latent directly -> decode. Representation
    ceiling = decode(true patch grid) fidelity (~2.7% self-recon).
  Path 2 (轻量): keep predicting pooled, learn an un-pool map pooled -> patch-grid
    -> decode. Ceiling = decode(unpool(pooled)) fidelity. The GAP vs path 1 is the
    un-pool information loss (does the pooled vector retain the patch grid?).

Both measured at the SELF-reconstruction level (removes the world-model prediction
error, isolates the representation/decoder gap). Held-out frames.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from crave.encoders import load_encoder  # noqa: E402
from crave.decoding.decoder import train_dec  # noqa: E402


def load_index(feature_dir: Path):
    idx = np.load(feature_dir / "index.npz")
    e, fr, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), int(idx["n"])
    valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard)
        valid[z["gidx"].astype(np.int64)] = z["valid"].astype(bool)
    return e[valid], fr[valid]


def read_frames(dataset_root, camera, E, FR, gidx, enc_res, tgt_res):
    cs = int(json.loads((dataset_root / "meta/info.json").read_text())["chunks_size"])
    ie = np.zeros((len(gidx), enc_res, enc_res, 3), np.uint8)
    it = np.zeros((len(gidx), tgt_res, tgt_res, 3), np.uint8)
    by_ep: dict[int, list[int]] = {}
    for k, gi in enumerate(gidx):
        by_ep.setdefault(int(E[gi]), []).append(k)
    for ep, ks in by_ep.items():
        cap = cv2.VideoCapture(str(dataset_root / f"videos/chunk-{ep // cs:03d}/{camera}/episode_{ep:06d}.mp4"))
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(FR[gidx[k]]))
            ok, fr = cap.read()
            if ok:
                rgb = fr[:, :, ::-1]
                ie[k] = cv2.resize(rgb, (enc_res, enc_res)); it[k] = cv2.resize(rgb, (tgt_res, tgt_res))
        cap.release()
    return ie, it


class UnPool(nn.Module):
    """pooled (d) -> patch grid (d, 16, 16)."""

    def __init__(self, d=1280):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, 512 * 4 * 4), nn.GELU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.GELU(),   # 4->8
            nn.ConvTranspose2d(512, d, 4, 2, 1),                                     # 8->16
        )

    def forward(self, z):
        return self.up(self.fc(z).view(-1, 512, 4, 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--epochs_dec", type=int, default=60)
    ap.add_argument("--epochs_unpool", type=int, default=60)
    ap.add_argument("--out_dir", default="lmwm/outputs/patch_decoder", type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    dev = "cuda"

    E, FR = load_index(args.feature_dir)
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    is_val = np.array([e in val_eps for e in E])
    tr_g = rng.choice(np.where(~is_val)[0], args.n_train, replace=False)
    va_g = rng.choice(np.where(is_val)[0], args.n_val, replace=False)

    print("reading frames ...", flush=True)
    tr_enc, tr_tgt = read_frames(args.dataset_root, args.camera, E, FR, tr_g, 256, 128)
    va_enc, va_tgt = read_frames(args.dataset_root, args.camera, E, FR, va_g, 256, 128)

    enc = load_encoder("dinov3-h", device=dev)
    print("encoding pooled + grid ...", flush=True)
    tr_grid = enc.encode_grid(tr_enc).astype(np.float32); va_grid = enc.encode_grid(va_enc).astype(np.float32)
    tr_pool = enc.encode_pooled(tr_enc).astype(np.float32); va_pool = enc.encode_pooled(va_enc).astype(np.float32)
    din = tr_grid.shape[1]

    print("training patch decoder ...", flush=True)
    decode = train_dec(tr_grid, tr_tgt, din, dec="small", epochs=args.epochs_dec, device=dev)

    # Path 1: direct patch decode fidelity
    l1_direct = float(np.abs(va_tgt.astype(float) - decode(va_grid).astype(float)).mean())

    # Path 2: train un-pool pooled -> grid, then decode
    print("training un-pool ...", flush=True)
    U = UnPool(din).to(dev)
    opt = torch.optim.AdamW(U.parameters(), lr=2e-4, weight_decay=1e-5)
    Xtr = torch.from_numpy(tr_pool).to(dev); Gtr = torch.from_numpy(tr_grid).to(dev)
    gmu, gsd = Gtr.mean(), Gtr.std() + 1e-6
    n, bs = len(Xtr), 128
    for ep in range(args.epochs_unpool):
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, bs):
            bi = perm[b:b + bs]
            loss = F.smooth_l1_loss(U(Xtr[bi]), Gtr[bi], beta=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
    U.eval()
    with torch.no_grad():
        va_grid_hat = []
        Xva = torch.from_numpy(va_pool).to(dev)
        for b in range(0, len(Xva), 256):
            va_grid_hat.append(U(Xva[b:b + 256]).cpu().numpy())
    va_grid_hat = np.concatenate(va_grid_hat).astype(np.float32)
    grid_mse = float(((va_grid_hat - va_grid) ** 2).mean())
    grid_cos = float((va_grid_hat.reshape(len(va_g), -1) * va_grid.reshape(len(va_g), -1)).sum(1).mean() /
                     (np.linalg.norm(va_grid_hat.reshape(len(va_g), -1), axis=1) * np.linalg.norm(va_grid.reshape(len(va_g), -1), axis=1) + 1e-8).mean())
    l1_unpool = float(np.abs(va_tgt.astype(float) - decode(va_grid_hat).astype(float)).mean())

    summary = {
        "n_train": len(tr_g), "n_val": len(va_g), "din": din,
        "path1_direct_patch_decode_L1_frac": round(l1_direct / 255, 4),
        "path2_unpool_then_decode_L1_frac": round(l1_unpool / 255, 4),
        "unpool_grid_mse": round(grid_mse, 4),
        "unpool_grid_cos": round(grid_cos, 4),
        "pooled_decoder_baseline_L1_frac": 0.062,
        "gap_path2_minus_path1_L1_frac": round((l1_unpool - l1_direct) / 255, 4),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "unpool_vs_patch.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 8, figsize=(16, 6))
    for j in range(8):
        ax[0, j].imshow(va_tgt[j]); ax[1, j].imshow(decode(va_grid[j:j+1])[0]); ax[2, j].imshow(decode(va_grid_hat[j:j+1])[0])
        for r in range(3): ax[r, j].axis("off")
    ax[0, 0].set_ylabel("real"); ax[1, 0].set_ylabel("path1 direct patch"); ax[2, 0].set_ylabel("path2 unpool")
    fig.suptitle(f"path1 direct={l1_direct/255:.3f}  vs  path2 unpool={l1_unpool/255:.3f}  (pooled baseline 0.062)", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out_dir / "unpool_vs_patch.png", dpi=110); plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"saved {args.out_dir}/unpool_vs_patch.json + .png")


if __name__ == "__main__":
    main()
