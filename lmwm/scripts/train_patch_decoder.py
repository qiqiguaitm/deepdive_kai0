#!/usr/bin/env python
"""Extract DINOv3-H PATCH-GRID features and train CRAVE's patch decoder; measure
self-reconstruction fidelity vs the pooled decoder (which was ~6.2% L1).

Patch-grid (N,1280,16,16) keeps spatial layout that pooling averages away, so its
decode should be more FAITHFUL to the original (the user's goal: preserve the same
information, not necessarily sharper).

Pipeline: sample train + held-out frames -> encode_grid (DINOv3-H) -> train
CRAVE's small spatial decoder -> report val L1 (decode(own grid) vs real frame).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

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
    """Return (imgs_enc [N,enc_res,enc_res,3] uint8, imgs_tgt [N,tgt_res,tgt_res,3] uint8)."""
    cs = int(json.loads((dataset_root / "meta/info.json").read_text())["chunks_size"])
    ie = np.zeros((len(gidx), enc_res, enc_res, 3), np.uint8)
    it = np.zeros((len(gidx), tgt_res, tgt_res, 3), np.uint8)
    by_ep: dict[int, list[int]] = {}
    for k, gi in enumerate(gidx):
        by_ep.setdefault(int(E[gi]), []).append(k)
    done = 0
    for ep, ks in by_ep.items():
        cap = cv2.VideoCapture(str(dataset_root / f"videos/chunk-{ep // cs:03d}/{camera}/episode_{ep:06d}.mp4"))
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(FR[gidx[k]]))
            ok, fr = cap.read()
            if ok:
                rgb = fr[:, :, ::-1]
                ie[k] = cv2.resize(rgb, (enc_res, enc_res))
                it[k] = cv2.resize(rgb, (tgt_res, tgt_res))
            done += 1
        cap.release()
        if done % 4000 < len(ks):
            print(f"  read {done}/{len(gidx)}", flush=True)
    return ie, it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--enc_res", type=int, default=256)
    ap.add_argument("--tgt_res", type=int, default=128)
    ap.add_argument("--dec", default="small")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out_dir", default="lmwm/outputs/patch_decoder", type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    E, FR = load_index(args.feature_dir)
    rng = np.random.default_rng(args.seed)
    eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    is_val = np.array([e in val_eps for e in E])
    tr_pool = np.where(~is_val)[0]; va_pool = np.where(is_val)[0]
    tr_g = rng.choice(tr_pool, min(args.n_train, len(tr_pool)), replace=False)
    va_g = rng.choice(va_pool, min(args.n_val, len(va_pool)), replace=False)

    print(f"reading {len(tr_g)} train + {len(va_g)} val frames ...", flush=True)
    tr_enc, tr_tgt = read_frames(args.dataset_root, args.camera, E, FR, tr_g, args.enc_res, args.tgt_res)
    va_enc, va_tgt = read_frames(args.dataset_root, args.camera, E, FR, va_g, args.enc_res, args.tgt_res)

    enc = load_encoder("dinov3-h", device="cuda")
    print("encoding patch grids ...", flush=True)
    tr_grid = enc.encode_grid(tr_enc).astype(np.float32)   # (N,1280,16,16)
    va_grid = enc.encode_grid(va_enc).astype(np.float32)
    din = tr_grid.shape[1]

    print(f"training CRAVE patch decoder (dec={args.dec}, din={din}) ...", flush=True)
    decode = train_dec(tr_grid, tr_tgt, din, dec=args.dec, epochs=args.epochs, device="cuda")

    va_rec = decode(va_grid)  # uint8
    l1 = float(np.abs(va_tgt.astype(float) - va_rec.astype(float)).mean())
    def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    summary = {
        "n_train": len(tr_g), "n_val": len(va_g), "din": din, "dec": args.dec, "epochs": args.epochs,
        "patch_decoder_val_L1_over255": round(l1, 2),
        "patch_decoder_val_L1_frac": round(l1 / 255, 4),
        "patch_decoder_val_sharpness": round(float(np.mean([sharp(x) for x in va_rec])), 1),
        "real_val_sharpness": round(float(np.mean([sharp(x) for x in va_tgt])), 1),
        "compare_pooled_decoder_L1_frac": 0.062,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # a comparison strip: real vs patch-decode for 8 val frames
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 8, figsize=(16, 4))
    for j in range(8):
        ax[0, j].imshow(va_tgt[j]); ax[0, j].axis("off")
        ax[1, j].imshow(va_rec[j]); ax[1, j].axis("off")
    ax[0, 0].set_ylabel("real", fontsize=9); ax[1, 0].set_ylabel("patch-decode", fontsize=9)
    fig.suptitle(f"DINOv3-H patch-grid decode (held-out) | val L1={l1/255:.3f} (pooled was 0.062)", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out_dir / "patch_recon_compare.png", dpi=110); plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"saved {args.out_dir}/summary.json + patch_recon_compare.png")


if __name__ == "__main__":
    main()
