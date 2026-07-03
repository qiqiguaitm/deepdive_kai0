#!/usr/bin/env python
"""Render: predicted-grid decode vs TRUE next-medoid vs TRUE-medoid encode->decode.

LMWM output feeds the VLA as a feature/subgoal, so the grid predictor is trained
ON-MANIFOLD with the LaWM feature-space loss (smooth_l1 + 1-cos), NOT the off-manifold
decode loss. Columns per row:
  current (real) | predicted grid -> decode | TRUE medoid -> encode -> decode (decoder
  self-recon CEILING) | TRUE medoid (real)
The gap (col2 vs col3) = PREDICTION error; the gap (col3 vs col4) = DECODER error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from lmwm.data import split_indices  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from crave.decoding.decoder import make_decoder  # noqa: E402
from track_b2_grid_predict import GridGen, render_medoid_images  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--patch_dec", default="lmwm/checkpoints/patch_decoder/patch_dec.pt", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", default="lmwm/docs/assets/grid_pred_vs_ceiling.png", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.patch_dec, map_location="cpu", weights_only=False)
    din_grid = int(ck["din"]); R = int(ck["res"])
    D = make_decoder(din_grid, ck["dec"]).to(dev); D.load_state_dict(ck["model"]); D.eval()
    for p in D.parameters():
        p.requires_grad_(False)
    muT = torch.from_numpy(ck["mu"]).view(1, din_grid, 1, 1).to(dev)
    sdT = torch.from_numpy(ck["sd"]).view(1, din_grid, 1, 1).to(dev)
    def decode(grid): return D((grid - muT) / sdT)
    def d2img(t): return np.clip((t.detach().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    ti, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    ti, vi = ti.numpy(), vi.numpy()
    ok = np.linalg.norm(z["next_medoid"], axis=1) > 1e-6
    rng = np.random.default_rng(0)
    ti = rng.choice(ti[ok[ti]], min(args.n_train, int(ok[ti].sum())), replace=False)
    vsel = rng.choice(vi[ok[vi]], args.rows, replace=False)
    din_in = z["current"].shape[1]

    print("rendering + encoding train medoid grids ...", flush=True)
    tr_img, _ = render_medoid_images(z, ti, args.dataset_root, args.camera, args.feature_dir, R)
    enc = load_encoder("dinov3-h", device=str(dev))
    tr_grid = torch.from_numpy(enc.encode_grid(tr_img).astype(np.float32))
    Xt = torch.from_numpy(z["current"][ti].astype(np.float32))
    ntr = len(ti)

    print("training on-manifold grid predictor (LaWM feature-space loss) ...", flush=True)
    torch.manual_seed(0)
    G = GridGen(din_in, din_grid).to(dev)
    opt = torch.optim.AdamW(G.parameters(), lr=3e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
    for s in range(args.steps):
        bi = torch.randint(0, ntr, (64,))
        grid = G(Xt[bi].to(dev)); tg = tr_grid[bi].to(dev)
        rt = grid.flatten(2).transpose(1, 2); tt = tg.flatten(2).transpose(1, 2)
        loss = F.smooth_l1_loss(rt, tt, beta=0.1) + (1.0 - F.cosine_similarity(rt, tt, dim=-1).mean())
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    G.eval()

    print("rendering val samples ...", flush=True)
    va_img, _ = render_medoid_images(z, vsel, args.dataset_root, args.camera, args.feature_dir, R)
    va_grid = torch.from_numpy(enc.encode_grid(va_img).astype(np.float32)).to(dev)
    Xv = torch.from_numpy(z["current"][vsel].astype(np.float32)).to(dev)
    with torch.no_grad():
        pred_grid = G(Xv)
        pred_dec = decode(pred_grid)        # predicted grid -> decode
        ceil_dec = decode(va_grid)          # true medoid encode -> decode (ceiling)
    # metrics
    rt = pred_grid.flatten(2).transpose(1, 2); tt = va_grid.flatten(2).transpose(1, 2)
    feat_cos = F.cosine_similarity(rt, tt, dim=-1).mean(1).cpu().numpy()
    Yreal = torch.from_numpy(va_img.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).to(dev)
    l1_pred = F.l1_loss(pred_dec, Yreal, reduction="none").mean((1, 2, 3)).cpu().numpy()
    l1_ceil = F.l1_loss(ceil_dec, Yreal, reduction="none").mean((1, 2, 3)).cpu().numpy()

    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    ep_s = z["episode_id"][vsel]; t_s = z["t"][vsel]
    caps: dict[int, cv2.VideoCapture] = {}
    def frame(ep, t):
        if ep not in caps:
            caps[ep] = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(t)); okf, im = caps[ep].read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if okf else np.zeros((R, R, 3), np.uint8)

    titles = ["current (real)", "PRED grid -> decode", "TRUE medoid enc->decode (ceiling)", "TRUE medoid (real)"]
    fig, axes = plt.subplots(args.rows, 4, figsize=(4 * 2.3, args.rows * 2.35))
    for i in range(args.rows):
        imgs = [frame(int(ep_s[i]), int(t_s[i])), d2img(pred_dec[i]), d2img(ceil_dec[i]), va_img[i]]
        for ci, im in enumerate(imgs):
            a = axes[i, ci]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(titles[ci], fontsize=8)
        axes[i, 1].set_ylabel(f"feat_cos={feat_cos[i]:.2f}\nL1 pred={l1_pred[i]:.2f}\nL1 ceil={l1_ceil[i]:.2f}", fontsize=6.5)
    for c in caps.values():
        c.release()
    fig.suptitle(f"grid subgoal (on-manifold, LaWM loss) | feat_cos={feat_cos.mean():.3f} | "
                 f"decode L1: PRED {l1_pred.mean():.3f} vs CEILING {l1_ceil.mean():.3f} (decoder floor)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"saved {args.out}")
    print(f"feat_cos={feat_cos.mean():.4f} | L1 pred={l1_pred.mean():.4f} ceil={l1_ceil.mean():.4f}")


if __name__ == "__main__":
    main()
