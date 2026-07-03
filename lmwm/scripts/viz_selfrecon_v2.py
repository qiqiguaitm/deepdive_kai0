#!/usr/bin/env python
"""dec_v2 self-reconstruction ceiling (unified space): real frame -> DINOv3-H encode_pooled
-> dec_v2 decode, vs the real frame. This is the BEST dec_v2 can do (no prediction).
Top row = real, bottom row = decode; reports L1 + sharpness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from train_dinov3h_decoder import PooledDecoder, load_features, l2  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", default="lmwm/checkpoints/dinov3h_decoder/dec_v2.pt")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h_v2")
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="lmwm/docs/assets/selfrecon_dec_v2.png", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.decoder, map_location="cpu"); R = int(ck["res"])
    D = PooledDecoder(din=int(ck["din"]), res=R).to(dev); D.load_state_dict(ck["model"]); D.eval()

    E, FR, Fb = load_features(Path(args.feature_dir)); Fn = l2(Fb.astype(np.float32))
    rng = np.random.default_rng(3); sel = rng.choice(len(E), args.n, replace=False)
    with torch.no_grad():
        o = D(torch.from_numpy(Fn[sel]).to(dev)).cpu().numpy()
    rec = np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)

    cs = int(__import__("json").loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    caps: dict[int, cv2.VideoCapture] = {}
    def frame(ep, t):
        if ep not in caps:
            caps[ep] = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(t)); okf, im = caps[ep].read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if okf else np.zeros((R, R, 3), np.uint8)
    reals = np.stack([frame(int(E[g]), int(FR[g])) for g in sel])
    for c in caps.values():
        c.release()

    l1 = float(np.abs(reals.astype(float) - rec.astype(float)).mean() / 255)
    def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    sr, srr = np.mean([sharp(x) for x in rec]), np.mean([sharp(x) for x in reals])

    fig, ax = plt.subplots(2, args.n, figsize=(args.n * 1.5, 3.4))
    for j in range(args.n):
        ax[0, j].imshow(reals[j]); ax[0, j].axis("off"); ax[1, j].imshow(rec[j]); ax[1, j].axis("off")
    ax[0, 0].set_ylabel("real", fontsize=9); ax[1, 0].set_ylabel("dec_v2", fontsize=9)
    fig.suptitle(f"dec_v2 self-reconstruction (unified space) | L1={l1:.3f} | sharpness {sr:.0f} vs real {srr:.0f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"saved {args.out} | self-recon L1={l1:.4f} sharpness={sr:.0f} (real {srr:.0f})")


if __name__ == "__main__":
    main()
