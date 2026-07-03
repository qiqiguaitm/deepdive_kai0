#!/usr/bin/env python
"""Self-reconstruction comparison (unified space): real frame -> DINOv3-H -> {dec_v2 L1,
dec_gan_v2 GAN} -> image. Rows: real | dec_v2(L1) | dec_gan_v2(GAN). Reports L1 + sharpness.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dinov3h_decoder import PooledDecoder, load_features, l2  # noqa: E402


def load_dec(p, dev):
    ck = torch.load(p, map_location="cpu"); R = int(ck["res"])
    D = PooledDecoder(din=int(ck["din"]), res=R).to(dev); D.load_state_dict(ck["model"]); D.eval(); return D, R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dec_l1", default="lmwm/checkpoints/dinov3h_decoder/dec_v2.pt")
    ap.add_argument("--dec_gdl", default="lmwm/checkpoints/dinov3h_decoder/dec_gdl_v2.pt")
    ap.add_argument("--dec_gan", default="lmwm/checkpoints/dinov3h_decoder/dec_gan_v2.pt")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h_v2")
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="lmwm/docs/assets/selfrecon_compare.png", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    DL, R = load_dec(args.dec_l1, dev); DGd, _ = load_dec(args.dec_gdl, dev); DG, _ = load_dec(args.dec_gan, dev)
    E, FR, Fb = load_features(Path(args.feature_dir)); Fn = l2(Fb.astype(np.float32))
    rng = np.random.default_rng(3); sel = rng.choice(len(E), args.n, replace=False)
    def dec(D):
        with torch.no_grad():
            o = D(torch.from_numpy(Fn[sel]).to(dev)).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
    rL, rGd, rG = dec(DL), dec(DGd), dec(DG)

    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    caps: dict[int, cv2.VideoCapture] = {}
    def frame(ep, t):
        if ep not in caps:
            caps[ep] = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(t)); okf, im = caps[ep].read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if okf else np.zeros((R, R, 3), np.uint8)
    reals = np.stack([frame(int(E[g]), int(FR[g])) for g in sel])
    for c in caps.values():
        c.release()

    def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    def l1(a): return float(np.abs(reals.astype(float) - a.astype(float)).mean() / 255)
    stat = (f"real sharp {np.mean([sharp(x) for x in reals]):.0f} | "
            f"L1 L1={l1(rL):.3f} sharp {np.mean([sharp(x) for x in rL]):.0f} | "
            f"GDL L1={l1(rGd):.3f} sharp {np.mean([sharp(x) for x in rGd]):.0f} | "
            f"GAN L1={l1(rG):.3f} sharp {np.mean([sharp(x) for x in rG]):.0f}")

    fig, ax = plt.subplots(4, args.n, figsize=(args.n * 1.5, 6.5))
    lab = ["real", "dec_v2 (L1)", "dec_gdl_v2 (GDL)", "dec_gan_v2 (GAN)"]
    for r, imgs in enumerate([reals, rL, rGd, rG]):
        for j in range(args.n):
            ax[r, j].imshow(imgs[j]); ax[r, j].axis("off")
        ax[r, 0].set_ylabel(lab[r], fontsize=9)
    fig.suptitle("self-reconstruction: L1 (faithful, soft) vs GAN (sharp, mild hallucination) | " + stat, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"saved {args.out} | {stat}")


if __name__ == "__main__":
    main()
