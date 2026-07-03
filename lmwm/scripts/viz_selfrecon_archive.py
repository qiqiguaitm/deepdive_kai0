#!/usr/bin/env python
"""Archived-decoders self-reconstruction comparison on the SAME original frames.
Same real frame -> gated DINOv3-H encode_pooled -> {dec_v2 (L1, 高一致性), flow_b160 (最好效果)}.
Rows: real | L1 | flow. Per decoder: re-encode cos (语义保真) + sharpness + pixel L1.
Only the two archived schemes are shown (GAN/GDL/retrieval/re-encode eliminated).
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
from train_dinov3h_decoder import PooledDecoder, load_features, l2  # noqa: E402
from decode_best import load_best_decoder  # noqa: E402
from dinov3h_gated import DINOv3HGated  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dec_l1", default="lmwm/checkpoints/dinov3h_decoder/dec_v2.pt")
    ap.add_argument("--flow", default="lmwm/checkpoints/dinov3h_decoder/dec_best.pt")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h_v2")
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--dino_path", default="/vePFS/xiezhicong/.cache/huggingface/hub/dinov3-vith16plus-pretrain-lvd1689m")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="lmwm/docs/assets/selfrecon_compare.png", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.dec_l1, map_location="cpu"); R = int(ck["res"])
    DL = PooledDecoder(din=int(ck["din"]), res=R).to(dev); DL.load_state_dict(ck["model"]); DL.eval()
    flow = load_best_decoder(args.flow, str(dev))

    E, FRi, Fb = load_features(Path(args.feature_dir)); Fn = l2(Fb.astype(np.float32))
    rng = np.random.default_rng(3); sel = rng.choice(len(E), args.n, replace=False)   # same seed as legacy compare
    lat = torch.from_numpy(Fn[sel]).to(dev)

    with torch.no_grad():
        oL = DL(lat).cpu().numpy()
    rL = np.clip((oL.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
    rF = flow(Fn[sel])

    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    caps: dict[int, cv2.VideoCapture] = {}
    def frame(ep, t):
        if ep not in caps:
            caps[ep] = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(t)); okf, im = caps[ep].read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if okf else np.zeros((R, R, 3), np.uint8)
    reals = np.stack([frame(int(E[g]), int(FRi[g])) for g in sel])
    for c in caps.values():
        c.release()

    reenc = DINOv3HGated(args.dino_path, device=str(dev)); lat_np = Fn[sel]
    def rcos(imgs):
        re = F.normalize(reenc.encode_pooled_tensor(torch.from_numpy(imgs.astype(np.float32) / 255).permute(0, 3, 1, 2).to(dev)), dim=-1)
        return float((re.float().cpu().numpy() * lat_np).sum(1).mean())
    def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    def sh(a): return float(np.mean([sharp(x) for x in a]))
    def l1(a): return float(np.abs(reals.astype(float) - a.astype(float)).mean() / 255)

    stat = {"real_sharp": round(sh(reals), 0),
            "L1": {"reencode_cos": round(rcos(rL), 4), "sharp": round(sh(rL), 0), "pixel_L1": round(l1(rL), 4)},
            "flow": {"reencode_cos": round(rcos(rF), 4), "sharp": round(sh(rF), 0), "pixel_L1": round(l1(rF), 4)}}
    print(json.dumps(stat, ensure_ascii=False), flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = [("real", reals),
            (f"dec_v2 L1 (high-consistency)\ncos {stat['L1']['reencode_cos']} . sharp {stat['L1']['sharp']:.0f}", rL),
            (f"flow (best) *\ncos {stat['flow']['reencode_cos']} . sharp {stat['flow']['sharp']:.0f}", rF)]
    fig, ax = plt.subplots(3, args.n, figsize=(args.n * 1.5, 5.4))
    for r, (lab, imgs) in enumerate(rows):
        for j in range(args.n):
            ax[r, j].imshow(imgs[j]); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            for sp in ax[r, j].spines.values():
                sp.set_visible(False)
        ax[r, 0].set_ylabel(lab, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=30)
    fig.suptitle("self-reconstruction: same original frame -> DINOv3-H encode -> decode  |  archived 2 schemes: "
                 "L1 (deterministic mean, high-consistency)  vs  flow (generative, sharp+faithful)  |  GAN/others dropped",
                 fontsize=8.5)
    fig.tight_layout(rect=[0.07, 0, 1, 0.95])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
