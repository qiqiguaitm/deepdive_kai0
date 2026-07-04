#!/usr/bin/env python
"""Full-episode self-reconstruction video @ 30Hz: real | L1 decode | flow decode.
Each frame -> gated DINOv3-H encode_pooled -> {dec_v2 (L1), dec_best (flow)} -> image.
Cross-domain check: does the kai0-trained decoder faithfully reproduce a vis_base episode,
or hallucinate kai0 appearance? No milestone pipeline needed (pure encode->decode).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dinov3h_decoder import PooledDecoder, l2  # noqa: E402
from decode_best import load_best_decoder  # noqa: E402
from dinov3h_gated import DINOv3HGated  # noqa: E402
from make_episode_native_video import label  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="", help="explicit episode mp4; else auto-pick first under --root")
    ap.add_argument("--root", default="kai0/data/Task_A/vis_base/v4/2026-04-23-v4")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--dec_l1", default="lmwm/checkpoints/dinov3h_decoder/dec_v2.pt")
    ap.add_argument("--flow", default="lmwm/checkpoints/dinov3h_decoder/dec_best.pt")
    ap.add_argument("--dino_path", default="/vePFS/xiezhicong/.cache/huggingface/hub/dinov3-vith16plus-pretrain-lvd1689m")
    ap.add_argument("--out_fps", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=3000)
    ap.add_argument("--enc_bs", type=int, default=48)
    ap.add_argument("--out", default="lmwm/docs/assets/visbase_selfrecon_flow.mp4", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vid = args.video or sorted(glob.glob(str(Path(args.root) / f"videos/chunk-*/{args.camera}/episode_*.mp4")))[0]
    print(f"episode: {vid}", flush=True)

    ck = torch.load(args.dec_l1, map_location="cpu"); R = int(ck["res"])
    DL = PooledDecoder(din=int(ck["din"]), res=R).to(dev); DL.load_state_dict(ck["model"]); DL.eval()
    flow = load_best_decoder(args.flow, str(dev))
    enc = DINOv3HGated(args.dino_path, device=str(dev))

    # 1) read all frames
    cap = cv2.VideoCapture(vid); frames = []
    while len(frames) < args.max_frames:
        okf, im = cap.read()
        if not okf:
            break
        frames.append(im[:, :, ::-1])                       # BGR->RGB
    cap.release()
    N = len(frames); print(f"{N} frames read", flush=True)
    small = np.stack([cv2.resize(f, (R, R)) for f in frames])

    # 2) batched encode_pooled (gated DINOv3-H)
    lats = []
    for s in range(0, N, args.enc_bs):
        b = torch.from_numpy(small[s:s + args.enc_bs].astype(np.float32) / 255).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad():
            lats.append(enc.encode_pooled_tensor(b).float().cpu().numpy())
        if s % (args.enc_bs * 10) == 0:
            print(f"  encoded {s}/{N}", flush=True)
    lat = l2(np.concatenate(lats).astype(np.float32))

    # 3) batched decode (L1 deterministic, flow ODE)
    def dec_l1(a):
        out = []
        for s in range(0, len(a), 64):
            with torch.no_grad():
                o = DL(torch.from_numpy(a[s:s + 64]).to(dev)).cpu().numpy()
            out.append(np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8))
        return np.concatenate(out)
    rL = dec_l1(lat)
    rF = np.concatenate([flow(lat[s:s + 64]) for s in range(0, N, 64)])
    print("decoded L1 + flow", flush=True)

    # 4) compose real | L1 | flow @ out_fps
    BIG = R * 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def compose(i):
        left = label(cv2.resize(frames[i], (BIG, BIG)), f"vis_base real (native) frame {i}")
        m1 = label(cv2.resize(rL[i], (BIG, BIG)), "-> L1 decode")
        m2 = label(cv2.resize(rF[i], (BIG, BIG)), "-> flow decode")
        gap = np.full((left.shape[0], 8, 3), 20, np.uint8)
        return np.hstack([left, gap, m1, gap, m2])
    c0 = compose(0); Hc, Wc = c0.shape[:2]
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.out_fps, (Wc, Hc))
    for i in range(N):
        vw.write(cv2.cvtColor(compose(i), cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"saved {args.out} | {N} frames @ {args.out_fps}fps | real|L1|flow self-recon (vis_base cross-domain)", flush=True)


if __name__ == "__main__":
    main()
