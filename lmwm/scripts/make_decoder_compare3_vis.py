#!/usr/bin/env python
"""3-way decoder comparison on a held-out kai0_base test episode (self-reconstruction — isolates
DECODER fidelity, same real frame representation decoded three ways):

  real | patch-grid decode (spatial) | pooled -> flow FIXED-noise | pooled -> dec_v2 (L1)

Reuses prior patch work: crave.encoders.encode_grid + crave.decoding.decoder.make_decoder + the
saved checkpoints/patch_decoder/patch_dec.pt (Track B1). Pooled decoders via load_any_decoder.
Outputs a consecutive-keyframe filmstrip PNG + a 4-panel video.
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
from make_prod_video_vis_fwd import load_any_decoder  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from crave.decoding.decoder import make_decoder  # noqa: E402
from make_episode_native_video import label  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="kai0/data/Task_A/kai0_base/videos/chunk-000/observation.images.top_head/episode_000008.mp4")
    ap.add_argument("--patch_ckpt", default="lmwm/checkpoints/patch_decoder/patch_dec.pt", type=Path)
    ap.add_argument("--flow", default="lmwm/checkpoints/dinov3h_decoder/dec_best.pt")
    ap.add_argument("--l1", default="lmwm/checkpoints/dinov3h_decoder/dec_v2.pt")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--enc_res", type=int, default=256)
    ap.add_argument("--max_frames", type=int, default=3000)
    ap.add_argument("--out_vid", default="lmwm/docs/assets/decoder_compare3_kai0base_testep8.mp4", type=Path)
    ap.add_argument("--out_png", default="lmwm/docs/assets/decoder_compare3_kai0base_testep8.png", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    enc = load_encoder("dinov3-h", device=str(dev))
    # patch-grid decoder (reused Track B1 checkpoint)
    pk = torch.load(args.patch_ckpt, map_location="cpu", weights_only=False); din = int(pk["din"])
    Dp = make_decoder(din, pk["dec"]).to(dev); Dp.load_state_dict(pk["model"]); Dp.eval()
    muT = torch.from_numpy(pk["mu"]).view(1, din, 1, 1).to(dev)
    sdT = torch.from_numpy(pk["sd"]).view(1, din, 1, 1).to(dev)
    print(f"patch decoder: din={din} dec={pk['dec']} val_L1={pk.get('val_L1_frac')} "
          f"sharp {pk.get('val_sharp'):.0f} vs real {pk.get('real_sharp'):.0f}", flush=True)
    def patch_decode(imgs256):
        out = []
        for s in range(0, len(imgs256), 64):
            grid = enc.encode_grid(imgs256[s:s + 64]).astype(np.float32)  # (n,1280,16,16)
            with torch.no_grad():
                o = Dp((torch.from_numpy(grid).to(dev) - muT) / sdT).cpu().numpy()
            out.append(np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8))
        return np.concatenate(out)

    dec_fx, Rf, tag_f = load_any_decoder(args.flow, dev)   # flow branch -> seed=0 fixed-noise
    dec_l1, Rl, _ = load_any_decoder(args.l1, dev)
    print(f"pooled decoders: flow [{tag_f}] + dec_v2", flush=True)

    def pooled_lat(imgs256):
        z = enc.encode_pooled(imgs256).astype(np.float32)
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    cap = cv2.VideoCapture(args.video); frames = []
    while len(frames) < args.max_frames:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im[:, :, ::-1])
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    out_fps = round(fps) if fps and fps > 1 else 30
    N = len(frames); kt = np.arange(0, N, args.stride)
    print(f"episode {Path(args.video).name}: {N} frames, {len(kt)} keyframes", flush=True)

    kimgs = np.stack([cv2.resize(frames[t], (args.enc_res, args.enc_res)) for t in kt])
    d_patch = patch_decode(kimgs)
    lat = pooled_lat(kimgs)
    d_fx, d_l1 = dec_fx(lat), dec_l1(lat)
    def sh(a): return float(np.mean([cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for x in a]))
    realk = np.stack([cv2.resize(frames[t], (128, 128)) for t in kt])
    def l1frac(a): return float(np.abs(np.stack([cv2.resize(x, (a.shape[1], a.shape[1])) for x in realk]).astype(float) - a.astype(float)).mean() / 255)
    print(f"patch : L1 {l1frac(d_patch):.4f}  sharp {sh(d_patch):.0f}", flush=True)
    print(f"flow  : L1 {l1frac(d_fx):.4f}  sharp {sh(d_fx):.0f}", flush=True)
    print(f"dec_v2: L1 {l1frac(d_l1):.4f}  sharp {sh(d_l1):.0f}  (real sharp {sh(realk):.0f})", flush=True)

    # filmstrip: 10 consecutive keyframes mid-episode
    s0 = len(kt) // 3; cols = list(range(s0, min(s0 + 10, len(kt)))); C = 150
    def strip(getter): return np.hstack([cv2.resize(getter(k), (C, C)) for k in cols])
    def lab(img, t):
        o = img.copy(); cv2.putText(o, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1); return o
    rows = [("kai0_base real (test ep8)", strip(lambda k: cv2.resize(frames[kt[k]], (C, C)))),
            ("patch-grid decode (spatial, L1 0.027)", strip(lambda k: d_patch[k])),
            ("pooled -> flow FIXED-noise (sharp)", strip(lambda k: d_fx[k])),
            ("pooled -> dec_v2 L1 (blurry)", strip(lambda k: d_l1[k]))]
    grid = np.vstack([lab(r[:, :, ::-1].copy(), name) for name, r in rows])
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_png), grid)
    print(f"saved {args.out_png} {grid.shape}", flush=True)

    # 4-panel video
    P = 256
    def L(img, t): return label(cv2.resize(img, (P, P)), t)
    def compose(f):
        k = int(np.searchsorted(kt, f, side="right") - 1); k = max(0, min(k, len(kt) - 1))
        p0 = L(frames[f], "kai0_base real (test ep8)")
        p1 = L(d_patch[k], "patch-grid decode")
        p2 = L(d_fx[k], "pooled -> flow FIXED-noise")
        p3 = L(d_l1[k], "pooled -> dec_v2 (L1)")
        gap = np.full((p0.shape[0], 8, 3), 20, np.uint8)
        return np.ascontiguousarray(np.hstack([p0, gap, p1, gap, p2, gap, p3]))
    c0 = compose(0); Hc, Wc = c0.shape[:2]; Hc -= Hc % 2; Wc -= Wc % 2
    vw = cv2.VideoWriter(str(args.out_vid), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (Wc, Hc))
    assert vw.isOpened(), f"VideoWriter failed ({Wc}x{Hc})"
    for f in range(N):
        vw.write(cv2.cvtColor(compose(f)[:Hc, :Wc], cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"saved {args.out_vid} | {N} frames @ {out_fps}fps | 4-panel real|patch|flow|dec_v2", flush=True)


if __name__ == "__main__":
    main()
