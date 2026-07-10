#!/usr/bin/env python
"""Render a side-by-side [real | decoder1 | decoder2 | ...] video of encode->decode fidelity,
so the best patch-grid decoder can be judged by eye (per the lesson: sharpness metrics can be
gamed, must look at pixels). Encodes each frame with DINOv3-H (standalone fallback ok), decodes
with each checkpoint, composes a labeled panel, writes mp4.
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
from crave.encoders import load_encoder  # noqa: E402
from crave.decoding.decoder import make_decoder  # noqa: E402


def load_decoder(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    D = make_decoder(ck["din"], ck["dec"]).to(device); D.load_state_dict(ck["model"]); D.eval()
    mu = torch.from_numpy(np.asarray(ck["mu"])).view(1, -1, 1, 1).to(device)
    sd = torch.from_numpy(np.asarray(ck["sd"])).view(1, -1, 1, 1).to(device)
    label = f"{ck.get('dec','?')}{'+GDL' if ck.get('gdl') else ''} L1={ck.get('val_L1_frac','?')} sharp={int(ck.get('val_sharp',0))}"

    def dec(grids_np):
        with torch.no_grad():
            o = D((torch.from_numpy(grids_np).to(device) - mu) / sd).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
    return dec, label


def label_bar(w, text, h=26):
    bar = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return bar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=100)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--decoders", nargs="+", required=True, help="decoder .pt checkpoints")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--cell", type=int, default=256, help="display size per panel")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default="lmwm/outputs/decoder_video.mp4")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    vid = args.dataset_root / f"videos/chunk-{args.episode // cs:03d}/{args.camera}/episode_{args.episode:06d}.mp4"
    cap = cv2.VideoCapture(str(vid)); frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(im[:, :, ::-1], (256, 256)))
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {vid}")
    if len(frames) > args.max_frames:
        idx = np.linspace(0, len(frames) - 1, args.max_frames).astype(int)
        frames = [frames[i] for i in idx]
    F = np.stack(frames).astype(np.uint8)
    print(f"ep{args.episode}: {len(F)} frames from {vid.name}", flush=True)

    enc = load_encoder("dinov3-h", device=dev)
    print("encoding grids ...", flush=True)
    grids = enc.encode_grid(F).astype(np.float32)

    decs, labels = [], ["real"]
    for cp in args.decoders:
        d, lab = load_decoder(cp, dev); decs.append(d); labels.append(lab)
        print(f"decoder: {lab}", flush=True)
    outs = [d(grids) for d in decs]  # each (N,128,128,3)

    C = args.cell
    panels = 1 + len(decs)
    W, H = C * panels, C + 26
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))
    for i in range(len(F)):
        cells = [cv2.resize(F[i], (C, C))] + [cv2.resize(o[i], (C, C)) for o in outs]
        row = np.concatenate(cells, axis=1)
        bars = np.concatenate([label_bar(C, labels[j]) for j in range(panels)], axis=1)
        frame = np.concatenate([bars, row], axis=0)
        vw.write(frame[:, :, ::-1])  # RGB->BGR
    vw.release()
    print(f"wrote {args.out} ({W}x{H}, {len(F)} frames @ {args.fps}fps)", flush=True)


if __name__ == "__main__":
    main()
