#!/usr/bin/env python
"""Show the CURRENT SigLIP decoder's effect: per episode frame, [ real | SigLIP-encode -> decode ].
Isolates DECODER reconstruction quality (real frames in, no prediction) in motion.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
import cv2  # noqa: E402
from train_lawm_patch import load_index, read_imgs  # noqa: E402
from _siglip_bigvision import SiglipBigVision  # noqa: E402
from crave.decoding.decoder import make_decoder  # noqa: E402

PI05_NPZ = "/vePFS/tim/workspace/openpi_cache/paligemma_weights/pt_224.npz"
PI05_NPZ_GF3 = "/vePFS-North-E/vis_robot/openpi_cache/paligemma_weights/pt_224.npz"


def bar(w, text, h=22):
    b = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(b, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=8)
    ap.add_argument("--dec_ckpt", default="lmwm/checkpoints/siglip_decoder/dec_big_enc224_out128.pt")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="lmwm/outputs/decode_recon_ep8.mp4")
    ap.add_argument("--pi05_npz", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    npz = args.pi05_npz or (PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3)

    E, FR, _ = load_index(args.feature_dir)
    loc = np.where(E == args.episode)[0]
    if len(loc) == 0:
        raise SystemExit(f"episode {args.episode} not in index")
    order = loc[np.argsort(FR[loc])]
    dc = torch.load(args.dec_ckpt, map_location="cpu", weights_only=False)
    R = dc["res"]
    encimg, disp = read_imgs(args.dataset_root, args.camera, E, FR, order, 224, R)
    print(f"ep{args.episode}: {len(order)} frames; SigLIP encoding ...", flush=True)
    enc = SiglipBigVision(npz, device=dev)
    grids = enc.encode_grid(encimg, bs=32); din = grids.shape[1]
    mu = np.asarray(dc["mu"]); sd = np.asarray(dc["sd"])
    dec = make_decoder(din, dc["dec"]).to(dev); dec.load_state_dict(dc["model"]); dec.eval()
    print(f"decoder {dc['dec']}@{R} (val_L1={dc.get('val_L1'):.4f})", flush=True)

    Xn = torch.from_numpy(((grids - mu) / sd).astype(np.float32))
    vw = cv2.VideoWriter(str(REPO / args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (R * 2, R + 22))
    with torch.no_grad():
        for s in range(0, len(order), 128):
            o = dec(Xn[s:s + 128].to(dev)).cpu().numpy()
            rec = np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
            for k in range(len(rec)):
                row = np.concatenate([disp[s + k], rec[k]], axis=1)[:, :, ::-1]
                lab = np.concatenate([bar(R, "REAL frame"), bar(R, "SigLIP encode->decode")], axis=1)
                vw.write(np.concatenate([lab, row], axis=0))
    vw.release()
    print(f"saved {REPO / args.out} ({len(order)} frames)", flush=True)


if __name__ == "__main__":
    main()
