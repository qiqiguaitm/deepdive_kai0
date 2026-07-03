#!/usr/bin/env python
"""Native-speed (stride-1, 30Hz) video of the PRODUCTION milestone+1 model.

Left: real episode frames (native fps). Right (stacked):
  top    = PREDICTED milestone+1 subgoal latent -> decode
  bottom = TRUE milestone+1 medoid latent -> encode->decode (decoder ceiling)
Both via the pooled DINOv3-H decoder. Per-frame augin+ is rebuilt (pooled + prev-milestone
latent + current-milestone latent + proprio) and fed to the 5-member ensemble.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from train_prod_milestone import ProdNet  # noqa: E402
from train_dinov3h_decoder import PooledDecoder, l2  # noqa: E402
from make_episode_native_video import state_stats, label  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=1819)
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--members", default="lmwm/checkpoints/prod_milestone/member_*.pt")
    ap.add_argument("--decoder", default="lmwm/checkpoints/dinov3h_decoder/dec.pt", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--max_frames", type=int, default=900)
    ap.add_argument("--out", default="lmwm/docs/assets/prod_milestone_native.mp4", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.decoder, map_location="cpu"); R = int(ck["res"])
    D = PooledDecoder(din=int(ck["din"]), res=R).to(dev); D.load_state_dict(ck["model"]); D.eval()
    def decode(lat):
        with torch.no_grad():
            o = D(torch.from_numpy(l2(np.atleast_2d(lat).astype(np.float32))).to(dev)).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)

    proto = np.load(args.graph_npz)["prototype_table"].astype(np.float32); START = len(proto)
    enc = load_encoder("dinov3-h", device=str(dev))
    paths = sorted(glob.glob(args.members))
    din = 1280 + 1280 + 1280 + 14
    models = []
    for p in paths:
        c = torch.load(p, map_location="cpu"); m = ProdNet(din, len(proto)).to(dev); m.load_state_dict(c["model"]); m.eval(); models.append(m)

    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    vid = args.dataset_root / f"videos/chunk-{args.episode // cs:03d}/{args.camera}/episode_{args.episode:06d}.mp4"
    cap = cv2.VideoCapture(str(vid)); fps = cap.get(cv2.CAP_PROP_FPS); out_fps = round(fps) if fps and fps > 1 else 30
    enc_frames, disp = [], []
    while len(enc_frames) < args.max_frames:
        okf, im = cap.read()
        if not okf:
            break
        rgb = im[:, :, ::-1]; enc_frames.append(cv2.resize(rgb, (256, 256))); disp.append(cv2.resize(rgb, (R * 2, R * 2)))
    cap.release()
    N = len(enc_frames); enc_frames = np.stack(enc_frames)
    print(f"ep{args.episode}: {N} frames @ {out_fps}fps", flush=True)

    pooled = enc.encode_pooled(enc_frames).astype(np.float32); pn = l2(pooled)
    seq = (pn @ proto.T).argmax(1)
    ch = np.where(np.diff(seq) != 0)[0] + 1; st = np.concatenate([[0], ch]); en = np.concatenate([ch, [N]])
    fr2s = np.zeros(N, np.int64); stage_m = []; stage_med = []
    for si, (s, e) in enumerate(zip(st, en)):
        fr2s[s:e] = si; stage_m.append(int(seq[s]))
        stage_med.append(pn[s + int((pn[s:e] @ proto[seq[s]]).argmax())])       # episode-medoid latent (L2)
    prev_lat = np.stack([proto[stage_m[fr2s[f] - 1]] if fr2s[f] >= 1 else np.zeros(1280, np.float32) for f in range(N)])
    cur_lat = np.stack([proto[stage_m[fr2s[f]]] for f in range(N)])
    next_med = np.stack([stage_med[min(fr2s[f] + 1, len(st) - 1)] for f in range(N)])   # true milestone+1 latent

    pq = args.dataset_root / f"data/chunk-{args.episode // cs:03d}/episode_{args.episode:06d}.parquet"
    arr = np.stack(pd.read_parquet(pq, columns=["observation.state"])["observation.state"].to_numpy()).astype(np.float32)
    mu_s, sd_s = state_stats(args.pairs, args.dataset_root, cs)
    proprio = (arr[np.clip(np.arange(N), 0, len(arr) - 1)] - mu_s) / sd_s

    aug = np.concatenate([pooled, prev_lat, cur_lat, proprio.astype(np.float32)], 1)  # (N,3854)
    protos = None
    with torch.no_grad():
        X = torch.from_numpy(aug.astype(np.float32)).to(dev)
        for m in models:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, pr = m(X)
            g = F.normalize(pr.float(), -1).cpu().numpy(); protos = g if protos is None else protos + g
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8

    pred_dec = [decode(protos[s:s + 256]) for s in range(0, N, 256)]; pred_dec = np.concatenate(pred_dec)
    # ceiling: true next-medoid decode, dedup per unique stage-medoid
    uniq_ns, inv = np.unique(next_med.round(3), axis=0, return_inverse=True)
    ceil_u = decode(uniq_ns);

    BIG = R * 2; RW = 230; W = BIG + 8 + RW; H = 26 + BIG + 4
    def rlabel(img, t): return label(cv2.resize(img, (RW, RW)), t)
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for f in range(N):
        left = label(disp[f], f"ep{args.episode} frame {f} (real, native)")
        rt = rlabel(pred_dec[f], "PRED milestone+1 -> decode")
        rb = rlabel(ceil_u[inv[f]], "TRUE milestone+1 enc->decode")
        right = cv2.resize(np.vstack([rt, rb]), (RW, left.shape[0]))
        canvas = np.hstack([left, np.full((left.shape[0], 8, 3), 20, np.uint8), right])
        vw.write(cv2.cvtColor(canvas[:H, :W], cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"saved {args.out} | {N} frames @ {out_fps}fps")


if __name__ == "__main__":
    main()
