#!/usr/bin/env python
"""Objectively rank the flow-decoder configs on a COMMON held-out set (gf3).
For each ckpt: sample self-recon on the same latents, measure re-encode cos (gated
DINOv3-H), sharpness, and pixel L1 to the real frame. Saves a table + montage.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flow_decoder_gf3 import UNet  # noqa: E402
from dinov3h_gated import DINOv3HGated  # noqa: E402


@torch.no_grad()
def sample(net, lat, res, ode_steps, dev):
    x = torch.randn(len(lat), 3, res, res, device=dev)
    for k in range(ode_steps):
        t = torch.full((len(lat),), k / ode_steps, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = net(x, t, lat)
        x = x + (1.0 / ode_steps) * v.float()
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="temp/lmwm_p0/flow_pairs_n50000_r128.npz")
    ap.add_argument("--ckpt_glob", default="temp/lmwm_p0/flow_*.pt")
    ap.add_argument("--dino_path", default="temp/lmwm_p0/dinov3h")
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--ode_steps", type=int, default=25)
    ap.add_argument("--out_json", default="temp/lmwm_p0/flow_eval.json")
    ap.add_argument("--out_png", default="temp/lmwm_p0/flow_eval.png")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    d = np.load(args.cache); lat = d["lat"].astype(np.float32); tgt = d["tgt"]
    lat /= np.linalg.norm(lat, axis=1, keepdims=True) + 1e-8
    he = np.arange(len(lat) - args.n_eval, len(lat))
    L = torch.from_numpy(lat[he]).to(dev); real = tgt[he]
    reenc = DINOv3HGated(args.dino_path, device=str(dev))
    def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

    res = tgt.shape[1]
    ckpts = [p for p in sorted(glob.glob(args.ckpt_glob)) if "pairs" not in p and Path(p).stem.startswith("flow_")]
    results = {}; samples = {}
    for p in ckpts:
        ck = torch.load(p, map_location="cpu")
        net = UNet(1280, ck["base"]).to(dev); net.load_state_dict(ck["model"]); net.eval()
        imgs = []
        for s in range(0, len(he), 64):
            x = sample(net, L[s:s + 64], res, args.ode_steps, dev)
            imgs.append(np.clip((x.cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8))
        img = np.concatenate(imgs)
        # re-encode cos
        cos = []
        for s in range(0, len(img), 64):
            re = F.normalize(reenc.encode_pooled_tensor(torch.from_numpy(img[s:s + 64].astype(np.float32) / 255).permute(0, 3, 1, 2).to(dev)), dim=-1)
            cos.append((re.float().cpu().numpy() * lat[he][s:s + 64]).sum(1))
        cos = np.concatenate(cos)
        l1 = float(np.abs(real.astype(float) - img.astype(float)).mean() / 255)
        results[Path(p).stem] = {"step": int(ck.get("step", 0)), "base": ck["base"],
                                 "reencode_cos": round(float(cos.mean()), 4),
                                 "sharpness": round(float(np.mean([sharp(x) for x in img[:64]])), 0),
                                 "pixel_L1": round(l1, 4)}
        samples[Path(p).stem] = img[:8]
        print(f"{Path(p).stem}: step {ck.get('step')} base {ck['base']} | reencode_cos={results[Path(p).stem]['reencode_cos']} "
              f"sharp={results[Path(p).stem]['sharpness']:.0f} L1={l1:.3f}", flush=True)

    results["_real_sharpness"] = round(float(np.mean([sharp(x) for x in real[:64]])), 0)
    best = max((k for k in results if k != "_real_sharpness"), key=lambda k: results[k]["reencode_cos"])
    results["_best_by_reencode_cos"] = best
    Path(args.out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = list(samples.keys()); rows = 1 + len(names)
    fig, ax = plt.subplots(rows, 8, figsize=(8 * 1.5, rows * 1.7))
    for j in range(8):
        ax[0, j].imshow(real[j]); ax[0, j].axis("off")
    ax[0, 0].set_ylabel("real", fontsize=8)
    for r, nm in enumerate(names):
        for j in range(8):
            ax[r + 1, j].imshow(samples[nm][j]); ax[r + 1, j].axis("off")
        ax[r + 1, 0].set_ylabel(nm.replace("flow_", ""), fontsize=8)
    fig.suptitle("flow decoder configs (common held-out) | " + " | ".join(f"{k.replace('flow_','')}:cos{results[k]['reencode_cos']}" for k in names), fontsize=8)
    fig.tight_layout(); fig.savefig(args.out_png, dpi=110); plt.close(fig)
    print(f"BEST by reencode_cos: {best} -> {results[best]}", flush=True)


if __name__ == "__main__":
    main()
