#!/usr/bin/env python
"""Track B2: predict a patch-GRID subgoal, trained with DECODE-SPACE loss.

The user's target form: LMWM predicts a patch grid whose DECODE (via the faithful
frozen patch decoder from B1, ~2.7% vs pooled ~13%) is closest to the true future
frame. A generator head maps augin pooled input -> (din,16,16) grid; the frozen
patch decoder maps it to a 128x128 image; loss = L1 to the REAL next-medoid frame.

Baselines for the same decoded-image metric:
  (A) grid + latent loss   : predict grid, MSE to the true medoid grid (no decoder)
  (B) grid + decode loss   : predict grid, L1(D(grid), real medoid image)  <- proposed
  (ref) pooled decode head : from stage3_decode_subgoal (Track A), decoded via pooled dec
Reports decoded-image L1/cos to the real medoid frame + renders a comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from lmwm.data import split_indices  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from crave.decoding.decoder import make_decoder  # noqa: E402
from train_dinov3h_decoder import load_features, l2  # noqa: E402


class GridGen(nn.Module):
    """augin pooled vector -> (din,16,16) grid, via a small conv generator."""

    def __init__(self, din_in, din_grid=1280, hid=1024):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(din_in, hid), nn.GELU(), nn.LayerNorm(hid),
                                nn.Linear(hid, 512 * 4 * 4), nn.GELU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 384, 4, 2, 1), nn.BatchNorm2d(384), nn.GELU(),   # 4->8
            nn.ConvTranspose2d(384, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.GELU(),   # 8->16
            nn.Conv2d(256, din_grid, 3, 1, 1))
        self.din_grid = din_grid

    def forward(self, x):
        h = self.fc(x).view(-1, 512, 4, 4)
        return self.up(h)  # (B,din_grid,16,16)


def render_medoid_images(z, idx, dataset_root, camera, feature_dir, res):
    """For each pair, real next-medoid frame image (nearest frame in-episode to medoid latent)."""
    E, FR, Fb = load_features(Path(feature_dir))
    Fn = l2(Fb.astype(np.float32))
    med = z["next_medoid"][idx].astype(np.float32)
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    eps = z["episode_id"][idx].astype(np.int64)
    # per-episode medoid frame gidx
    tgt_g = np.zeros(len(idx), np.int64)
    for e in np.unique(eps):
        qloc = np.where(E == e)[0]
        sub = np.where(eps == e)[0]
        sims = Fn[qloc] @ med[sub].T   # (nframes, nsub)
        tgt_g[sub] = qloc[sims.argmax(0)]
    cs = int(json.loads((dataset_root / "meta/info.json").read_text())["chunks_size"])
    imgs = np.zeros((len(idx), res, res, 3), np.uint8)
    by_ep: dict[int, list[int]] = {}
    for k, gi in enumerate(tgt_g):
        by_ep.setdefault(int(E[gi]), []).append(k)
    for ep, ks in by_ep.items():
        cap = cv2.VideoCapture(str(dataset_root / f"videos/chunk-{ep // cs:03d}/{camera}/episode_{ep:06d}.mp4"))
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(FR[tgt_g[k]]))
            ok, fr = cap.read()
            if ok:
                imgs[k] = cv2.resize(fr[:, :, ::-1], (res, res))
        cap.release()
    return imgs, med


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--patch_dec", default="lmwm/checkpoints/patch_decoder/patch_dec.pt", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=16000)
    ap.add_argument("--n_val", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", default="lmwm/outputs/track_b2_grid/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.patch_dec, map_location="cpu", weights_only=False)
    din_grid = int(ck["din"]); Pg = int(ck["grid_P"]); R = int(ck["res"])
    D = make_decoder(din_grid, ck["dec"]).to(dev); D.load_state_dict(ck["model"]); D.eval()
    for p in D.parameters():
        p.requires_grad_(False)
    muT = torch.from_numpy(ck["mu"]).view(1, din_grid, 1, 1).to(dev)
    sdT = torch.from_numpy(ck["sd"]).view(1, din_grid, 1, 1).to(dev)

    def decode(grid):  # grid raw (B,din,16,16) -> image [-1,1]
        return D((grid - muT) / sdT)

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    ti, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    ti, vi = ti.numpy(), vi.numpy()
    med = z["next_medoid"].astype(np.float32); ok = np.linalg.norm(med, axis=1) > 1e-6
    rng = np.random.default_rng(0)
    ti = rng.choice(ti[ok[ti]], min(args.n_train, int(ok[ti].sum())), replace=False)
    vi = rng.choice(vi[ok[vi]], min(args.n_val, int(ok[vi].sum())), replace=False)

    din_in = z["current"].shape[1]
    Xt = torch.from_numpy(z["current"][ti].astype(np.float32))
    Xv = torch.from_numpy(z["current"][vi].astype(np.float32)).to(dev)

    print("rendering medoid target frames + encoding target grids ...", flush=True)
    tr_img, _ = render_medoid_images(z, ti, args.dataset_root, args.camera, args.feature_dir, R)
    va_img, _ = render_medoid_images(z, vi, args.dataset_root, args.camera, args.feature_dir, R)
    Yt = torch.from_numpy(tr_img.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).contiguous()
    Yv = torch.from_numpy(va_img.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2).to(dev)
    # target feature grids (real medoid frame -> encoder), for the latent-loss baseline
    # AND for the LaWM-style feature-space metric (per-token cosine + feature L1).
    enc = load_encoder("dinov3-h", device=str(dev))
    tr_tgt_grid = torch.from_numpy(enc.encode_grid(tr_img).astype(np.float32))
    va_tgt_grid = torch.from_numpy(enc.encode_grid(va_img).astype(np.float32))
    ntr = len(ti)

    def run(mode):
        torch.manual_seed(0)
        G = GridGen(din_in, din_grid).to(dev)
        opt = torch.optim.AdamW(G.parameters(), lr=3e-4, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
        for s in range(args.steps):
            bi = torch.randint(0, ntr, (64,))
            xb = Xt[bi].to(dev)
            grid = G(xb)
            if mode == "latent":
                loss = F.mse_loss(grid, tr_tgt_grid[bi].to(dev))
            elif mode == "lawm":  # LaWM recipe: feature-space smooth_l1 + (1 - per-token cosine)
                tg = tr_tgt_grid[bi].to(dev)
                rt = grid.flatten(2).transpose(1, 2); tt = tg.flatten(2).transpose(1, 2)  # (B,256,din)
                loss = F.smooth_l1_loss(rt, tt, beta=0.1) + (1.0 - F.cosine_similarity(rt, tt, dim=-1).mean())
            else:  # decode-space
                loss = F.l1_loss(decode(grid), Yt[bi].to(dev)) + 0.5 * ((decode(grid) - Yt[bi].to(dev)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        G.eval()
        il, ic, fcos, fl1 = [], [], [], []
        with torch.no_grad():
            for s in range(0, len(vi), 128):
                pg = G(Xv[s:s + 128])
                dp = decode(pg); dm = Yv[s:s + 128]
                il.append(F.l1_loss(dp, dm, reduction="none").mean((1, 2, 3)).cpu().numpy())
                ic.append(F.cosine_similarity(dp.flatten(1), dm.flatten(1), dim=1).cpu().numpy())
                tg = va_tgt_grid[s:s + 128].to(dev)
                rt = pg.flatten(2).transpose(1, 2); tt = tg.flatten(2).transpose(1, 2)     # (B,256,din)
                fcos.append(F.cosine_similarity(rt, tt, dim=-1).mean(1).cpu().numpy())      # LaWM per-token cos
                fl1.append((rt - tt).abs().mean((1, 2)).cpu().numpy())                      # LaWM feature L1
        return G, np.concatenate(il), np.concatenate(ic), np.concatenate(fcos), np.concatenate(fl1)

    res = {}; models = {}
    for mode in ["latent", "lawm", "decode"]:
        print(f"training grid ({mode}) ...", flush=True)
        G, il, ic, fcos, fl1 = run(mode); models[mode] = G
        res[f"grid_{mode}"] = {"decode_img_L1": round(float(il.mean()), 5), "decode_img_L1_std": round(float(il.std()), 5),
                               "decode_img_cos": round(float(ic.mean()), 4),
                               "lawm_feat_cos": round(float(fcos.mean()), 4), "lawm_feat_cos_std": round(float(fcos.std()), 4),
                               "lawm_feat_L1": round(float(fl1.mean()), 5)}
        r = res[f"grid_{mode}"]
        print(f"  grid_{mode:7s}| DECODE img_L1={r['decode_img_L1']:.4f} img_cos={r['decode_img_cos']:.4f} "
              f"| LaWM feat_cos={r['lawm_feat_cos']:.4f} feat_L1={r['lawm_feat_L1']:.4f}", flush=True)

    # render comparison: current | grid-latent decode | grid-decode decode | real medoid
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def d2img(t): return np.clip((t.detach().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)
    sel = np.linspace(0, len(vi) - 1, 6).astype(int)
    si = torch.from_numpy(sel).to(dev)
    with torch.no_grad():
        gl = decode(models["latent"](Xv[si])); gw = decode(models["lawm"](Xv[si])); gd = decode(models["decode"](Xv[si]))
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    ep_s = z["episode_id"][vi][sel]; t_s = z["t"][vi][sel]
    caps: dict[int, cv2.VideoCapture] = {}
    def frame(ep, t):
        if ep not in caps:
            caps[ep] = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(t)); okf, im = caps[ep].read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if okf else np.zeros((R, R, 3), np.uint8)
    titles = ["current (real)", "grid+latent-loss", "grid+LaWM-loss", "grid+decode-loss", "true medoid (real)"]
    fig, axes = plt.subplots(len(sel), 5, figsize=(5 * 2.2, len(sel) * 2.2))
    for i, k in enumerate(sel):
        imgs = [frame(int(ep_s[i]), int(t_s[i])), d2img(gl[i]), d2img(gw[i]), d2img(gd[i]), va_img[k]]
        for ci, im in enumerate(imgs):
            a = axes[i, ci]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(titles[ci], fontsize=8)
    for c in caps.values():
        c.release()
    fig.suptitle("patch-grid subgoal loss A/B | decode img L1: "
                 f"latent {res['grid_latent']['decode_img_L1']:.3f} / LaWM {res['grid_lawm']['decode_img_L1']:.3f} / decode {res['grid_decode']['decode_img_L1']:.3f}  ||  "
                 f"LaWM feat_cos: latent {res['grid_latent']['lawm_feat_cos']:.3f} / LaWM {res['grid_lawm']['lawm_feat_cos']:.3f} / decode {res['grid_decode']['lawm_feat_cos']:.3f}", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    vout = Path("lmwm/docs/assets/grid_decode_loss_compare.png"); vout.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(vout, dpi=120); plt.close(fig)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"saved {vout}")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
