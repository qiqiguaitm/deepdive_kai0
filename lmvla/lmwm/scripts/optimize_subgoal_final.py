#!/usr/bin/env python
"""FINAL subgoal model: attack the ~0.05 'predict-from-current' information loss from the PERCEPTION
side by enriching the deploy context with multi-view (top_head + hand_left + hand_right) + temporal
history, on top of the decided VAE latent-action head (kl~1e-2, LaWM-aligned).

Mechanism unchanged (forward-from-current): forward(cur_top_grid, code) -> future_top_grid, but the
inverse (teacher) and predm (deploy) now see a fused (V views x K frames) context instead of a single
grid -> more information about the current state -> better future prediction if the loss is perceptual.

Ablation via --views / --temporal: baseline (top_head,1) vs +multiview vs +temporal vs both(final).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from train_lawm_patch import load_index  # noqa: E402
from optimize_subgoal import build_pairs  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402


class ConvCode(nn.Module):
    """(Cin,16,16) -> code(cd). Stride-2 convs + global pool + linear (scales to any input channels)."""
    def __init__(self, cin, cd, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU(),
            nn.Conv2d(hid, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU())
        self.head = nn.Linear(hid, cd); self.ln = nn.LayerNorm(cd)

    def forward(self, x):
        return self.ln(self.head(self.net(x).mean((2, 3))))


class Fwd(nn.Module):
    """(cur_top_grid, code) -> future_top_grid. Spatial base = current top-head."""
    def __init__(self, din, cd, hid=512):
        super().__init__()
        self.proj = nn.Conv2d(din + cd, hid, 3, 1, 1)
        self.body = nn.Sequential(nn.GroupNorm(8, hid), nn.GELU(), nn.Conv2d(hid, hid, 3, 1, 1),
                                  nn.GroupNorm(8, hid), nn.GELU(), nn.Conv2d(hid, hid, 3, 1, 1),
                                  nn.GroupNorm(8, hid), nn.GELU())
        self.out = nn.Conv2d(hid, din, 3, 1, 1)

    def forward(self, gt, code):
        c = code[:, :, None, None].expand(-1, -1, gt.shape[2], gt.shape[3])
        return self.out(self.body(self.proj(torch.cat([gt, c], 1))))


def read_frames(dataset_root, cs, need):
    """need: set of (ep, frame, cam) -> {(ep,frame,cam): img256 uint8}."""
    by = {}
    for (ep, fr, cam) in need:
        by.setdefault((cam, ep), []).append(fr)
    out = {}
    for (cam, ep), frs in by.items():
        camdir = cam if cam.startswith("observation") else f"observation.images.{cam}"
        cap = cv2.VideoCapture(str(dataset_root / f"videos/chunk-{ep // cs:03d}/{camdir}/episode_{ep:06d}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9
        for fr in set(frs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(fr, 0), n - 1)); ok, im = cap.read()
            if ok:
                out[(ep, fr, cam)] = cv2.resize(im[:, :, ::-1], (256, 256))
        cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nearfuture", "milestone"], default="milestone")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--code_dim", type=int, default=128)
    ap.add_argument("--views", default="top_head", help="csv: top_head,hand_left,hand_right")
    ap.add_argument("--temporal", type=int, default=1, help="K frames (current + K-1 past)")
    ap.add_argument("--temporal_stride", type=int, default=15, help="frames between temporal samples (0.5s@30fps)")
    ap.add_argument("--kl_weight", type=float, default=1e-2)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    views = args.views.split(","); V = len(views); K = args.temporal
    tag = f"{args.mode}_V{V}_K{K}_cd{args.code_dim}"
    out = Path(args.out) if args.out else Path(f"lmwm/outputs/subgoal_final/{tag}.json")

    E, FR, Fn = load_index(args.feature_dir)
    proto = np.load("lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")["prototype_table"].astype(np.float32)
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    tr, va = build_pairs(E, FR, Fn, proto, args.mode, args.horizon, val_eps, args.seed)
    tr = tr[:args.n_train]; va = va[:args.n_val]
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    print(f"[{tag}] {len(tr)}+{len(va)} pairs, V={V} K={K} stride={args.temporal_stride}", flush=True)

    # gather needed frames: context = views x K past frames of current; future = top_head of target
    def ctx_samples(gidx):
        ep, fr = int(E[gidx]), int(FR[gidx])
        return [(ep, fr - k * args.temporal_stride, cam) for cam in views for k in range(K)]
    need = set()
    for a, b in tr + va:
        need.update(ctx_samples(a)); need.add((int(E[b]), int(FR[b]), "top_head"))
    need = sorted(need)
    print(f"[{tag}] reading+encoding {len(need)} unique frames ...", flush=True)
    imgs = read_frames(args.dataset_root, cs, need)
    keys = [k for k in need if k in imgs]; kidx = {k: i for i, k in enumerate(keys)}
    enc = load_encoder("dinov3-h", device=dev)
    arr = np.stack([imgs[k] for k in keys])
    G = enc.encode_grid(arr).astype(np.float32); din = G.shape[1]
    gmu, gsd = G.mean(), G.std() + 1e-6
    GZ = torch.from_numpy(((G - gmu) / gsd).astype(np.float16))

    def stack_ctx(pairs):
        cx, ct, ft = [], [], []
        for a, b in pairs:
            cs_ = [kidx[k] for k in ctx_samples(a)]        # V*K context grid indices
            cx.append(cs_); ct.append(kidx[(int(E[a]), int(FR[a]), "top_head")])
            ft.append(kidx[(int(E[b]), int(FR[b]), "top_head")])
        return np.array(cx), np.array(ct), np.array(ft)
    trc, trt, trf = stack_ctx(tr); vac, vat, vaf = stack_ctx(va)

    predm = ConvCode(V * K * din, args.code_dim).to(dev)
    inv = ConvCode((V * K + 1) * din, args.code_dim).to(dev)
    fwd = Fwd(din, args.code_dim).to(dev)
    cd = args.code_dim
    vae_i = nn.Linear(cd, 2 * cd).to(dev); vae_p = nn.Linear(cd, 2 * cd).to(dev)

    def reparam(head, h, sample):
        mu, lv = head(h).chunk(2, -1); lv = lv.clamp(-8, 8)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp() if sample else mu
        return z, mu, lv, -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()

    def gctx(idx):                                          # (B, V*K*din, 16,16)
        g = GZ[torch.from_numpy(idx)].to(dev).float()       # (B,VK,din,16,16)
        return g.reshape(g.shape[0], -1, 16, 16)

    def gtop(idx):
        return GZ[torch.from_numpy(idx)].to(dev).float()    # (B,din,16,16)

    o1 = torch.optim.AdamW(list(inv.parameters()) + list(fwd.parameters()) + list(vae_i.parameters()), lr=2e-4, weight_decay=1e-5)
    o2 = torch.optim.AdamW(list(predm.parameters()) + list(vae_p.parameters()), lr=2e-4, weight_decay=1e-5)
    n_params = sum(p.numel() for m in [predm, inv, fwd, vae_i, vae_p] for p in m.parameters())
    print(f"[{tag}] params {n_params/1e6:.1f}M; training ...", flush=True)
    for step in range(args.steps):
        sel = np.random.randint(0, len(trc), 24)
        ctx = gctx(trc[sel]); cur = gtop(trt[sel]); fut = gtop(trf[sel])
        z, _, _, kl = reparam(vae_i, inv(torch.cat([ctx, cur], 1)), True)
        l1 = F.smooth_l1_loss(fwd(cur, z), fut, beta=1.0) + args.kl_weight * kl
        o1.zero_grad(); l1.backward(); o1.step()
        zp, _, _, klp = reparam(vae_p, predm(ctx), True)
        l2 = F.smooth_l1_loss(fwd(cur, zp), fut, beta=1.0) + args.kl_weight * klp
        o2.zero_grad(); l2.backward(); o2.step()
    for m in [predm, inv, fwd, vae_i, vae_p]:
        m.eval()

    def cos(a, b): return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    f = lambda x: (x.detach().cpu().numpy() * gsd + gmu).reshape(len(x), -1)
    co, cdp, cbest, cp = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(vac), 128):
            ctx = gctx(vac[s:s + 128]); cur = gtop(vat[s:s + 128]); fut = gtop(vaf[s:s + 128]); gtr = f(fut)
            _, mi, _, _ = reparam(vae_i, inv(torch.cat([ctx, cur], 1)), False)
            _, mp, lp, _ = reparam(vae_p, predm(ctx), False)
            co.append(cos(f(fwd(cur, mi)), gtr)); cdp.append(cos(f(fwd(cur, mp)), gtr)); cp.append(cos(f(cur), gtr))
            best = None
            for _ in range(8):
                ck = cos(f(fwd(cur, mp + torch.randn_like(mp) * (0.5 * lp).exp())), gtr)
                best = ck if best is None else np.maximum(best, ck)
            cbest.append(best)
    res = {"mode": args.mode, "views": views, "V": V, "temporal_K": K, "stride": args.temporal_stride,
           "code_dim": cd, "kl_weight": args.kl_weight, "params_M": round(n_params / 1e6, 1),
           "n_train": len(tr), "n_val": len(va),
           "oracle_grid_cos": round(float(np.concatenate(co).mean()), 4),
           "deploy_grid_cos": round(float(np.concatenate(cdp).mean()), 4),
           "deploy_bestof8_cos": round(float(np.concatenate(cbest).mean()), 4),
           "persistence_grid_cos": round(float(np.concatenate(cp).mean()), 4),
           "baseline_V1K1_deploy": 0.720 if args.mode == "milestone" else 0.698}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
