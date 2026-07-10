#!/usr/bin/env python
"""L1: patch-token INPUT + small transformer vs the pooled-MLP baseline.

Pooling discards spatial layout; feeding the 256 DINOv3-H patch tokens to a small
transformer lets it use that layout for the INPUT (unlike the failed high-dim
patch-grid OUTPUT). Predicts next milestone (CE) + episode-medoid subgoal (regress).
Trained on the SAME subset as a pooled-MLP baseline (frame+prev+state) for a fair
A/B. Reports mean + variance for both.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from lmwm.data import split_indices  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402


def read_enc(dataset_root, camera, eps, ts, res):
    cs = int(json.loads((dataset_root / "meta/info.json").read_text())["chunks_size"])
    out = np.zeros((len(eps), res, res, 3), np.uint8)
    by_ep: dict[int, list[int]] = {}
    for k in range(len(eps)):
        by_ep.setdefault(int(eps[k]), []).append(k)
    for ep, ks in by_ep.items():
        cap = cv2.VideoCapture(str(dataset_root / f"videos/chunk-{ep // cs:03d}/{camera}/episode_{ep:06d}.mp4"))
        for k in ks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts[k]))
            ok, fr = cap.read()
            if ok:
                out[k] = cv2.resize(fr[:, :, ::-1], (res, res))
        cap.release()
    return out


class PooledMLP(nn.Module):
    def __init__(self, din, num_m, ld=1280, hid=512):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid),
                                   nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid))
        self.cls = nn.Linear(hid, num_m)
        self.proto = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, ld))

    def forward(self, pooled, prev, state):
        h = self.trunk(torch.cat([pooled, prev, state], -1))
        return self.cls(h), F.normalize(self.proto(h), dim=-1)


class PatchTransformer(nn.Module):
    def __init__(self, din, num_m, n_prev, ld=1280, d=384, layers=3, heads=6):
        super().__init__()
        self.tok = nn.Linear(din, d)
        self.pos = nn.Parameter(torch.zeros(1, 256, d))
        self.prev = nn.Linear(n_prev, d); self.state = nn.Linear(14, d)
        self.q = nn.Parameter(torch.zeros(1, 2, d))  # [milestone, subgoal] queries
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.0, batch_first=True, activation="gelu", norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.cls = nn.Linear(d, num_m)
        self.proto = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, ld))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.q, std=0.02)

    def forward(self, grid, prev, state):
        B = grid.shape[0]
        t = self.tok(grid) + self.pos
        extra = torch.stack([self.prev(prev), self.state(state)], 1)
        q = self.q.expand(B, -1, -1)
        x = torch.cat([q, extra, t], 1)
        x = self.tr(x)
        return self.cls(x[:, 0]), F.normalize(self.proto(x[:, 1]), dim=-1)


def stats(logits_or_probs, protos, y, med, is_prob=False):
    p = logits_or_probs if is_prob else F.softmax(torch.from_numpy(logits_or_probs), -1).numpy()
    per = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    order = np.argsort(-p, 1); rank = (order == y[:, None]).argmax(1)
    c = (protos * med).sum(1)
    return {"top1": round(float((rank == 0).mean()), 4), "top5": round(float((rank < 5).mean()), 4),
            "nll": round(float(per.mean()), 4), "nll_std": round(float(per.std()), 4),
            "cos": round(float(c.mean()), 4), "cos_std": round(float(c.std()), 4),
            "cos_lt07": round(float((c < 0.7).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=18000)
    ap.add_argument("--n_val", type=int, default=2500)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default="lmwm/outputs/lever_patch_token/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"]); num_m = int(np.load(z["graph_npz"].item() if "graph_npz" in z.files else "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")["transition_probs"].shape[0]) if False else 37
    _, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    vi = vi.numpy(); itr = np.setdiff1d(np.arange(n), vi)
    rng = np.random.default_rng(0)
    med_ok = np.linalg.norm(z["next_medoid"], axis=1) > 1e-6
    itr = itr[med_ok[itr]]; vi = vi[med_ok[vi]]
    tr = rng.choice(itr, min(args.n_train, len(itr)), replace=False)
    va = rng.choice(vi, min(args.n_val, len(vi)), replace=False)
    sel = np.concatenate([tr, va])

    cur = z["current"][sel].astype(np.float32)         # 1332 = pooled(1280)+prev(38)+state(14)
    pooled = cur[:, :1280]; prev = cur[:, 1280:1280+38]; state = cur[:, -14:]
    y = z["future_milestone"][sel].astype(np.int64)
    med = z["next_medoid"][sel].astype(np.float32); med /= np.linalg.norm(med, axis=1, keepdims=True) + 1e-8
    eps = z["episode_id"][sel].astype(np.int64); ts = z["t"][sel].astype(np.int64)

    print(f"encoding {len(sel)} patch grids ...", flush=True)
    enc_imgs = read_enc(args.dataset_root, args.camera, eps, ts, 256)
    enc = load_encoder("dinov3-h", device=str(dev))
    grid = enc.encode_grid(enc_imgs).astype(np.float32)         # (N,1280,16,16)
    grid = grid.reshape(len(sel), 1280, 256).transpose(0, 2, 1)  # (N,256,1280)

    ntr = len(tr)
    def to(a): return torch.from_numpy(a)
    Gg, Pp, Pr, St = to(grid), to(pooled), to(prev), to(state)
    Md, Yy = to(med), torch.from_numpy(y)
    n_prev = prev.shape[1]

    def train(model, use_grid):
        model = model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
        for s in range(args.steps):
            bi = torch.randint(0, ntr, (256,))
            args_in = (Gg[bi].to(dev), Pr[bi].to(dev), St[bi].to(dev)) if use_grid else (Pp[bi].to(dev), Pr[bi].to(dev), St[bi].to(dev))
            lg, pr = model(*args_in)
            loss = F.cross_entropy(lg, Yy[bi].to(dev)) + 5.0 * F.smooth_l1_loss(pr, Md[bi].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        model.eval()
        lgs, prs = [], []
        with torch.no_grad():
            for s in range(ntr, len(sel), 512):
                idx = slice(s, min(s + 512, len(sel)))
                args_in = (Gg[idx].to(dev), Pr[idx].to(dev), St[idx].to(dev)) if use_grid else (Pp[idx].to(dev), Pr[idx].to(dev), St[idx].to(dev))
                lg, pr = model(*args_in); lgs.append(lg.cpu().numpy()); prs.append(pr.cpu().numpy())
        return np.concatenate(lgs), np.concatenate(prs)

    print("training pooled-MLP baseline ...", flush=True)
    lg_b, pr_b = train(PooledMLP(1280 + n_prev + 14, num_m), use_grid=False)
    print("training patch-token transformer ...", flush=True)
    lg_t, pr_t = train(PatchTransformer(1280, num_m, n_prev), use_grid=True)

    yv = y[ntr:]; mv = med[ntr:]
    res = {"n_train": ntr, "n_val": len(sel) - ntr,
           "pooled_mlp_baseline": stats(lg_b, pr_b, yv, mv),
           "patch_token_transformer": stats(lg_t, pr_t, yv, mv)}
    args.out = Path(args.out); args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
