#!/usr/bin/env python
"""TRUE deployment test: predict the future patch-grid from the CURRENT grid only
(no future peek), on held-out episodes, then decode.

Trains a grid predictor P(g_t) -> g_future_next-stage-medoid on train episodes,
evaluates on held-out episodes, and visualizes one episode:
  current | predicted-future decoded | persistence decoded | real future
Compares against: oracle reconstruction 5.6% (uses future), true-grid decode 2.7%
(ceiling), pooled subgoal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from train_lawm_patch import load_index, read_imgs  # noqa: E402
from crave.encoders import load_encoder  # noqa: E402
from crave.decoding.decoder import train_dec  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def stage_transitions(E, FR, Fn, proto, ep):
    loc = np.where(E == ep)[0]; order = loc[np.argsort(FR[loc])]
    seq = (Fn[order] @ proto.T).argmax(1)
    ch = np.where(np.diff(seq) != 0)[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
    reps = []
    for s, e in zip(st, en):
        m = int(seq[s]); sub = order[s:e]; med = sub[(Fn[sub] @ proto[m]).argmax()]
        reps.append((int(order[e - 1]), int(med), m))
    return [(reps[i][0], reps[i + 1][1], reps[i][2], reps[i + 1][2]) for i in range(len(reps) - 1)]


class GridPredictor(nn.Module):
    """current grid -> future grid (predict residual on top of current)."""
    def __init__(self, din, hid=512):
        super().__init__()
        self.proj = nn.Conv2d(din, hid, 3, 1, 1)
        self.body = nn.Sequential(
            nn.GroupNorm(8, hid), nn.GELU(), nn.Conv2d(hid, hid, 3, 1, 1),
            nn.GroupNorm(8, hid), nn.GELU(), nn.Conv2d(hid, hid, 3, 1, 1),
            nn.GroupNorm(8, hid), nn.GELU(), nn.Conv2d(hid, hid, 3, 1, 1),
            nn.GroupNorm(8, hid), nn.GELU(),
        )
        self.out = nn.Conv2d(hid, din, 3, 1, 1)

    def forward(self, gt):
        return gt + self.out(self.body(self.proj(gt)))  # residual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=793)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out_dir", default="lmwm/docs/assets", type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    dev = "cuda"

    E, FR, Fn = load_index(args.feature_dir)
    proto = np.load("lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")["prototype_table"].astype(np.float32)
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())

    tr, va = [], []
    for ep in eps:
        if ep == args.episode:
            continue
        tgt = va if ep in val_eps else tr
        tgt.extend([(a, b) for a, b, _, _ in stage_transitions(E, FR, Fn, proto, ep)])
    rng.shuffle(tr); rng.shuffle(va); tr = tr[:args.n_train]; va = va[:args.n_val]
    ep_trans = stage_transitions(E, FR, Fn, proto, args.episode)

    uniq = sorted(set([g for p in tr + va for g in p] + [g for t in ep_trans for g in (t[0], t[1])]))
    u2k = {g: k for k, g in enumerate(uniq)}
    print(f"{len(tr)} train + {len(va)} val transitions, ep{args.episode} {len(ep_trans)}, {len(uniq)} frames", flush=True)
    enc_imgs, tgt_imgs = read_imgs(args.dataset_root, args.camera, E, FR, np.array(uniq), 256, 128)
    enc = load_encoder("dinov3-h", device=dev)
    grids = enc.encode_grid(enc_imgs).astype(np.float32); din = grids.shape[1]
    gmu, gsd = grids.mean(), grids.std() + 1e-6
    gz = ((grids - gmu) / gsd).astype(np.float32)

    print("training patch decoder + grid predictor ...", flush=True)
    decode = train_dec(grids, tgt_imgs, din, dec="small", epochs=50, device=dev)
    GZ = torch.from_numpy(gz)
    tra = torch.from_numpy(np.array([u2k[c] for c, _ in tr])); trb = torch.from_numpy(np.array([u2k[n] for _, n in tr]))
    P = GridPredictor(din).to(dev)
    opt = torch.optim.AdamW(P.parameters(), lr=2e-4, weight_decay=1e-5)
    for step in range(args.steps):
        sel = torch.randint(0, len(tra), (32,))
        gt = GZ[tra[sel]].to(dev); gf = GZ[trb[sel]].to(dev)
        loss = F.smooth_l1_loss(P(gt), gf, beta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    P.eval()

    def cos(a, b): return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    vaa = np.array([u2k[c] for c, _ in va]); vab = np.array([u2k[n] for _, n in va])
    cr, cp, li_pred, li_persist, li_true = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, len(vaa), 256):
            gt = GZ[vaa[s:s+256]].to(dev); gf_std = gz[vab[s:s+256]]
            gp = P(gt).cpu().numpy()
            gh = (gp * gsd + gmu).reshape(len(gp), -1); gtr = (gf_std * gsd + gmu).reshape(len(gf_std), -1)
            gc = (gt.cpu().numpy() * gsd + gmu).reshape(len(gt), -1)
            cr.append(cos(gh, gtr)); cp.append(cos(gc, gtr))
            real = tgt_imgs[vab[s:s+256]]
            li_pred.append(np.abs(real.astype(float) - decode((gp*gsd+gmu).astype(np.float32)).astype(float)).mean((1,2,3)))
            li_persist.append(np.abs(real.astype(float) - decode((gt.cpu().numpy()*gsd+gmu).astype(np.float32)).astype(float)).mean((1,2,3)))
            li_true.append(np.abs(real.astype(float) - decode(gf_std*gsd+gmu).astype(float)).mean((1,2,3)))
    summary = {
        "n_train": len(tr), "n_val": len(va), "mode": "TRUE deployment: predict future grid from current only (no future peek), held-out",
        "predict_grid_cos": round(float(np.concatenate(cr).mean()), 4),
        "persistence_grid_cos": round(float(np.concatenate(cp).mean()), 4),
        "predict_decode_L1_frac": round(float(np.concatenate(li_pred).mean())/255, 4),
        "persistence_decode_L1_frac": round(float(np.concatenate(li_persist).mean())/255, 4),
        "true_grid_decode_L1_frac": round(float(np.concatenate(li_true).mean())/255, 4),
        "ref_oracle_recon_L1": 0.056, "ref_pooled_L1": 0.062,
    }
    Path("lmwm/outputs/lawm_patch").mkdir(parents=True, exist_ok=True)
    Path("lmwm/outputs/lawm_patch/deploy.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    # viz episode
    def dg(gstd): return decode((gstd*gsd+gmu).astype(np.float32))[0]
    rows = []
    for cg, ng, cm, nm in ep_trans:
        gt = GZ[u2k[cg]:u2k[cg]+1].to(dev)
        with torch.no_grad(): gp = P(gt).cpu().numpy()
        rows.append({"cur": tgt_imgs[u2k[cg]], "pred": dg(gp), "real": tgt_imgs[u2k[ng]], "cm": cm, "nm": nm})
    sel = rows if len(rows) <= 10 else [rows[i] for i in np.linspace(0, len(rows)-1, 10).astype(int)]
    titles = ["current", "PREDICTED future (deploy)", "real future"]
    fig, ax = plt.subplots(len(sel), 3, figsize=(3*2.2, len(sel)*2.2))
    if len(sel) == 1: ax = ax[None, :]
    for ri, r in enumerate(sel):
        for ci, im in enumerate([r["cur"], r["pred"], r["real"]]):
            ax[ri, ci].imshow(im); ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
            if ri == 0: ax[ri, ci].set_title(titles[ci], fontsize=9)
        ax[ri, 0].set_ylabel(f"m{r['cm']}->m{r['nm']}", fontsize=7)
    fig.suptitle(f"ep{args.episode}: TRUE deployment prediction (no future peek) | pred cos={summary['predict_grid_cos']} L1={summary['predict_decode_L1_frac']}", fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(args.out_dir / f"ep{args.episode}_deploy_predict.png", dpi=115); plt.close(fig)
    print(f"saved {args.out_dir}/ep{args.episode}_deploy_predict.png")


if __name__ == "__main__":
    main()
