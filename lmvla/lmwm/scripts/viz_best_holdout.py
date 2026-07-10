#!/usr/bin/env python
"""Visualize the BEST model's predictions on a held-out (out-of-training) episode.

Driven by the augin pair set (1332-D frame+prev-milestone+state input already built
per frame). The best ensemble (big3+mixed6, fused with graph prior) predicts, per
sampled frame: next milestone (discrete) + subgoal latent. The subgoal is shown by
RETRIEVAL (nearest real frame to the predicted latent, excluding the query episode
-- canonical faithful decode). Columns per row:
  current frame | predicted next-subgoal -> nearest REAL frame | true next-medoid REAL frame
Green/red border on the prediction = predicted milestone matches the true next or not.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import split_indices  # noqa: E402
from eval_mean_variance import load_model  # noqa: E402
from train_dinov3h_decoder import load_features, l2, PooledDecoder  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=2481)
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--out_dir", default="lmwm/docs/assets", type=Path)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    R = 224

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    _, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    vi = set(vi.numpy().tolist())
    ep_all = z["episode_id"]
    idx = np.array([i for i in np.where(ep_all == args.episode)[0] if i in vi])
    assert len(idx) > args.rows, f"episode {args.episode} not enough held-out rows"
    idx = idx[np.argsort(z["t"][idx])]

    Xe = z["current"][idx].astype(np.float32)
    fut_m = z["future_milestone"][idx].astype(np.int64)
    cur_m = z["current_milestone"][idx].astype(np.int64)
    med = z["next_medoid"][idx].astype(np.float32)
    ts = z["t"][idx].astype(np.int64)

    # ---- best ensemble: big3 + mixed6, fused with graph prior ----
    paths = (sorted(glob.glob("lmwm/checkpoints/stage3_augin_big/*/best.pt"))
             + sorted(glob.glob("lmwm/checkpoints/stage3_augin_ens/*/best.pt"))
             + sorted(glob.glob("lmwm/checkpoints/stage3_augin/*/best.pt"))[-1:]
             + sorted(glob.glob("lmwm/checkpoints/stage3_augin_tail/*cecvar*/best.pt")))
    models = [load_model(p, dev)[0] for p in paths]
    print(f"best model = {len(paths)}-member ensemble", flush=True)
    probs = None; protos = None
    with torch.no_grad():
        xb = torch.from_numpy(Xe).to(dev)
        for m in models:
            out = m(xb)
            p = F.softmax(out["greedy_logits"], -1).cpu().numpy()
            g = out["greedy_proto"].cpu().numpy()
            probs = p if probs is None else probs + p
            protos = g if protos is None else protos + g
    probs /= len(models)
    protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)

    g = np.load(args.graph_npz); proto_tbl = g["prototype_table"].astype(np.float32)
    trans = g["transition_probs"].astype(np.float64); trans = trans / trans.sum(1, keepdims=True).clip(1e-12)
    prior = trans[cur_m]
    lp = (1 - args.lam) * np.log(np.clip(probs, 1e-12, 1)) + args.lam * np.log(np.clip(prior, 1e-12, 1))
    pred_m = lp.argmax(1)

    # ---- retrieval bank (pooled features -> frames), exclude query episode ----
    E, FR, Fb = load_features(args.feature_dir)
    Fn = l2(Fb.astype(np.float32))
    keep = E != args.episode
    Fn_gpu = torch.from_numpy(Fn[keep]).to(dev); Ek, FRk = E[keep], FR[keep]
    protos_g = torch.from_numpy(protos.astype(np.float32)).to(dev)
    with torch.no_grad():
        retr = (protos_g @ Fn_gpu.T).argmax(1).cpu().numpy()

    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    caps: dict[int, cv2.VideoCapture] = {}
    def frame(ep, t):
        if ep not in caps:
            caps[ep] = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, im = caps[ep].read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if ok else np.zeros((R, R, 3), np.uint8)

    # true next-medoid real frame: nearest frame WITHIN query episode to medoid latent
    qloc = np.where(E == args.episode)[0]
    qFn = Fn[qloc]
    def med_frame(mvec):
        j = qloc[(qFn @ l2(mvec[None])[0]).argmax()]
        return frame(int(E[j]), int(FR[j]))

    sel = np.linspace(0, len(idx) - 1, args.rows).astype(int)
    cos_all = (protos * (med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8))).sum(1)
    match_all = pred_m == fut_m

    titles = ["current frame", "PRED next-subgoal -> real frame", "TRUE next-medoid (real)"]
    fig, axes = plt.subplots(len(sel), 3, figsize=(3 * 2.4, len(sel) * 2.4))
    if len(sel) == 1:
        axes = axes[None, :]
    for ri, k in enumerate(sel):
        cur = frame(args.episode, int(ts[k]))
        pf = frame(int(Ek[retr[k]]), int(FRk[retr[k]]))
        mf = med_frame(med[k])
        for ci, im in enumerate([cur, pf, mf]):
            a = axes[ri, ci]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
            if ri == 0:
                a.set_title(titles[ci], fontsize=9)
        good = bool(match_all[k])
        axes[ri, 0].set_ylabel(f"m{cur_m[k]}->m{fut_m[k]}\npred m{pred_m[k]} {'OK' if good else 'x'}\ncos={cos_all[k]:.2f}", fontsize=7)
        for s in axes[ri, 1].spines.values():
            s.set_color("#2ca02c" if good else "#d62728"); s.set_linewidth(3)
    acc = float(match_all.mean()); mc = float(cos_all.mean())
    fig.suptitle(f"ep{args.episode} (HELD-OUT): best {len(paths)}-member ensemble | "
                 f"next-milestone acc={acc:.0%}  mean subgoal cos={mc:.3f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for c in caps.values():
        c.release()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"ep{args.episode}_best_holdout.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}")
    print(f"ep{args.episode}: {len(idx)} held-out frames | next-milestone top1={acc:.3f} | "
          f"retrieved-milestone match={float((np.array([(Fn[keep][retr[k]:retr[k]+1] @ proto_tbl.T).argmax() for k in range(len(idx))]) == fut_m).mean()):.3f} | "
          f"mean subgoal cos={mc:.4f}")


if __name__ == "__main__":
    main()
