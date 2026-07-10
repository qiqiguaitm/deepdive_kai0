#!/usr/bin/env python
"""Visualize the LaWM-style patch inverse/forward FUTURE-grid decode on one episode.

Trains inverse/forward (code_dim=32, LaWM default) + CRAVE patch decoder, then for
a chosen episode's stage transitions shows, per row:
  current frame | reconstructed-future decoded | true-future decoded | real future frame
where reconstructed-future = decode(forward(g_t, inverse(g_t, g_future))). The code
comes from the true future (reconstruction/oracle mode, = LaWM's cos_sim view).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from train_lawm_patch import InverseEnc, ForwardDec, load_index, read_imgs  # noqa: E402
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--code_dim", type=int, default=32)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out_dir", default="lmwm/docs/assets", type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    dev = "cuda"

    E, FR, Fn = load_index(args.feature_dir)
    proto = np.load("lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")["prototype_table"].astype(np.float32)
    rng = np.random.default_rng(args.seed); eps = np.unique(E); rng.shuffle(eps)
    val_eps = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())

    # train transitions (exclude target ep) + the target episode transitions
    tr_pairs = []
    for ep in eps:
        if ep == args.episode or ep in val_eps:
            continue
        tr_pairs.extend([(a, b) for a, b, _, _ in stage_transitions(E, FR, Fn, proto, ep)])
    rng.shuffle(tr_pairs); tr_pairs = tr_pairs[:args.n_train]
    ep_trans = stage_transitions(E, FR, Fn, proto, args.episode)

    uniq = sorted(set([g for p in tr_pairs for g in p] + [g for t in ep_trans for g in (t[0], t[1])]))
    u2k = {g: k for k, g in enumerate(uniq)}
    print(f"{len(tr_pairs)} train transitions, ep{args.episode} {len(ep_trans)} transitions, {len(uniq)} frames", flush=True)
    enc_imgs, tgt_imgs = read_imgs(args.dataset_root, args.camera, E, FR, np.array(uniq), 256, 128)
    enc = load_encoder("dinov3-h", device=dev)
    grids = enc.encode_grid(enc_imgs).astype(np.float32)
    din = grids.shape[1]; gmu, gsd = grids.mean(), grids.std() + 1e-6
    gz = ((grids - gmu) / gsd).astype(np.float32)

    print("training patch decoder + inverse/forward ...", flush=True)
    decode = train_dec(grids, tgt_imgs, din, dec="small", epochs=50, device=dev)
    GZ = torch.from_numpy(gz)
    tra = torch.from_numpy(np.array([u2k[c] for c, _ in tr_pairs]))
    trb = torch.from_numpy(np.array([u2k[n] for _, n in tr_pairs]))
    inv = InverseEnc(din, args.code_dim).to(dev); fwd = ForwardDec(din, args.code_dim).to(dev)
    opt = torch.optim.AdamW(list(inv.parameters()) + list(fwd.parameters()), lr=2e-4, weight_decay=1e-5)
    for step in range(args.steps):
        sel = torch.randint(0, len(tra), (32,))
        gt = GZ[tra[sel]].to(dev); gf = GZ[trb[sel]].to(dev)
        loss = F.smooth_l1_loss(fwd(gt, inv(gt, gf)), gf, beta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    inv.eval(); fwd.eval()

    def dec_grid(g_std):
        return decode((g_std * gsd + gmu).astype(np.float32))[0]

    rows = []
    for cg, ng, cm, nm in ep_trans:
        gt = GZ[u2k[cg]:u2k[cg]+1].to(dev); gf = GZ[u2k[ng]:u2k[ng]+1].to(dev)
        with torch.no_grad():
            gf_hat = fwd(gt, inv(gt, gf)).cpu().numpy()
        rows.append({
            "cur": tgt_imgs[u2k[cg]],
            "recon": dec_grid(gf_hat),                       # forward-reconstructed future -> decode
            "true_dec": dec_grid(gz[u2k[ng]:u2k[ng]+1]),     # true future grid -> decode (ceiling)
            "real": tgt_imgs[u2k[ng]],
            "cm": cm, "nm": nm,
        })
    sel = rows if len(rows) <= 10 else [rows[i] for i in np.linspace(0, len(rows)-1, 10).astype(int)]
    titles = ["current", "recon future (forward)", "true future (decode)", "real future"]
    fig, ax = plt.subplots(len(sel), 4, figsize=(4*2.1, len(sel)*2.2))
    if len(sel) == 1: ax = ax[None, :]
    for ri, r in enumerate(sel):
        for ci, im in enumerate([r["cur"], r["recon"], r["true_dec"], r["real"]]):
            ax[ri, ci].imshow(im); ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
            if ri == 0: ax[ri, ci].set_title(titles[ci], fontsize=8)
        ax[ri, 0].set_ylabel(f"m{r['cm']}->m{r['nm']}", fontsize=7)
    fig.suptitle(f"ep{args.episode}: LaWM-patch future-grid decode (code_dim={args.code_dim})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out_dir / f"ep{args.episode}_lawm_patch_decode.png"
    fig.savefig(out, dpi=115); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
