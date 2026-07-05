#!/usr/bin/env python
"""Visualize how the CURRENT CRAVE method processes our dataset (kai0_base): the 37-milestone
recurrence graph applied per episode -> milestone assignment + forward-Viterbi progress/value.

Shows the effect + quality:
  (1) corr(value, normalized-time) + monotonicity distribution across ALL episodes (the label quality)
  (2) example episodes: value curve vs time + per-frame milestone (raw argmax vs Viterbi-smoothed)
  (3) milestone gallery: medoid frame of each of the 37 milestones (what CRAVE discovered)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
from train_lawm_patch import load_index, read_imgs  # noqa: E402
from crave.utils import med, smooth_monotone, viterbi_forward  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--out", default="lmwm/outputs/crave_effect", type=Path)
    ap.add_argument("--n_example", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    z = np.load(args.graph)
    proto = z["prototype_table"].astype(np.float32); pord = z["pord"].astype(np.float32); M = len(proto)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    E, FR, Fn = load_index(args.feature_dir)                       # Fn L2-normed pooled feats
    eps = np.unique(E)
    print(f"CRAVE graph: {M} milestones; {len(eps)} episodes, {len(E)} frames", flush=True)

    # per-episode: milestone assign + Viterbi value + quality
    corrs, monos, nseg_raw, nseg_vit = [], [], [], []
    per_ep = {}
    for e in eps:
        fi = np.where(E == e)[0]; fi = fi[np.argsort(FR[fi])]
        Fq = Fn[fi]
        emit = np.linalg.norm(Fq[:, None] - protoL[None], axis=2)  # (n, M)
        raw = emit.argmin(1)
        ms = viterbi_forward(emit, pord, up=3.0, down=25.0, hard_start=True)
        val = smooth_monotone(med(pord[ms], 5), fps=3.0)
        tq = np.arange(len(fi)) / max(1, len(fi) - 1)
        if val.std() > 1e-6:
            corrs.append(float(np.corrcoef(val, tq)[0, 1]))
        monos.append(float(np.mean(np.diff(val) >= -1e-6)))
        nseg_raw.append(int((np.diff(raw) != 0).sum() + 1)); nseg_vit.append(int((np.diff(ms) != 0).sum() + 1))
        per_ep[int(e)] = (fi, raw, ms, val, tq)
    corrs = np.array(corrs); monos = np.array(monos)
    summary = {"n_milestones": M, "n_eps": len(eps),
               "corr_mean": round(float(corrs.mean()), 3), "corr_median": round(float(np.median(corrs)), 3),
               "corr_p25": round(float(np.percentile(corrs, 25)), 3), "frac_corr_ge_0.7": round(float(np.mean(corrs >= 0.7)), 3),
               "mono_mean": round(float(monos.mean()), 3),
               "raw_segments_per_ep_median": int(np.median(nseg_raw)), "viterbi_segments_per_ep_median": int(np.median(nseg_vit))}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    # (1) quality distributions
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    ax[0].hist(corrs, bins=30, color="#4477aa"); ax[0].axvline(np.median(corrs), color="r", ls="--")
    ax[0].set_title(f"corr(value, time): median {np.median(corrs):.2f}, %>=0.7 {np.mean(corrs>=0.7):.0%}"); ax[0].set_xlabel("corr")
    ax[1].hist(monos, bins=30, color="#66aa55"); ax[1].set_title(f"monotonicity: mean {monos.mean():.2f}"); ax[1].set_xlabel("frac non-decreasing")
    ax[2].hist(nseg_raw, bins=30, alpha=0.6, label=f"raw argmax (med {int(np.median(nseg_raw))})")
    ax[2].hist(nseg_vit, bins=30, alpha=0.6, label=f"Viterbi (med {int(np.median(nseg_vit))})")
    ax[2].set_title("milestone segments / episode"); ax[2].set_xlabel("# segments"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.out / "quality.png", dpi=120); plt.close(fig)

    # (2) example episodes: value vs time + milestone assignment
    rng = np.random.default_rng(0)
    ex = sorted(rng.choice(list(per_ep), min(args.n_example, len(per_ep)), replace=False).tolist())
    fig, ax = plt.subplots(len(ex), 1, figsize=(9, 2.2 * len(ex)))
    if len(ex) == 1:
        ax = [ax]
    for a, e in zip(ax, ex):
        fi, raw, ms, val, tq = per_ep[e]
        a.plot(tq, val, "b-", lw=2, label="CRAVE value")
        a.plot(tq, tq, "k:", lw=1, label="ideal (=time)")
        a2 = a.twinx(); a2.plot(tq, pord[raw], ".", ms=2, color="orange", alpha=0.4, label="raw milestone prog")
        a2.plot(tq, pord[ms], "-", color="green", lw=1, alpha=0.7, label="Viterbi milestone prog")
        a.set_title(f"ep{e}: corr={np.corrcoef(val,tq)[0,1]:.2f}", fontsize=9); a.set_ylabel("value"); a.legend(fontsize=6, loc="upper left")
    fig.tight_layout(); fig.savefig(args.out / "examples.png", dpi=120); plt.close(fig)

    # (3) milestone gallery: medoid frame of each milestone (by pord order)
    med_gidx = []
    for m in range(M):
        sims = Fn @ protoL[m]; med_gidx.append(int(np.argmax(sims)))
    imgs, _ = read_imgs(args.dataset_root, args.camera, E, FR, np.array(med_gidx), 160, 160)
    ordm = np.argsort(pord)
    cols = 8; rows = int(np.ceil(M / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6))
    ax = np.array(ax).reshape(-1)
    for i in range(rows * cols):
        ax[i].set_xticks([]); ax[i].set_yticks([])
        if i < M:
            m = ordm[i]; ax[i].imshow(imgs[m]); ax[i].set_title(f"m{m} p={pord[m]:.2f}", fontsize=6)
        else:
            ax[i].axis("off")
    fig.suptitle(f"CRAVE {M} milestones (kai0_base), ordered by progress", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(args.out / "milestone_gallery.png", dpi=120); plt.close(fig)
    print(f"wrote quality.png, examples.png, milestone_gallery.png to {args.out}", flush=True)


if __name__ == "__main__":
    main()
