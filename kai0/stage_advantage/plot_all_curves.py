"""
Plot all episode curves from eval_adv_est_out/inference_results.pkl.
Each curve is labeled positive (blue) or negative (red).

Usage (from kai0/):
    uv run python stage_advantage/plot_all_curves.py [--out eval_adv_est_out]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent


def plot_all(pos_results: dict, neg_results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    POS_COLOR = "#2166ac"   # blue
    NEG_COLOR = "#d6604d"   # red
    ALPHA = 0.35
    LW = 0.9

    # ------------------------------------------------------------------ #
    # 1. absolute_value — all curves on one axes
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(22, 9))

    for ep_idx, results in neg_results.items():
        if not results:
            continue
        fi = np.array([r["frame_idx"] for r in results], dtype=float)
        av = np.array([r["absolute_value"] for r in results])
        x = (fi - fi[0]) / max(fi[-1] - fi[0], 1)
        ax.plot(x, av, color=NEG_COLOR, alpha=ALPHA, linewidth=LW)

    for ep_idx, results in pos_results.items():
        if not results:
            continue
        fi = np.array([r["frame_idx"] for r in results], dtype=float)
        av = np.array([r["absolute_value"] for r in results])
        x = (fi - fi[0]) / max(fi[-1] - fi[0], 1)
        ax.plot(x, av, color=POS_COLOR, alpha=ALPHA, linewidth=LW)

    # Mean curves
    xs = np.linspace(0, 1, 200)
    for results_dict, color, style in [
        (neg_results, NEG_COLOR, "--"),
        (pos_results, POS_COLOR, "-"),
    ]:
        curves = []
        for results in results_dict.values():
            if not results:
                continue
            fi = np.array([r["frame_idx"] for r in results], dtype=float)
            x = (fi - fi[0]) / max(fi[-1] - fi[0], 1)
            av = np.array([r["absolute_value"] for r in results])
            curves.append(np.interp(xs, x, av))
        if curves:
            mean_c = np.mean(curves, axis=0)
            ax.plot(xs, mean_c, color=color, linewidth=3.0, linestyle=style,
                    zorder=10, alpha=0.95)

    legend_handles = [
        mlines.Line2D([], [], color=POS_COLOR, linewidth=2.5, label=f"Positive (n={len(pos_results)})"),
        mlines.Line2D([], [], color=NEG_COLOR, linewidth=2.5, label=f"Negative (n={len(neg_results)})"),
        mlines.Line2D([], [], color=POS_COLOR, linewidth=3, linestyle="-",  label="Positive mean"),
        mlines.Line2D([], [], color=NEG_COLOR, linewidth=3, linestyle="--", label="Negative mean"),
    ]
    ax.legend(handles=legend_handles, fontsize=12, loc="upper left")
    ax.set_xlabel("Normalized frame position  (0 = start, 1 = end)", fontsize=12)
    ax.set_ylabel("absolute_value  (predicted cumulative progress)", fontsize=12)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"adv_est_v1 — absolute_value  ·  all {len(pos_results)+len(neg_results)} episodes\n"
        f"(frame_interval=10, blue=positive, red=negative, thick=mean)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    p = out_dir / "all_curves_absolute_value.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[plot] {p}")

    # ------------------------------------------------------------------ #
    # 2. relative_advantage — all curves
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(figsize=(22, 9))

    for ep_idx, results in neg_results.items():
        if not results:
            continue
        fi = np.array([r["frame_idx"] for r in results], dtype=float)
        ra = np.array([r["relative_advantage"] for r in results])
        x = (fi - fi[0]) / max(fi[-1] - fi[0], 1)
        ax2.plot(x, ra, color=NEG_COLOR, alpha=ALPHA, linewidth=LW)

    for ep_idx, results in pos_results.items():
        if not results:
            continue
        fi = np.array([r["frame_idx"] for r in results], dtype=float)
        ra = np.array([r["relative_advantage"] for r in results])
        x = (fi - fi[0]) / max(fi[-1] - fi[0], 1)
        ax2.plot(x, ra, color=POS_COLOR, alpha=ALPHA, linewidth=LW)

    for results_dict, color, style in [
        (neg_results, NEG_COLOR, "--"),
        (pos_results, POS_COLOR, "-"),
    ]:
        curves = []
        for results in results_dict.values():
            if not results:
                continue
            fi = np.array([r["frame_idx"] for r in results], dtype=float)
            x = (fi - fi[0]) / max(fi[-1] - fi[0], 1)
            ra = np.array([r["relative_advantage"] for r in results])
            curves.append(np.interp(xs, x, ra))
        if curves:
            ax2.plot(xs, np.mean(curves, axis=0), color=color, linewidth=3.0,
                     linestyle=style, zorder=10, alpha=0.95)

    ax2.legend(handles=legend_handles, fontsize=12, loc="upper right")
    ax2.set_xlabel("Normalized frame position", fontsize=12)
    ax2.set_ylabel("relative_advantage  (Δprogress over 50 frames)", fontsize=12)
    ax2.set_ylim(-1.05, 1.05)
    ax2.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax2.grid(True, alpha=0.25)
    ax2.set_title(
        f"adv_est_v1 — relative_advantage  ·  all {len(pos_results)+len(neg_results)} episodes",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    p2 = out_dir / "all_curves_relative_advantage.png"
    plt.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[plot] {p2}")

    # ------------------------------------------------------------------ #
    # 3. 2×2 grid: top=pos, bottom=neg, left=absolute_value, right=relative_advantage
    # ------------------------------------------------------------------ #
    fig3, axes3 = plt.subplots(2, 2, figsize=(24, 14), sharex=True)
    pairs = [
        (axes3[0, 0], pos_results, POS_COLOR, "absolute_value",     "Positive — absolute_value"),
        (axes3[0, 1], pos_results, POS_COLOR, "relative_advantage",  "Positive — relative_advantage"),
        (axes3[1, 0], neg_results, NEG_COLOR, "absolute_value",     "Negative — absolute_value"),
        (axes3[1, 1], neg_results, NEG_COLOR, "relative_advantage",  "Negative — relative_advantage"),
    ]
    for ax, results_dict, color, key, title in pairs:
        curves = []
        for ep_idx, results in results_dict.items():
            if not results:
                continue
            fi = np.array([r["frame_idx"] for r in results], dtype=float)
            vals = np.array([r[key] for r in results])
            x = (fi - fi[0]) / max(fi[-1] - fi[0], 1)
            ax.plot(x, vals, color=color, alpha=0.28, linewidth=0.8)
            curves.append(np.interp(xs, x, vals))
        if curves:
            mean_c = np.mean(curves, axis=0)
            p25 = np.percentile(curves, 25, axis=0)
            p75 = np.percentile(curves, 75, axis=0)
            ax.plot(xs, mean_c, color="black", linewidth=2.5, label="mean", zorder=10)
            ax.fill_between(xs, p25, p75, color=color, alpha=0.2, label="IQR [25–75]")
        ax.set_title(f"{title}  (n={len([v for v in results_dict.values() if v])})", fontsize=11)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    for ax in axes3[1]:
        ax.set_xlabel("Normalized frame position", fontsize=10)
    for ax in axes3[:, 0]:
        ax.set_ylabel("predicted value", fontsize=10)

    fig3.suptitle(
        "adv_est_v1 — all episode curves  (frame_interval=10)\n"
        "Blue = Positive (task_index=1)   Red = Negative (task_index=0)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    p3 = out_dir / "all_curves_2x2_grid.png"
    plt.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[plot] {p3}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="eval_adv_est_out")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    pkl = out_dir / "inference_results.pkl"
    if not pkl.exists():
        print(f"[error] {pkl} not found. Run eval_adv_est.py first.")
        sys.exit(1)

    with open(pkl, "rb") as f:
        data = pickle.load(f)

    pos_results = data["positive"]
    neg_results = data["negative"]
    print(f"[load] pos={len(pos_results)} neg={len(neg_results)} episodes")

    plot_all(pos_results, neg_results, out_dir)
    print("[done]")


if __name__ == "__main__":
    main()
