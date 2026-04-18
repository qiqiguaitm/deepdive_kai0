"""
Plot one subplot per episode, labeled POS/NEG.
Outputs:
  - per_episode_grid.png  : all episodes in one large grid
  - per_episode_grid.pdf  : same, PDF (zoomable)
  - per_episode/ep_XXXXXX_[POS|NEG].png : individual episode PNGs

Usage (from kai0/):
    uv run python stage_advantage/plot_per_episode.py [--out eval_adv_est_out] [--no-individual]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent

POS_COLOR  = "#2166ac"
NEG_COLOR  = "#d6604d"
POS_BG     = "#ddeeff"
NEG_BG     = "#ffeeee"


def load_data(out_dir: Path):
    with open(out_dir / "inference_results.pkl", "rb") as f:
        d = pickle.load(f)
    return d["positive"], d["negative"]


def episode_arrays(results: list):
    fi  = np.array([r["frame_idx"]          for r in results], dtype=float)
    av  = np.array([r["absolute_value"]      for r in results])
    ra  = np.array([r["relative_advantage"]  for r in results])
    aa  = np.array([r["absolute_advantage"]  for r in results])
    x   = (fi - fi[0]) / max(fi[-1] - fi[0], 1.0)
    return x, av, ra, aa


def draw_subplot(ax, results, ep_idx, label, color, bg):
    """Draw one episode curve into ax."""
    ax.set_facecolor(bg)
    if not results:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                fontsize=6, transform=ax.transAxes)
        ax.set_title(f"ep {ep_idx}\n[{label}]", fontsize=6, color=color)
        return

    x, av, ra, aa = episode_arrays(results)

    ax.plot(x, av, color=color,     linewidth=1.2, label="abs_val")
    ax.plot(x, ra, color="gray",    linewidth=0.7, alpha=0.7, linestyle="--", label="rel_adv")
    ax.axhline(0, color="black", linewidth=0.4, linestyle=":")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=4, length=2, pad=1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([-1, 0, 1])

    # Spearman r as annotation
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x, av)
    mono   = np.mean(np.diff(av) >= 0)
    ax.set_title(
        f"ep {ep_idx}  [{label}]\nρ={rho:.2f}  ↑{mono:.2f}",
        fontsize=5.5, color=color, pad=2
    )


# ------------------------------------------------------------------ #
# Grid plot (all episodes)
# ------------------------------------------------------------------ #

def plot_grid(pos_results, neg_results, out_dir: Path):
    # Sort: positive first (sorted by ep_idx), then negative
    pos_items = sorted(pos_results.items())
    neg_items = sorted(neg_results.items())
    all_items = [(ep, res, "POS", POS_COLOR, POS_BG) for ep, res in pos_items] + \
                [(ep, res, "NEG", NEG_COLOR, NEG_BG) for ep, res in neg_items]

    n = len(all_items)
    ncols = 15
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.2, nrows * 2.5),
        squeeze=False,
    )

    for i, (ep_idx, results, label, color, bg) in enumerate(all_items):
        r, c = divmod(i, ncols)
        draw_subplot(axes[r][c], results, ep_idx, label, color, bg)

    # Hide unused cells
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"adv_est_v1 — per-episode curves  "
        f"(blue bg = POS n={len(pos_items)}, red bg = NEG n={len(neg_items)})\n"
        "solid = absolute_value, dashed gray = relative_advantage  |  "
        "ρ = Spearman r vs frame  |  ↑ = monotonicity",
        fontsize=10, fontweight="bold", y=1.002
    )
    plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.3)

    png_path = out_dir / "per_episode_grid.png"
    pdf_path = out_dir / "per_episode_grid.pdf"

    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    print(f"[plot] {png_path}")

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    print(f"[plot] {pdf_path}")

    plt.close(fig)


# ------------------------------------------------------------------ #
# Individual episode PNGs
# ------------------------------------------------------------------ #

def plot_individuals(pos_results, neg_results, out_dir: Path):
    ind_dir = out_dir / "per_episode"
    ind_dir.mkdir(exist_ok=True)

    all_items = \
        [(ep, res, "POS", POS_COLOR, POS_BG) for ep, res in sorted(pos_results.items())] + \
        [(ep, res, "NEG", NEG_COLOR, NEG_BG) for ep, res in sorted(neg_results.items())]

    from tqdm import tqdm
    for ep_idx, results, label, color, bg in tqdm(all_items, desc="individual plots"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

        # Left: absolute_value
        for ax, key, ylabel in [
            (axes[0], "absolute_value",    "absolute_value"),
            (axes[1], "relative_advantage","relative_advantage"),
        ]:
            ax.set_facecolor(bg)
            if results:
                x, av, ra, aa = episode_arrays(results)
                vals = av if key == "absolute_value" else ra
                ax.plot(x, vals, color=color, linewidth=1.6)
                ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Normalized frame position", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)

        if results:
            x, av, ra, aa = episode_arrays(results)
            from scipy.stats import spearmanr
            rho, _ = spearmanr(x, av)
            mono = np.mean(np.diff(av) >= 0)
            n_frames = len(results)
            fig.suptitle(
                f"Episode {ep_idx}  [{label}]   "
                f"frames={n_frames}  Spearman ρ={rho:.3f}  monotonicity={mono:.3f}",
                fontsize=11, fontweight="bold", color=color
            )
        else:
            fig.suptitle(f"Episode {ep_idx}  [{label}]  (no data)", fontsize=11, color=color)

        plt.tight_layout()
        fname = ind_dir / f"ep_{ep_idx:06d}_{label}.png"
        fig.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"[plot] {len(all_items)} individual PNGs → {ind_dir}/")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="eval_adv_est_out")
    parser.add_argument("--no-individual", action="store_true",
                        help="Skip individual per-episode PNGs")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    pos_results, neg_results = load_data(out_dir)
    print(f"[load] pos={len(pos_results)}  neg={len(neg_results)}")

    print("[grid] generating per-episode grid...")
    plot_grid(pos_results, neg_results, out_dir)

    if not args.no_individual:
        print("[individual] generating per-episode PNGs...")
        plot_individuals(pos_results, neg_results, out_dir)

    print("[done]")


if __name__ == "__main__":
    main()
