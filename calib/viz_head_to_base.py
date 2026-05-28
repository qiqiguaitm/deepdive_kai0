"""Compose T_baseL_camF / T_baseR_camF from calibration.yml and visualize.

Run:
    .venv/bin/python calib/viz_head_to_base.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


REPO = Path(__file__).resolve().parents[1]
CALIB_YML = REPO / "config" / "calibration.yml"
OUT_PNG = REPO / "calib" / "data" / "head_to_base_viz.png"


def draw_frame(ax, T: np.ndarray, label: str, scale: float = 0.1) -> None:
    """Draw a coordinate frame at pose T (4x4) with RGB = X/Y/Z axes."""
    origin = T[:3, 3]
    R = T[:3, :3]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]  # X red, Y green, Z blue
    for i, c in enumerate(colors):
        end = origin + R[:, i] * scale
        ax.plot(*zip(origin, end), color=c, linewidth=2.5)
    ax.scatter(*origin, color="black", s=25)
    ax.text(origin[0] + 0.02, origin[1] + 0.02, origin[2] + 0.02, label, fontsize=11, weight="bold")


def main() -> None:
    with CALIB_YML.open() as f:
        d = yaml.safe_load(f)["transforms"]
    T_world_camF = np.array(d["T_world_camF"])
    T_world_baseL = np.array(d["T_world_baseL"])
    T_world_baseR = np.array(d["T_world_baseR"])

    T_baseL_camF = np.linalg.inv(T_world_baseL) @ T_world_camF
    T_baseR_camF = np.linalg.inv(T_world_baseR) @ T_world_camF

    np.set_printoptions(precision=4, suppress=True)
    print("== T_baseL_camF (head cam in left base frame) ==")
    print(T_baseL_camF)
    print("== T_baseR_camF (head cam in right base frame) ==")
    print(T_baseR_camF)

    # Round-trip sanity: T_world_baseL @ T_baseL_camF should equal T_world_camF.
    err_L = np.linalg.norm(T_world_baseL @ T_baseL_camF - T_world_camF)
    err_R = np.linalg.norm(T_world_baseR @ T_baseR_camF - T_world_camF)
    print(f"\nRound-trip error  L: {err_L:.2e}   R: {err_R:.2e}  (both should be ~1e-15)")

    # ---- Visualization: draw 3 frames in world coordinates ----
    fig = plt.figure(figsize=(13, 6))

    # Left: world frame view
    ax1 = fig.add_subplot(121, projection="3d")
    draw_frame(ax1, np.eye(4), "world", scale=0.08)
    draw_frame(ax1, T_world_baseL, "baseL", scale=0.10)
    draw_frame(ax1, T_world_baseR, "baseR", scale=0.10)
    draw_frame(ax1, T_world_camF, "camF (head)", scale=0.10)
    # connecting lines (origin-to-origin) for visual reference
    for T, color in [(T_world_baseL, "#9b59b6"), (T_world_baseR, "#f39c12")]:
        ax1.plot(
            *zip(T[:3, 3], T_world_camF[:3, 3]),
            linestyle="--",
            color=color,
            alpha=0.6,
            linewidth=1.2,
        )
    ax1.set_title("World frame view\n(R=X G=Y B=Z)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    _equal_axes(ax1, np.stack([T_world_baseL[:3, 3], T_world_baseR[:3, 3], T_world_camF[:3, 3], [0, 0, 0]]))

    # Right: baseL frame view — using composed T_baseL_camF directly
    ax2 = fig.add_subplot(122, projection="3d")
    T_baseL_baseR = np.linalg.inv(T_world_baseL) @ T_world_baseR
    draw_frame(ax2, np.eye(4), "baseL", scale=0.10)
    draw_frame(ax2, T_baseL_baseR, "baseR", scale=0.10)
    draw_frame(ax2, T_baseL_camF, "camF (head)", scale=0.10)
    ax2.plot(
        *zip([0, 0, 0], T_baseL_camF[:3, 3]),
        linestyle="--",
        color="#9b59b6",
        alpha=0.6,
        linewidth=1.2,
    )
    ax2.plot(
        *zip(T_baseL_baseR[:3, 3], T_baseL_camF[:3, 3]),
        linestyle="--",
        color="#f39c12",
        alpha=0.6,
        linewidth=1.2,
    )
    t_camF = T_baseL_camF[:3, 3]
    ax2.set_title(
        f"baseL frame view (composed)\ncamF @ baseL: ({t_camF[0]:+.3f}, {t_camF[1]:+.3f}, {t_camF[2]:+.3f}) m"
    )
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    _equal_axes(ax2, np.stack([[0, 0, 0], T_baseL_baseR[:3, 3], T_baseL_camF[:3, 3]]))

    plt.suptitle("Head camera to dual arm base — left: raw world transforms,  right: composed T_baseL_*", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")


def _equal_axes(ax, pts: np.ndarray) -> None:
    """Make 3D axes have equal scale around the given points."""
    mn = pts.min(axis=0) - 0.15
    mx = pts.max(axis=0) + 0.15
    span = (mx - mn).max()
    mid = (mn + mx) / 2
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)


if __name__ == "__main__":
    main()
