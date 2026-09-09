from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def spatial_scatter(
    ax: plt.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cmap: str = "viridis",
    s: float = 0.2,
    vmin: float | None = 0,
    vmax: float | None = 1,
) -> None:
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=s, linewidths=0, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)


def save_figure(fig: plt.Figure, output_path, dpi: int = 300) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

