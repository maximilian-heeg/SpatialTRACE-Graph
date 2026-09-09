from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def read_adata(path: str | Path, backed: str | None = None) -> ad.AnnData:
    return ad.read_h5ad(path, backed=backed)


def choose_batches(obs: pd.DataFrame, batch_key: str, requested: list[str] | None) -> list[str]:
    if requested:
        return list(requested)
    values = obs[batch_key]
    if hasattr(values, "cat"):
        return list(values.cat.categories)
    return sorted(map(str, pd.unique(values)))


def categorical_or_values(series: pd.Series) -> np.ndarray:
    if hasattr(series, "cat"):
        return series.astype(str).to_numpy()
    return series.to_numpy()


def sample_indices(indices: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices)
    if len(indices) <= max_points:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_points, replace=False))


def scale_spatial_to_image(coords: np.ndarray, width: int, margin: int = 20) -> tuple[np.ndarray, dict[str, float]]:
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = (width - 2 * margin) / max(span_x, span_y)
    height = int(np.ceil(span_y * scale + 2 * margin))
    pixels = np.empty_like(coords, dtype=float)
    pixels[:, 0] = (coords[:, 0] - min_x) * scale + margin
    pixels[:, 1] = (coords[:, 1] - min_y) * scale + margin
    transform = {
        "min_x": float(min_x),
        "min_y": float(min_y),
        "max_x": float(max_x),
        "max_y": float(max_y),
        "scale": float(scale),
        "margin": float(margin),
        "image_width": float(width),
        "image_height": float(height),
    }
    return pixels, transform


def image_to_spatial(points: np.ndarray, transform: dict[str, float]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    spatial = np.empty_like(points, dtype=float)
    spatial[:, 0] = (points[:, 0] - transform["margin"]) / transform["scale"] + transform["min_x"]
    spatial[:, 1] = (points[:, 1] - transform["margin"]) / transform["scale"] + transform["min_y"]
    return spatial

