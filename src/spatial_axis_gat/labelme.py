from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from .data import image_to_spatial


def load_labelme_json(path: str | Path) -> dict:
    with Path(path).open("r") as handle:
        return json.load(handle)


def shape_points_to_spatial(shape: dict, transform: dict[str, float]) -> np.ndarray:
    return image_to_spatial(np.asarray(shape["points"], dtype=float), transform)


def keypoints_from_shape(shape: dict, transform: dict[str, float]) -> np.ndarray:
    """Return all bottom-keypoint candidates encoded by a LabelMe shape.

    In this workflow, a `bottom_keypoint` LabelMe polygon is used as a compact
    way to mark several villus bottoms in one shape. Each vertex is a candidate
    bottom point, matching the original notebook convention.
    """
    return shape_points_to_spatial(shape, transform)


def cells_in_polygon(coords: np.ndarray, polygon_points: np.ndarray) -> np.ndarray:
    poly = Polygon(polygon_points)
    if not poly.is_valid or poly.area == 0:
        return np.array([], dtype=int)
    prepared = prep(poly)
    min_x, min_y, max_x, max_y = poly.bounds
    bbox_mask = (
        (coords[:, 0] >= min_x)
        & (coords[:, 0] <= max_x)
        & (coords[:, 1] >= min_y)
        & (coords[:, 1] <= max_y)
    )
    candidate_idx = np.flatnonzero(bbox_mask)
    inside = [prepared.covers(Point(coords[i, 0], coords[i, 1])) for i in candidate_idx]
    return candidate_idx[np.asarray(inside, dtype=bool)]


def parse_batch_annotations(
    *,
    json_path: str | Path,
    manifest_row: pd.Series,
    coords: np.ndarray,
    existing_epithelial: np.ndarray,
    batch: str,
    villus_labels: set[str],
    bottom_labels: set[str],
) -> pd.DataFrame:
    data = load_labelme_json(json_path)
    transform = {k: float(manifest_row[k]) for k in ["min_x", "min_y", "max_x", "max_y", "scale", "margin"]}

    bottom_points = []
    villus_shapes = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if label in bottom_labels:
            bottom_points.extend(keypoints_from_shape(shape, transform))
        elif label in villus_labels:
            villus_shapes.append(shape)

    if not bottom_points:
        raise ValueError(f"No bottom_keypoint shapes found in {json_path}")
    if not villus_shapes:
        raise ValueError(f"No villus polygons found in {json_path}")

    bottom_points = np.asarray(bottom_points, dtype=float)
    bottom_tree = cKDTree(bottom_points)
    rows = []
    for local_villus_id, shape in enumerate(villus_shapes):
        polygon_points = shape_points_to_spatial(shape, transform)
        local_cell_idx = cells_in_polygon(coords, polygon_points)
        if len(local_cell_idx) == 0:
            continue
        _, nearest_bottom_idx = bottom_tree.query(coords[local_cell_idx], k=1)
        nearest_bottom_counts = np.bincount(np.asarray(nearest_bottom_idx, dtype=int), minlength=len(bottom_points))
        bottom = bottom_points[int(nearest_bottom_counts.argmax())]
        distances = np.linalg.norm(coords[local_cell_idx] - bottom, axis=1)
        max_distance = float(distances.max())
        crypt_label = distances / max_distance if max_distance > 0 else np.zeros_like(distances)

        epi = np.asarray(existing_epithelial[local_cell_idx], dtype=float)
        epi_min = np.nanmin(epi)
        epi_max = np.nanmax(epi)
        epi_label = (epi - epi_min) / (epi_max - epi_min) if epi_max > epi_min else np.zeros_like(epi)

        villus_id = f"{batch}__villus_{local_villus_id:03d}"
        for cell_idx, crypt, epi_value in zip(local_cell_idx, crypt_label, epi_label, strict=False):
            rows.append(
                {
                    "batch": batch,
                    "local_cell_index": int(cell_idx),
                    "training_villus_id": villus_id,
                    "crypt_villus_label": float(crypt),
                    "epithelial_distance_label": float(epi_value),
                }
            )

    if not rows:
        raise ValueError(f"Villus annotations in {json_path} did not contain any cells")
    return pd.DataFrame(rows)
