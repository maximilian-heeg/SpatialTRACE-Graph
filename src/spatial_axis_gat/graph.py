from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import coalesce


def build_batch_spatial_edge_index(
    coords: np.ndarray,
    batches: np.ndarray,
    *,
    n_neighbors: int,
    neighbor_policy: str = "knn",
) -> torch.Tensor:
    coords = np.asarray(coords, dtype=float)
    batches = np.asarray(batches).astype(str)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(batches):
        raise ValueError("Expected spatial coordinates (cells, 2) and one section per cell")
    if not np.isfinite(coords).all() or n_neighbors < 1:
        raise ValueError("Coordinates must be finite and n_neighbors positive")
    if neighbor_policy not in {"knn", "paper-v1", "paper-peyer-v1"}:
        raise ValueError("Unknown neighbor_policy")
    if neighbor_policy == 'paper-peyer-v1':
        # The original classifier rounded positions to float32 before querying.
        coords = coords.astype(np.float32)
    edge_chunks = []
    for batch in np.unique(batches):
        batch_idx = np.flatnonzero(batches == batch)
        if len(batch_idx) <= 1:
            continue
        count = len(batch_idx)
        if neighbor_policy == "paper-v1" and count <= n_neighbors + 1:
            raise ValueError("paper-v1 needs at least n_neighbors + 2 cells per section")
        k = min(n_neighbors, count - 1)
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(coords[batch_idx])
        if neighbor_policy == "paper-peyer-v1":
            from scipy.spatial import cKDTree
            _, neighbors = cKDTree(coords[batch_idx]).query(coords[batch_idx], k=k + 1)
            neighbors = neighbors[:, 1:]
        elif neighbor_policy == "paper-v1":
            # Exact frozen coordinate-model topology: sklearn excludes self
            # for X=None; the original pipeline additionally dropped rank 1.
            neighbors = nn.kneighbors(n_neighbors=k + 1, return_distance=False)[:, 1:]
        else:
            neighbors = nn.kneighbors(n_neighbors=k, return_distance=False)
        src = np.repeat(batch_idx, k)
        dst = batch_idx[neighbors.reshape(-1)]
        edge_chunks.append(np.vstack([src, dst]))
    if not edge_chunks:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = np.concatenate(edge_chunks, axis=1)
    reverse = edge_index[[1, 0], :]
    edge_index = np.concatenate([edge_index, reverse], axis=1)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    return coalesce(edge_index)


def load_or_build_edge_index(
    coords: np.ndarray,
    batches: np.ndarray,
    *,
    cache_path: str | Path,
    n_neighbors: int,
    rebuild: bool = False,
) -> torch.Tensor:
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        print(f"Loading cached edge_index from {cache_path}")
        return torch.load(cache_path, map_location="cpu")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    edge_index = build_batch_spatial_edge_index(coords, batches, n_neighbors=n_neighbors)
    torch.save(edge_index, cache_path)
    print(f"Wrote edge_index cache to {cache_path}: {tuple(edge_index.shape)}")
    return edge_index
