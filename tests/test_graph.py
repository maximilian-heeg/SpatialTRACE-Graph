import numpy as np

from spatial_axis_gat.graph import build_batch_spatial_edge_index


def test_build_batch_spatial_edge_index_stays_within_batch():
    coords = np.array([[0, 0], [1, 0], [10, 0], [11, 0]], dtype=float)
    batches = np.array(["a", "a", "b", "b"])
    edge_index = build_batch_spatial_edge_index(coords, batches, n_neighbors=1)
    edges = set(map(tuple, edge_index.numpy().T.tolist()))
    assert (0, 2) not in edges
    assert (2, 0) not in edges
    assert edge_index.shape[0] == 2
