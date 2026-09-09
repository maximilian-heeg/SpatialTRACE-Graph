import torch

from spatial_axis_gat.model import GATModel


def test_gat_model_forward_shape():
    model = GATModel(in_features=3, hidden_features=4, n_heads=2, dropout=0.0)
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    y = model(x, edge_index)
    assert tuple(y.shape) == (5,)
    assert torch.all((0 <= y) & (y <= 1))
