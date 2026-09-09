"""Frozen BinaryGAT architecture; preprocessing is recorded in its checkpoint."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class BinaryGAT(nn.Module):
    def __init__(self, *, in_features, hidden_features, n_heads, dropout):
        super().__init__()
        self.dropout = float(dropout)
        self.conv1 = GATConv(in_features, hidden_features, heads=n_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_features * n_heads, hidden_features, heads=1, dropout=dropout)
        self.head = nn.Linear(hidden_features, 1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(F.elu(self.conv2(x, edge_index))).squeeze(-1)
